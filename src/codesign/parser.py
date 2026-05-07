"""Tree-sitter parser for AST, DFG, CFG and sink extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tree_sitter
    from tree_sitter_python import language as python_language
    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False

try:
    from tree_sitter_c import language as c_language
    TS_C_AVAILABLE = True
except ImportError:
    TS_C_AVAILABLE = False


SUPPORTED_LANGUAGES = ("python", "c")

# Identifiers we never want to rename. Builtins + common conventions.
PY_DENY = frozenset({
    "True", "False", "None", "self", "cls",
    "print", "os", "sys", "int", "str", "list", "dict", "set", "tuple",
    "len", "range", "open", "type", "input", "Exception",
})

# Heuristic sink list. Not exhaustive; conservative on purpose.
PY_DANGEROUS_CALLS = frozenset({
    "system", "popen", "exec", "eval",
    "execute", "executemany",
    "Popen", "subprocess.call", "subprocess.run",
    "compile", "loads", "load",
    "render_template_string", "format",
})


@dataclass(frozen=True)
class NodeWrapper:
    val: str
    type: str
    start_byte: int
    end_byte: int
    start_row: int
    end_row: int

    @classmethod
    def from_ts(cls, ts_node: Any, source: bytes) -> NodeWrapper:
        return cls(
            val=source[ts_node.start_byte:ts_node.end_byte].decode("utf-8", errors="replace"),
            type=ts_node.type,
            start_byte=ts_node.start_byte,
            end_byte=ts_node.end_byte,
            start_row=ts_node.start_point[0],
            end_row=ts_node.end_point[0],
        )


@dataclass
class ProgramGraph:
    source: str
    language: str
    variables: dict[str, list[NodeWrapper]] = field(default_factory=dict)
    dfg: dict[str, set[str]] = field(default_factory=dict)
    cfg_blocks: list[NodeWrapper] = field(default_factory=list)
    sinks: list[NodeWrapper] = field(default_factory=list)

    @property
    def all_variable_nodes(self) -> list[NodeWrapper]:
        return [n for nodes in self.variables.values() for n in nodes]

    def variable_byte_ranges(self) -> list[tuple[int, int, str]]:
        return [(n.start_byte, n.end_byte, n.val) for n in self.all_variable_nodes]


class ProgramGraphExtractor:
    def __init__(self, language: str = "python") -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}")
        if not TS_AVAILABLE:
            raise ImportError("tree_sitter_python not installed")
        if language == "c" and not TS_C_AVAILABLE:
            raise ImportError("tree_sitter_c not installed (pip install codesign[c-lang])")

        self.language = language
        if language == "python":
            self.ts_lang = tree_sitter.Language(python_language())
        else:
            self.ts_lang = tree_sitter.Language(c_language())
        self.parser = tree_sitter.Parser(self.ts_lang)

    def parse(self, source_code: str):
        if not isinstance(source_code, str):
            raise TypeError(f"source_code must be str, got {type(source_code).__name__}")
        if not source_code.strip():
            raise ValueError("source_code is empty")
        return self.parser.parse(bytes(source_code, "utf8"))

    def build(self, source_code: str) -> ProgramGraph:
        tree = self.parse(source_code)
        source_bytes = source_code.encode("utf8")

        g = ProgramGraph(source=source_code, language=self.language)
        g.variables = self.get_variables(tree, source_bytes)
        g.dfg = self.extract_dfg(source_code)
        if self.language == "python":
            g.cfg_blocks = self._cfg_blocks_python(tree, source_bytes)
            g.sinks = self._sinks_python(tree, source_bytes)
        return g

    def get_variables(self, tree, source: bytes) -> dict[str, list[NodeWrapper]]:
        # Same query for both languages; filtering happens in _is_renameable.
        query = tree_sitter.Query(self.ts_lang, "(identifier) @var")
        cursor = tree_sitter.QueryCursor(query)
        out: dict[str, list[NodeWrapper]] = {}
        seen: set[tuple[int, int]] = set()
        for node in cursor.captures(tree.root_node).get("var", []):
            key = (node.start_byte, node.end_byte)
            if key in seen:
                continue
            seen.add(key)
            if not self._is_renameable(node):
                continue
            w = NodeWrapper.from_ts(node, source)
            out.setdefault(w.val, []).append(w)
        return out

    def _is_renameable(self, node) -> bool:
        p = node.parent
        if p is None:
            return True
        if self.language == "python":
            if p.type == "call" and p.child_by_field_name("function") == node:
                return False
            if p.type == "function_definition" and p.child_by_field_name("name") == node:
                return False
            # don't rename `.method` part of attribute access
            if p.type == "attribute" and p.child_by_field_name("attribute") == node:
                return False
        else:
            if p.type == "call_expression" and p.child_by_field_name("function") == node:
                return False
            if p.type == "function_declarator":
                return False
        return True

    def extract_dfg(self, source_code: str) -> dict[str, set[str]]:
        if not isinstance(source_code, str):
            raise TypeError("source_code must be str")
        source_bytes = source_code.encode("utf8")
        tree = self.parse(source_code)

        if self.language == "python":
            q_text = "(assignment left: (_) @target right: (_) @value)"
        else:
            q_text = "(assignment_expression left: (_) @target right: (_) @value)"
        query = tree_sitter.Query(self.ts_lang, q_text)
        cursor = tree_sitter.QueryCursor(query)

        # matches() pairs target+value within the same match, which is what
        # we actually want; captures() loses that grouping.
        sub_query = tree_sitter.Query(self.ts_lang, "(identifier) @id")

        dfg: dict[str, set[str]] = {}
        for _pattern_idx, caps in cursor.matches(tree.root_node):
            t_nodes = caps.get("target", [])
            v_nodes = caps.get("value", [])
            if not t_nodes or not v_nodes:
                continue
            t_node = t_nodes[0]
            v_node = v_nodes[0]
            target = source_bytes[t_node.start_byte:t_node.end_byte].decode("utf-8")
            sinks = dfg.setdefault(target, set())

            sub_cursor = tree_sitter.QueryCursor(sub_query)
            for sub_node in sub_cursor.captures(v_node).get("id", []):
                # skip callee names on the RHS
                if (sub_node.parent and sub_node.parent.type == "call"
                        and sub_node.parent.child_by_field_name("function") == sub_node):
                    continue
                sinks.add(source_bytes[sub_node.start_byte:sub_node.end_byte].decode("utf-8"))
        return dfg

    def _cfg_blocks_python(self, tree, source: bytes) -> list[NodeWrapper]:
        # Block-level approximation. Coarser than a real CFG but enough for
        # locating regions to feed the control-flow flattening mutator.
        targets = ("function_definition", "if_statement", "for_statement",
                   "while_statement", "try_statement")
        out: list[NodeWrapper] = []
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.type in targets:
                out.append(NodeWrapper.from_ts(n, source))
            stack.extend(n.children)
        return out

    def _sinks_python(self, tree, source: bytes) -> list[NodeWrapper]:
        out: list[NodeWrapper] = []
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.type == "call":
                fn = n.child_by_field_name("function")
                if fn is not None:
                    name = source[fn.start_byte:fn.end_byte].decode("utf-8")
                    leaf = name.split(".")[-1]
                    if leaf in PY_DANGEROUS_CALLS:
                        out.append(NodeWrapper.from_ts(fn, source))
            stack.extend(n.children)
        return out
