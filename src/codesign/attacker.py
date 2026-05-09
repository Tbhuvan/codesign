"""MAB-driven problem-space attacker."""

from __future__ import annotations

import ast
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from codesign.parser import NodeWrapper

log = logging.getLogger(__name__)


class Mutator(Protocol):
    name: str

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str: ...


@dataclass
class VariableRenaming:
    """VRTG via byte-precise tree-sitter ranges.

    Naive str.replace breaks when identifiers share prefixes (i vs index).
    We sort by start_byte descending and rewrite in-place so earlier offsets
    stay valid as later bytes grow.
    """
    name: str = "variable_renaming"
    deny: frozenset[str] = field(default_factory=lambda: frozenset({
        "True", "False", "None", "self", "cls", "print", "os", "sys"
    }))
    suffix_range: tuple[int, int] = (100, 999)

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        if not variables:
            return code

        rename: dict[str, str] = {}
        ordered = sorted(variables, key=lambda n: n.start_byte, reverse=True)
        out = bytearray(code.encode("utf8"))

        for var in ordered:
            if var.val in self.deny:
                continue
            new = rename.setdefault(var.val, f"{var.val}_adv{rng.randint(*self.suffix_range)}")
            out[var.start_byte:var.end_byte] = new.encode("utf8")
        return out.decode("utf8")


# Context-aware natural name pools, keyed by inferred role. Picked from
# common Python conventions a real developer would use; not exhaustive.
# The role is inferred from the original identifier (no model call needed).
_NATURAL_POOLS: dict[str, tuple[str, ...]] = {
    "loop_index":  ("idx", "i", "j", "k", "n", "pos", "ix"),
    "count":       ("count", "total", "num", "n_items", "size", "length"),
    "input":       ("user_input", "raw_input", "request", "payload", "data", "body"),
    "command":     ("cmd", "command", "instruction", "shell_cmd"),
    "query":       ("query", "stmt", "sql", "statement"),
    "filename":    ("path", "filename", "filepath", "fname", "target_path"),
    "user":        ("username", "user", "uid", "account_name", "principal"),
    "password":    ("password", "passwd", "secret", "creds", "key"),
    "result":      ("result", "out", "ret", "value", "output", "rv"),
    "buffer":      ("buf", "buffer", "data", "blob"),
    "url":         ("url", "endpoint", "uri", "target_url"),
    "generic":     ("x", "y", "z", "tmp", "val", "item", "entry"),
}

# For each role: (exact_or_token, suffix_substr).
# - exact_or_token: matched as a whole identifier or as a `_`-separated token
# - suffix_substr: matched as a substring at the end of the identifier,
#   only for hints >= 4 chars to avoid false positives like "frobnicate"
#   matching "i".
_ROLE_HINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "loop_index": (("i", "j", "k", "idx", "ix"), ()),
    "count":      (("n", "count", "total", "len", "size"), ("count", "total", "size")),
    "input":      (("input", "data", "raw", "request", "payload", "body"),
                   ("input", "data", "request", "payload")),
    "command":    (("cmd", "command", "shell"), ("cmd", "command")),
    "query":      (("query", "sql", "stmt"), ("query", "sql")),
    "user":       (("user", "username", "uid", "account"),
                   ("user", "username", "account")),
    "filename":   (("file", "path", "fname", "filename", "filepath"),
                   ("file", "path", "fname", "filename", "filepath")),
    "password":   (("password", "passwd", "pwd", "pass", "secret", "key", "creds"),
                   ("password", "passwd", "secret", "creds")),
    "result":     (("result", "out", "ret", "value", "rv"),
                   ("result", "value", "output")),
    "buffer":     (("buf", "blob", "bytes", "buffer"), ("buffer", "blob")),
    "url":        (("url", "uri", "endpoint", "target"),
                   ("url", "endpoint", "target")),
}


def _infer_role(name: str) -> str:
    """Infer the role of an identifier from its surface form.

    Match priority:
      1. exact match (`cmd` -> command)
      2. underscore-separated token component (`shell_cmd` -> command)
      3. suffix substring, only for hints >= 4 chars (`filename` -> filename)
    """
    n = name.lower()
    tokens = n.split("_")
    for role, (exact_hints, suffix_hints) in _ROLE_HINTS.items():
        for h in exact_hints:
            if h == n:
                return role
            if h in tokens:
                return role
        for h in suffix_hints:
            if len(h) >= 4 and n.endswith(h):
                return role
    return "generic"


@dataclass
class NaturalIdentifierRenaming:
    """VRTG with context-likely names instead of `name_advNNN`.

    Mimics what a real developer would do when refactoring: rename `i`
    to `idx`, rename `cmd` to `command`, rename `user_input` to
    `payload`. The pool is keyed by an inferred role from the original
    name. This is the WP3 "mimic real-world developer practices"
    framing applied directly.

    Substitution is byte-precise (same mechanism as VariableRenaming);
    only the new-name selection differs.
    """

    name: str = "natural_identifier_renaming"
    deny: frozenset[str] = field(default_factory=lambda: frozenset({
        "True", "False", "None", "self", "cls", "print", "os", "sys",
        "range", "len", "open", "type", "input", "Exception", "int",
        "str", "list", "dict", "set", "tuple", "bool", "float", "bytes",
    }))

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        if not variables:
            return code

        # Pick a fresh natural name per original identifier; reserve the
        # ones we've already chosen so two source names don't collide on
        # the same target.
        rename: dict[str, str] = {}
        used: set[str] = set()
        # Preserve everything currently in the source (so we don't rename
        # `i` to `count` if `count` already exists and means something else).
        existing = {v.val for v in variables}
        used.update(existing)
        used.update(self.deny)

        for var in variables:
            if var.val in rename or var.val in self.deny:
                continue
            role = _infer_role(var.val)
            pool = _NATURAL_POOLS.get(role, _NATURAL_POOLS["generic"])
            shuffled = list(pool)
            rng.shuffle(shuffled)
            chosen = next((c for c in shuffled if c not in used), None)
            if chosen is None:
                # All natural names are taken; fall back to a numbered
                # variant of the role's first option (still natural-ish).
                chosen = f"{shuffled[0]}_{rng.randint(2, 9)}"
                while chosen in used:
                    chosen = f"{shuffled[0]}_{rng.randint(2, 9)}"
            rename[var.val] = chosen
            used.add(chosen)

        # Rewrite in descending byte order so later changes don't shift
        # earlier offsets.
        ordered = sorted(variables, key=lambda n: n.start_byte, reverse=True)
        out = bytearray(code.encode("utf8"))
        for var in ordered:
            if var.val in self.deny or var.val not in rename:
                continue
            out[var.start_byte:var.end_byte] = rename[var.val].encode("utf8")
        return out.decode("utf8")


@dataclass
class EquivalentExpressionSubstitution:
    """Substitute equivalent expressions via AST rewrites.

    Each rule preserves semantics under standard Python evaluation:
      not (a == b)  ->  a != b
      not (a != b)  ->  a == b
      not (a < b)   ->  a >= b   (and the three cousins)
      a + 0         ->  a        (and 0 + a, a - 0)
      a * 1         ->  a        (and 1 * a)
      not not a     ->  bool(a)

    The mutator picks a random eligible site per call and rewrites it.
    Idempotent in expectation: re-running can produce a different
    site, but the source converges as eligible sites are exhausted.
    """

    name: str = "equivalent_expression_substitution"

    _CMP_FLIP: dict[type, type] = field(default_factory=lambda: {
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE, ast.GtE: ast.Lt,
        ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    })

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        candidates: list[tuple[ast.AST, ast.AST]] = []
        for parent in ast.walk(tree):
            for _field_name, field_val in ast.iter_fields(parent):
                if isinstance(field_val, list):
                    for child in field_val:
                        if self._rewrite(child) is not None:
                            candidates.append((parent, child))
                elif self._rewrite(field_val) is not None:
                    candidates.append((parent, field_val))

        if not candidates:
            return code

        parent, target = rng.choice(candidates)
        new_node = self._rewrite(target)
        if new_node is None:
            return code

        # Replace target with new_node inside parent. ast.NodeTransformer
        # is overkill for a single-site swap; we walk parent's fields and
        # substitute.
        for field_name, field_val in ast.iter_fields(parent):
            if isinstance(field_val, list):
                for i, child in enumerate(field_val):
                    if child is target:
                        field_val[i] = new_node
                        break
            elif field_val is target:
                setattr(parent, field_name, new_node)

        ast.fix_missing_locations(tree)
        try:
            return ast.unparse(tree)
        except (AttributeError, ValueError):
            return code

    def _rewrite(self, node: ast.AST | None) -> ast.AST | None:
        if not isinstance(node, ast.AST):
            return None

        # not (a OP b)  ->  a (flip OP) b
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1):
            cmp = node.operand
            op_type = type(cmp.ops[0])
            flip = self._CMP_FLIP.get(op_type)
            if flip is not None:
                return ast.Compare(left=cmp.left, ops=[flip()], comparators=cmp.comparators)

        # not not a  ->  bool(a)
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.UnaryOp)
                and isinstance(node.operand.op, ast.Not)):
            inner = node.operand.operand
            return ast.Call(
                func=ast.Name(id="bool", ctx=ast.Load()),
                args=[inner],
                keywords=[],
            )

        # a + 0  ->  a
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                and isinstance(node.right, ast.Constant) and node.right.value == 0):
            return node.left
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                and isinstance(node.left, ast.Constant) and node.left.value == 0):
            return node.right
        # a - 0  ->  a
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
                and isinstance(node.right, ast.Constant) and node.right.value == 0):
            return node.left
        # a * 1  ->  a
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                and isinstance(node.right, ast.Constant) and node.right.value == 1):
            return node.left
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                and isinstance(node.left, ast.Constant) and node.left.value == 1):
            return node.right

        return None


@dataclass
class DeadCodeInsertion:
    name: str = "dead_code_insertion"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables  # unused
        decoy = f"_vrtg_decoy_{rng.randint(1000, 9999)}"
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") and stripped.endswith(":"):
                indent = " " * (len(line) - len(line.lstrip()) + 4)
                lines.insert(i + 1, f"{indent}{decoy} = {{'safe': True}}; {decoy}['_'] = len({decoy})")
                break
        return "\n".join(lines)


# Neutral, generic developer-style docstrings. We do NOT include any
# claim about safety or correctness; that would push toward malware-
# style commenting. These are the kind of one-liners a developer
# writes when annotating an existing function.
_DOCSTRING_POOL: tuple[str, ...] = (
    "Process the input and return the result.",
    "Run the operation.",
    "Build and return the value.",
    "Validate the input and proceed.",
    "Handle the request.",
    "Execute the workflow.",
    "Return the computed result.",
    "Carry out the task on the given input.",
)

_INLINE_COMMENT_POOL: tuple[str, ...] = (
    "build the value",
    "run it",
    "process input",
    "compute result",
    "format output",
    "main step",
    "extract values",
    "wrap up",
)


@dataclass
class DocstringInsertion:
    """Insert a neutral docstring as the first statement of every function.

    Real developers add docstrings during refactoring. The pool
    contains generic one-liners drawn from observed conventions; we
    avoid any claim about safety or correctness so this stays
    naturalistic, not adversarially-suggestive.
    """

    name: str = "docstring_insertion"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.body:
                continue
            # Already has a docstring? Skip.
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                continue
            text = rng.choice(_DOCSTRING_POOL)
            doc = ast.Expr(value=ast.Constant(value=text))
            node.body.insert(0, doc)
            changed = True

        if not changed:
            return code
        try:
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except (AttributeError, ValueError):
            return code


@dataclass
class InlineCommentInsertion:
    """Append a neutral inline comment to a randomly chosen function-body line.

    Modifies the source text directly (rather than the AST) because
    Python's ``ast`` discards comments. The comment is appended to
    a non-blank, non-comment line inside a function body.
    """

    name: str = "inline_comment_insertion"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables

        lines = code.split("\n")
        # Find candidate lines: indented, non-blank, no existing comment.
        candidates: list[int] = []
        in_function = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") and stripped.endswith(":"):
                in_function = True
                continue
            if not in_function:
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if "#" in line:  # already has an inline comment
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                in_function = False
                continue
            candidates.append(i)

        if not candidates:
            return code

        idx = rng.choice(candidates)
        comment = rng.choice(_INLINE_COMMENT_POOL)
        # Strip any trailing whitespace before appending.
        lines[idx] = lines[idx].rstrip() + f"  # {comment}"
        return "\n".join(lines)


@dataclass
class TypeAnnotationsAdded:
    """Annotate untyped function parameters with a conservative ``str`` hint.

    Real Python developers add type hints during code review or
    refactoring. We only annotate parameters that have no existing
    annotation, and we use ``str`` as the conservative default. We
    do not add return-type annotations because inferring them
    requires data-flow analysis we don't ship.
    """

    name: str = "type_annotations_added"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables, rng

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in node.args.args:
                if arg.annotation is not None:
                    continue
                if arg.arg in ("self", "cls"):
                    continue
                arg.annotation = ast.Name(id="str", ctx=ast.Load())
                changed = True

        if not changed:
            return code
        try:
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except (AttributeError, ValueError):
            return code


@dataclass
class ControlFlowFlattening:
    """Wrap each function body in `if True: ...`. CFG-level no-op."""
    name: str = "control_flow_flattening"

    def __call__(self, code: str, *, variables: list[NodeWrapper], rng: random.Random) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be str")
        del variables, rng

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.body:
                    continue
                first = node.body[0]
                # already wrapped, skip so we stay idempotent
                if (len(node.body) == 1 and isinstance(first, ast.If)
                        and isinstance(first.test, ast.Constant) and first.test.value is True):
                    continue
                node.body = [ast.If(test=ast.Constant(value=True), body=node.body, orelse=[])]
                changed = True

        if not changed:
            return code
        try:
            return ast.unparse(tree)
        except (AttributeError, ValueError):
            return code


@dataclass
class MutationStep:
    step: int
    strategy: str
    score_before: float
    score_after: float
    reward: float
    parse_valid: bool


@dataclass
class AttackTrace:
    steps: list[MutationStep] = field(default_factory=list)
    q_table: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    best_score: float = 1.0
    final_code: str = ""


ScoreFn = Callable[[str], float]


class RLAdversary:
    """Epsilon-greedy MAB over a set of mutators.

    Reward = score_before - score_after. A bad mutation (parse fail or
    score regression) gets -1.0 so the bandit learns to avoid it.
    """

    def __init__(
        self,
        target_model_score_fn: ScoreFn,
        mutators: list[Mutator] | None = None,
        epsilon: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if not callable(target_model_score_fn):
            raise TypeError("target_model_score_fn must be callable")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.score_fn = target_model_score_fn
        self.mutators = mutators or [
            VariableRenaming(),
            NaturalIdentifierRenaming(),
            DeadCodeInsertion(),
            ControlFlowFlattening(),
            EquivalentExpressionSubstitution(),
            DocstringInsertion(),
            InlineCommentInsertion(),
            TypeAnnotationsAdded(),
        ]
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table = {m.name: 0.0 for m in self.mutators}
        self.counts = {m.name: 0 for m in self.mutators}

    @staticmethod
    def _ast_ok(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _select(self) -> Mutator:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.mutators)
        best = max(self.q_table, key=lambda n: self.q_table[n])
        for m in self.mutators:
            if m.name == best:
                return m
        return self.mutators[0]

    def attack(
        self,
        clean_code: str,
        variables: list[NodeWrapper],
        max_steps: int = 15,
    ) -> tuple[str, AttackTrace]:
        if not isinstance(clean_code, str):
            raise TypeError("clean_code must be str")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        try:
            base = float(self.score_fn(clean_code))
        except Exception as e:
            log.warning("score_fn failed on clean code: %s", e)
            return clean_code, AttackTrace(final_code=clean_code, best_score=1.0)

        best_code = code = clean_code
        best = base
        trace = AttackTrace(final_code=clean_code, best_score=base)

        for step in range(max_steps):
            m = self._select()
            try:
                cand = m(code, variables=variables, rng=self.rng)
            except Exception as e:
                log.warning("mutator %s raised: %s", m.name, e)
                trace.steps.append(MutationStep(step, m.name, base, base, -1.0, False))
                self._update_q(m.name, -1.0)
                continue

            ok = self._ast_ok(cand)
            if not ok:
                reward = -1.0
                new = base
            else:
                try:
                    new = float(self.score_fn(cand))
                except Exception as e:
                    log.warning("score_fn failed on mutation: %s", e)
                    new = base
                    reward = -1.0
                else:
                    reward = base - new
                    if new < best:
                        best = new
                        best_code = cand
                    if new < base:
                        # commit the improvement and keep mutating from here
                        code = cand
                        base = new

            trace.steps.append(MutationStep(step, m.name, base, new, reward, ok))
            self._update_q(m.name, reward)

        trace.q_table = dict(self.q_table)
        trace.counts = dict(self.counts)
        trace.best_score = best
        trace.final_code = best_code
        return best_code, trace

    def _update_q(self, name: str, reward: float) -> None:
        self.counts[name] += 1
        a = 1.0 / self.counts[name]
        self.q_table[name] += a * (reward - self.q_table[name])
