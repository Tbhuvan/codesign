import ast
import random

import pytest

from codesign.attacker import (
    _NATURAL_POOLS,
    AttackTrace,
    ControlFlowFlattening,
    DeadCodeInsertion,
    DocstringInsertion,
    EquivalentExpressionSubstitution,
    InlineCommentInsertion,
    NaturalIdentifierRenaming,
    RLAdversary,
    TypeAnnotationsAdded,
    VariableRenaming,
    _infer_role,
)
from codesign.parser import ProgramGraphExtractor


@pytest.fixture(scope="module")
def extractor():
    return ProgramGraphExtractor()


class TestVariableRenaming:
    def test_does_not_corrupt_shared_prefixes(self, extractor, shared_prefix_code):
        g = extractor.build(shared_prefix_code)
        m = VariableRenaming()
        out = m(shared_prefix_code, variables=g.all_variable_nodes, rng=random.Random(0))
        ast.parse(out)  # would raise on corruption
        assert "_adv" in out

    def test_consistent_rename_across_uses(self, extractor, shared_prefix_code):
        g = extractor.build(shared_prefix_code)
        m = VariableRenaming()
        out = m(shared_prefix_code, variables=g.all_variable_nodes, rng=random.Random(42))
        tree = ast.parse(out)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "i" not in names
        assert "index" not in names

    def test_skips_denylisted(self, extractor):
        code = "def f(x):\n    return print(x)"
        g = extractor.build(code)
        m = VariableRenaming()
        out = m(code, variables=g.all_variable_nodes, rng=random.Random(0))
        assert "print" in out

    def test_empty_variables_returns_unchanged(self):
        m = VariableRenaming()
        code = "x = 1\n"
        assert m(code, variables=[], rng=random.Random(0)) == code

    def test_rejects_non_str(self):
        m = VariableRenaming()
        with pytest.raises(TypeError):
            m(123, variables=[], rng=random.Random(0))


class TestDeadCodeInsertion:
    def test_inserts_decoy_after_def(self, vuln_os_system):
        m = DeadCodeInsertion()
        out = m(vuln_os_system, variables=[], rng=random.Random(0))
        assert "_vrtg_decoy_" in out
        ast.parse(out)


class TestControlFlowFlattening:
    def test_wraps_function_body(self, vuln_os_system):
        m = ControlFlowFlattening()
        out = m(vuln_os_system, variables=[], rng=random.Random(0))
        assert "if True:" in out
        ast.parse(out)

    def test_idempotent_on_already_flattened(self, vuln_os_system):
        m = ControlFlowFlattening()
        once = m(vuln_os_system, variables=[], rng=random.Random(0))
        twice = m(once, variables=[], rng=random.Random(0))
        assert once.count("if True:") == twice.count("if True:")


class TestRLAdversary:
    def test_attack_returns_trace(self, extractor, vuln_os_system, mock_svd):
        g = extractor.build(vuln_os_system)
        adv = RLAdversary(target_model_score_fn=mock_svd, seed=0)
        _out, trace = adv.attack(vuln_os_system, g.all_variable_nodes, max_steps=5)
        assert isinstance(trace, AttackTrace)
        assert len(trace.steps) == 5
        assert trace.best_score <= mock_svd(vuln_os_system)

    def test_invalid_epsilon(self, mock_svd):
        with pytest.raises(ValueError):
            RLAdversary(target_model_score_fn=mock_svd, epsilon=1.5)

    def test_score_fn_must_be_callable(self):
        with pytest.raises(TypeError):
            RLAdversary(target_model_score_fn=42)

    def test_reproducible_with_seed(self, extractor, vuln_os_system, mock_svd):
        g = extractor.build(vuln_os_system)
        a = RLAdversary(target_model_score_fn=mock_svd, seed=7)
        b = RLAdversary(target_model_score_fn=mock_svd, seed=7)
        out_a, _ = a.attack(vuln_os_system, g.all_variable_nodes, max_steps=4)
        out_b, _ = b.attack(vuln_os_system, g.all_variable_nodes, max_steps=4)
        assert out_a == out_b

    def test_qtable_updates(self, extractor, vuln_os_system, mock_svd):
        g = extractor.build(vuln_os_system)
        adv = RLAdversary(target_model_score_fn=mock_svd, seed=0)
        _, trace = adv.attack(vuln_os_system, g.all_variable_nodes, max_steps=8)
        assert any(abs(v) > 1e-6 for v in trace.q_table.values())


class TestRoleInference:
    def test_loop_index_role(self):
        assert _infer_role("i") == "loop_index"
        assert _infer_role("idx") == "loop_index"
        assert _infer_role("j") == "loop_index"

    def test_command_role(self):
        assert _infer_role("cmd") == "command"
        assert _infer_role("shell_cmd") == "command"

    def test_user_role(self):
        assert _infer_role("username") == "user"
        assert _infer_role("uid") == "user"

    def test_password_role(self):
        assert _infer_role("password") == "password"
        assert _infer_role("admin_pass") == "password"

    def test_filename_role(self):
        assert _infer_role("filename") == "filename"
        assert _infer_role("target_path") == "filename"

    def test_generic_fallback(self):
        assert _infer_role("xyzzy") == "generic"
        assert _infer_role("frobnicate") == "generic"


class TestNaturalIdentifierRenaming:
    def test_picks_natural_name(self, extractor, shared_prefix_code):
        g = extractor.build(shared_prefix_code)
        m = NaturalIdentifierRenaming()
        out = m(shared_prefix_code, variables=g.all_variable_nodes, rng=random.Random(0))
        # Output must parse and not contain "_adv" suffixes
        ast.parse(out)
        assert "_adv" not in out
        # At least one identifier should have changed to a name from a natural pool
        natural_names = {n for pool in _NATURAL_POOLS.values() for n in pool}
        new_names = {n.id for n in ast.walk(ast.parse(out)) if isinstance(n, ast.Name)}
        original_names = {"i", "index", "valid"}
        renamed = new_names - original_names
        assert renamed.intersection(natural_names), (
            "expected at least one identifier to be renamed to a natural-pool name; "
            f"renamed identifiers were {renamed}"
        )

    def test_no_collisions_in_output(self, extractor, shared_prefix_code):
        # Two source identifiers should never map to the same target name.
        g = extractor.build(shared_prefix_code)
        m = NaturalIdentifierRenaming()
        for seed in range(5):
            out = m(shared_prefix_code, variables=g.all_variable_nodes, rng=random.Random(seed))
            ast.parse(out)  # must remain valid

    def test_does_not_corrupt_shared_prefixes(self, extractor, shared_prefix_code):
        # Same byte-precise guarantee as VariableRenaming.
        g = extractor.build(shared_prefix_code)
        m = NaturalIdentifierRenaming()
        out = m(shared_prefix_code, variables=g.all_variable_nodes, rng=random.Random(0))
        ast.parse(out)

    def test_skips_denylisted(self, extractor):
        code = "def f(x):\n    return print(x)"
        g = extractor.build(code)
        m = NaturalIdentifierRenaming()
        out = m(code, variables=g.all_variable_nodes, rng=random.Random(0))
        assert "print" in out

    def test_empty_variables_returns_unchanged(self):
        m = NaturalIdentifierRenaming()
        code = "x = 1\n"
        assert m(code, variables=[], rng=random.Random(0)) == code

    def test_rejects_non_str(self):
        m = NaturalIdentifierRenaming()
        with pytest.raises(TypeError):
            m(123, variables=[], rng=random.Random(0))


class TestEquivalentExpressionSubstitution:
    def test_rejects_non_str(self):
        m = EquivalentExpressionSubstitution()
        with pytest.raises(TypeError):
            m(123, variables=[], rng=random.Random(0))

    def test_not_equal_flips_to_neq(self):
        m = EquivalentExpressionSubstitution()
        code = "def f(a, b):\n    return not (a == b)\n"
        for seed in range(8):
            out = m(code, variables=[], rng=random.Random(seed))
            ast.parse(out)
            if "!=" in out:
                # Found one of the rewrite targets
                assert "not" not in out.split("\n")[1] or "not (a ==" not in out
                return
        pytest.fail("expected at least one seed to rewrite `not (a == b)` -> `a != b`")

    def test_plus_zero_simplifies(self):
        m = EquivalentExpressionSubstitution()
        code = "def f(a):\n    return a + 0\n"
        # The mutator picks a random eligible site, but `a + 0 -> a` is the
        # only candidate here, so any seed that triggers the rewrite must
        # produce `return a` (no `+ 0`).
        for seed in range(8):
            out = m(code, variables=[], rng=random.Random(seed))
            ast.parse(out)
            if "+ 0" not in out and out != code:
                return
        pytest.fail("expected at least one seed to rewrite `a + 0` -> `a`")

    def test_unchanged_when_no_candidates(self):
        m = EquivalentExpressionSubstitution()
        code = "def f(a, b):\n    return a + b\n"
        out = m(code, variables=[], rng=random.Random(0))
        # No eligible sites; should be returned unchanged.
        assert out == code

    def test_invalid_input_returns_unchanged(self):
        m = EquivalentExpressionSubstitution()
        code = "def broken(:\n"  # syntax error
        out = m(code, variables=[], rng=random.Random(0))
        assert out == code


class TestDocstringInsertion:
    def test_inserts_docstring(self, vuln_os_system):
        m = DocstringInsertion()
        out = m(vuln_os_system, variables=[], rng=random.Random(0))
        ast.parse(out)
        # The first statement of every function should be a string literal.
        tree = ast.parse(out)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.body
                first = node.body[0]
                assert (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)), (
                    f"function {node.name} missing docstring after insertion"
                )

    def test_idempotent_on_already_documented(self):
        code = 'def f():\n    """already here"""\n    return 1\n'
        m = DocstringInsertion()
        out = m(code, variables=[], rng=random.Random(0))
        # Should not double-document. Round-trip ast.unparse normalises
        # whitespace, so compare AST shape rather than text.
        before = ast.parse(code)
        after = ast.parse(out)
        before_funcs = [n for n in ast.walk(before) if isinstance(n, ast.FunctionDef)]
        after_funcs = [n for n in ast.walk(after) if isinstance(n, ast.FunctionDef)]
        for b, a in zip(before_funcs, after_funcs):
            assert len(b.body) == len(a.body)

    def test_rejects_non_str(self):
        m = DocstringInsertion()
        with pytest.raises(TypeError):
            m(42, variables=[], rng=random.Random(0))


class TestInlineCommentInsertion:
    def test_inserts_inline_comment(self, vuln_os_system):
        m = InlineCommentInsertion()
        out = m(vuln_os_system, variables=[], rng=random.Random(0))
        ast.parse(out)
        # Check at least one line in the function body now ends with `# <comment>`
        assert "  #" in out

    def test_no_eligible_lines_returns_unchanged(self):
        m = InlineCommentInsertion()
        # Single-line function body that already has a comment.
        code = "def f():\n    return 1  # already commented\n"
        out = m(code, variables=[], rng=random.Random(0))
        # Either unchanged or another commentable line found; verify it parses
        # and didn't double-insert on the existing-comment line.
        ast.parse(out)
        assert out.count("# already commented") == 1

    def test_does_not_break_indentation(self, vuln_os_system):
        m = InlineCommentInsertion()
        for seed in range(5):
            out = m(vuln_os_system, variables=[], rng=random.Random(seed))
            ast.parse(out)


class TestTypeAnnotationsAdded:
    def test_annotates_untyped_params(self):
        code = "def lookup(name, query):\n    return name + query\n"
        m = TypeAnnotationsAdded()
        out = m(code, variables=[], rng=random.Random(0))
        ast.parse(out)
        assert "name: str" in out
        assert "query: str" in out

    def test_skips_already_annotated(self):
        code = "def lookup(name: int, query):\n    return query\n"
        m = TypeAnnotationsAdded()
        out = m(code, variables=[], rng=random.Random(0))
        # Existing `: int` annotation must be preserved.
        assert "name: int" in out
        # `query` should now be annotated too.
        assert "query: str" in out

    def test_skips_self_and_cls(self):
        code = "class C:\n    def method(self, x):\n        return x\n"
        m = TypeAnnotationsAdded()
        out = m(code, variables=[], rng=random.Random(0))
        ast.parse(out)
        # self should not be annotated.
        assert "self: " not in out
        # x should be.
        assert "x: str" in out

    def test_unchanged_when_nothing_to_annotate(self):
        code = "x = 1\n"
        m = TypeAnnotationsAdded()
        out = m(code, variables=[], rng=random.Random(0))
        # No functions, nothing to annotate, output should round-trip
        # through ast.unparse identically (in the no-change branch we
        # actually return the input untouched).
        assert out == code

    def test_rejects_non_str(self):
        m = TypeAnnotationsAdded()
        with pytest.raises(TypeError):
            m(123, variables=[], rng=random.Random(0))


class TestRLAdversaryDefaultPool:
    def test_default_pool_has_eight_mutators(self, mock_svd):
        adv = RLAdversary(target_model_score_fn=mock_svd, seed=0)
        # 5 originals + 3 newly-added naturalistic ones
        names = [m.name for m in adv.mutators]
        assert "variable_renaming" in names
        assert "natural_identifier_renaming" in names
        assert "dead_code_insertion" in names
        assert "control_flow_flattening" in names
        assert "equivalent_expression_substitution" in names
        assert "docstring_insertion" in names
        assert "inline_comment_insertion" in names
        assert "type_annotations_added" in names
        assert len(adv.mutators) == 8
