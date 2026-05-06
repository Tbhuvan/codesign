import ast
import random

import pytest

from codesign.attacker import (
    AttackTrace,
    ControlFlowFlattening,
    DeadCodeInsertion,
    RLAdversary,
    VariableRenaming,
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
        out, trace = adv.attack(vuln_os_system, g.all_variable_nodes, max_steps=5)
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
