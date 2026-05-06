import pytest

from codesign.metrics import (
    AttackResult,
    confidence_drop,
    dfg_preserved,
    evaded,
    head_importance_logit_diff,
    parse_valid,
    summarise,
)


class TestParseValid:
    def test_valid(self):
        assert parse_valid("x = 1\n")

    def test_invalid(self):
        assert not parse_valid("def (\n")

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            parse_valid(42)


class TestDFGPreserved:
    def test_identical(self):
        d = {"a": {"b", "c"}}
        assert dfg_preserved(d, d)

    def test_renamed(self):
        original = {"x": {"y", "z"}}
        renamed = {"x_adv1": {"y_adv2", "z_adv3"}}
        rename = {"x": "x_adv1", "y": "y_adv2", "z": "z_adv3"}
        assert dfg_preserved(original, renamed, rename_map=rename)

    def test_different_topology(self):
        a = {"x": {"y"}}
        b = {"x": {"y", "z"}}
        assert not dfg_preserved(a, b)


class TestConfidenceDrop:
    def test_basic(self):
        assert confidence_drop(0.9, 0.3) == pytest.approx(0.6)

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            confidence_drop(1.5, 0.0)


class TestEvaded:
    def test_below_threshold(self):
        assert evaded(0.3)

    def test_above_threshold(self):
        assert not evaded(0.7)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            evaded(0.5, threshold=2.0)


def _result(sample_id: str, drop: float) -> AttackResult:
    return AttackResult(
        sample_id=sample_id,
        cwe="CWE-78",
        original_score=0.95,
        final_score=0.95 - drop,
        steps_taken=5,
        evaded=(0.95 - drop) < 0.5,
        parse_valid=True,
        dfg_preserved=True,
        mutations_applied=["variable_renaming", "dead_code_insertion"],
        final_code="...",
    )


class TestSummarise:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summarise([])

    def test_aggregates(self):
        s = summarise([_result("a", 0.6), _result("b", 0.1)])
        assert s.n_samples == 2
        assert s.evasion_rate == 0.5
        assert s.parse_validity_rate == 1.0
        assert "variable_renaming" in s.mutation_diversity


class TestHeadImportance:
    def test_basic(self):
        out = head_importance_logit_diff(
            clean_logit=10.0,
            ablated_logits={(0, 0): 8.0, (1, 2): 9.5},
        )
        assert out[(0, 0)] == pytest.approx(2.0)
        assert out[(1, 2)] == pytest.approx(0.5)

    def test_invalid_keys(self):
        with pytest.raises(TypeError):
            head_importance_logit_diff(10.0, {("a", "b"): 5.0})
