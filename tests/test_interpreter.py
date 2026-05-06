import pytest

from codesign.interpreter import (
    CircuitReport,
    HeadImportance,
    VulnerabilityCircuitProbe,
    expected_random_logit_diff,
)


class TestProbeValidation:
    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError):
            VulnerabilityCircuitProbe(model_name="")

    def test_rejects_unknown_device(self):
        with pytest.raises(ValueError):
            VulnerabilityCircuitProbe(device="tpu")

    def test_align_sinks_requires_loaded_model(self):
        probe = VulnerabilityCircuitProbe()
        with pytest.raises(RuntimeError):
            probe.align_sinks_to_tokens("x = 1", ["x"])


class TestExpectedRandomLogitDiff:
    def test_positive(self):
        assert expected_random_logit_diff(12, 12) > 0

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            expected_random_logit_diff(0, 12)


class TestCircuitReport:
    def test_to_dict(self):
        report = CircuitReport(
            model_name="test",
            n_layers=2,
            n_heads=4,
            sink_token_positions=[3],
            dfg_node_count=2,
            layer_logit_diffs={0: 0.5, 1: -0.2},
            head_logit_diffs=[],
            top_k_heads=[
                HeadImportance(layer=0, head=2, clean_logit=10.0, ablated_logit=8.0, logit_diff=2.0)
            ],
            notes="ok",
        )
        d = report.to_dict()
        assert d["model_name"] == "test"
        assert d["top_k_heads"][0]["logit_diff"] == 2.0


@pytest.mark.slow
def test_full_sweep_pythia_160m(vuln_os_system: str):
    pytest.importorskip("transformer_lens")
    from codesign.parser import ProgramGraphExtractor

    g = ProgramGraphExtractor().build(vuln_os_system)
    probe = VulnerabilityCircuitProbe(model_name="EleutherAI/pythia-160m")
    report = probe.run_dfg_guided_patching(vuln_os_system, g.dfg, top_k=5, max_layers=2)
    assert report.n_layers >= 2
    assert len(report.top_k_heads) <= 5
    assert all(isinstance(h.logit_diff, float) for h in report.top_k_heads)
