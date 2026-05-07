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

    def test_ablation_mode_round_trips(self):
        r = CircuitReport(
            model_name="x", n_layers=1, n_heads=1,
            sink_token_positions=[2, 5], dfg_node_count=1,
            ablation_mode="sink_positions",
        )
        assert r.to_dict()["ablation_mode"] == "sink_positions"


class TestPositionSpecificAblation:
    """The hook should zero only sink positions when supplied, vs all positions otherwise."""

    def _run_hook(self, sink_positions, head=2, n_pos=8):
        # We replicate the hook logic out-of-band so we can test the masking
        # without a real model. value shape: [batch, pos, head, d_head]
        import torch
        v = torch.ones(1, n_pos, 4, 16)

        def hook(value, hook):
            if head is None:
                return torch.zeros_like(value)
            if sink_positions:
                value[:, sink_positions, head, :] = 0.0
            else:
                value[:, :, head, :] = 0.0
            return value

        return hook(v, None)

    def test_whole_head_zeroes_all_positions(self):
        out = self._run_hook(sink_positions=None, head=2)
        # head 2 should be 0 everywhere
        assert (out[0, :, 2, :] == 0).all()
        # other heads untouched
        assert (out[0, :, 0, :] == 1).all()

    def test_sink_positions_only_zeroes_those_positions(self):
        sinks = [1, 3, 5]
        out = self._run_hook(sink_positions=sinks, head=2)
        # head 2 at sink positions: zero
        assert (out[0, sinks, 2, :] == 0).all()
        # head 2 at non-sink positions: untouched
        non_sinks = [i for i in range(out.shape[1]) if i not in sinks]
        assert (out[0, non_sinks, 2, :] == 1).all()

    def test_empty_sink_positions_falls_back_to_whole_head(self):
        out = self._run_hook(sink_positions=[], head=2)
        assert (out[0, :, 2, :] == 0).all()


@pytest.mark.slow
def test_full_sweep_qwen_coder(vuln_os_system: str):
    """Smoke test the full pipeline against the real default model."""
    pytest.importorskip("transformer_lens")
    from codesign.parser import ProgramGraphExtractor

    g = ProgramGraphExtractor().build(vuln_os_system)
    probe = VulnerabilityCircuitProbe(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    report = probe.run_dfg_guided_patching(vuln_os_system, g.dfg, top_k=5, max_layers=2)
    assert report.n_layers >= 2
    assert len(report.top_k_heads) <= 5
    assert all(isinstance(h.logit_diff, float) for h in report.top_k_heads)
    assert report.ablation_mode in ("sink_positions", "whole_head")
