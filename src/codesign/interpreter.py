"""DFG-guided activation patching for code LLMs."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    import torch
    from transformer_lens import HookedTransformer
    from transformer_lens import utils as tl_utils
    TL_AVAILABLE = True
except ImportError:
    TL_AVAILABLE = False


@dataclass
class HeadImportance:
    layer: int
    head: int
    clean_logit: float
    ablated_logit: float
    logit_diff: float


@dataclass
class CircuitReport:
    model_name: str
    n_layers: int
    n_heads: int
    sink_token_positions: list[int]
    dfg_node_count: int
    layer_logit_diffs: dict[int, float] = field(default_factory=dict)
    head_logit_diffs: list[HeadImportance] = field(default_factory=list)
    top_k_heads: list[HeadImportance] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "sink_token_positions": self.sink_token_positions,
            "dfg_node_count": self.dfg_node_count,
            "layer_logit_diffs": {str(k): v for k, v in self.layer_logit_diffs.items()},
            "top_k_heads": [
                {"layer": h.layer, "head": h.head, "logit_diff": h.logit_diff}
                for h in self.top_k_heads
            ],
            "notes": self.notes,
        }


class VulnerabilityCircuitProbe:
    """Zero-ablate every (layer, head) and rank by logit_diff."""

    def __init__(self, model_name: str = "EleutherAI/pythia-160m", device: str = "cpu") -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"unsupported device {device!r}")
        self.model_name = model_name
        self.device = device
        self.model: Any | None = None

    def load_model(self) -> None:
        if not TL_AVAILABLE:
            raise ImportError("transformer_lens not installed")
        log.info("loading %s on %s", self.model_name, self.device)
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
        )

    def align_sinks_to_tokens(self, code: str, sink_names: list[str]) -> list[int]:
        if self.model is None:
            raise RuntimeError("call load_model() first")
        toks: list[str] = self.model.to_str_tokens(code)
        targets = set(sink_names)
        return [i for i, t in enumerate(toks) if t.strip() in targets]

    def run_dfg_guided_patching(
        self,
        code: str,
        dfg: dict[str, set[str]],
        sink_token_names: list[str] | None = None,
        top_k: int = 10,
        max_layers: int | None = None,
    ) -> CircuitReport:
        if self.model is None:
            self.load_model()
        assert self.model is not None

        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")

        cfg = self.model.cfg
        n_layers = cfg.n_layers if max_layers is None else min(cfg.n_layers, max_layers)
        n_heads = cfg.n_heads

        sink_names = sink_token_names if sink_token_names is not None else list(dfg.keys())
        sink_positions = self.align_sinks_to_tokens(code, sink_names) if sink_names else []

        tokens = self.model.to_tokens(code)
        if tokens.shape[1] > 256:
            tokens = tokens[:, :256]  # keep the sweep tractable

        with torch.no_grad():
            clean_logits, _ = self.model.run_with_cache(tokens)
        clean_metric = float(clean_logits[0, -1, :].max().item())

        layer_diffs: dict[int, float] = {}
        for layer in range(n_layers):
            ablated = self._ablate(tokens, layer=layer, head=None)
            layer_diffs[layer] = clean_metric - float(ablated[0, -1, :].max().item())

        head_results: list[HeadImportance] = []
        for layer in range(n_layers):
            for head in range(n_heads):
                ablated = self._ablate(tokens, layer=layer, head=head)
                a = float(ablated[0, -1, :].max().item())
                head_results.append(HeadImportance(
                    layer=layer, head=head,
                    clean_logit=clean_metric, ablated_logit=a,
                    logit_diff=clean_metric - a,
                ))

        head_results.sort(key=lambda h: abs(h.logit_diff), reverse=True)
        top = head_results[:top_k]

        return CircuitReport(
            model_name=self.model_name,
            n_layers=n_layers,
            n_heads=n_heads,
            sink_token_positions=sink_positions,
            dfg_node_count=len(dfg),
            layer_logit_diffs=layer_diffs,
            head_logit_diffs=head_results,
            top_k_heads=top,
            notes=self._notes(layer_diffs, top, sink_positions, dfg),
        )

    # Thin wrapper for the CLI's scan command.
    def activation_patch_dfg_nodes(
        self, clean_code: str, dfg_dict: dict[str, set[str]]
    ) -> dict[str, Any]:
        r = self.run_dfg_guided_patching(clean_code, dfg_dict, top_k=5)
        critical = (
            f"L{max(r.layer_logit_diffs, key=lambda k: abs(r.layer_logit_diffs[k]))}"
            if r.layer_logit_diffs else "n/a"
        )
        return {
            "model": r.model_name,
            "n_layers": r.n_layers,
            "n_heads": r.n_heads,
            "dfg_nodes_analysed": r.dfg_node_count,
            "sink_token_positions": r.sink_token_positions,
            "top_heads": [f"L{h.layer}H{h.head}={h.logit_diff:+.3f}" for h in r.top_k_heads],
            "critical_layer": critical,
            "notes": r.notes,
        }

    def _ablate(self, tokens, *, layer: int, head: int | None):
        # head=None -> ablate the whole layer's attention output
        assert self.model is not None
        hook_name = tl_utils.get_act_name("z", layer)

        def hook(value, hook):  # noqa: ARG001
            if head is None:
                return torch.zeros_like(value)
            value[:, :, head, :] = 0.0
            return value

        with torch.no_grad():
            return self.model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])

    def _notes(self, layer_diffs, top, sink_positions, dfg) -> str:
        if not layer_diffs:
            return "no layers swept"
        crit = max(layer_diffs, key=lambda k: abs(layer_diffs[k]))
        head_str = ", ".join(f"L{h.layer}H{h.head}" for h in top[:3]) or "n/a"
        sink_str = (f"{len(sink_positions)} sink-aligned token(s)"
                    if sink_positions else "no sink tokens aligned")
        return (f"swept {len(dfg)} dfg target(s); {sink_str}; "
                f"critical layer {crit} (logit_diff={layer_diffs[crit]:+.3f}); "
                f"top heads: {head_str}")


def expected_random_logit_diff(n_layers: int, n_heads: int) -> float:
    """Order-of-magnitude noise floor used as an upper bound in tests."""
    if n_layers <= 0 or n_heads <= 0:
        raise ValueError("n_layers and n_heads must be positive")
    return 1.0 / math.sqrt(n_layers * n_heads)
