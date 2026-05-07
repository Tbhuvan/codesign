"""Evaluation metrics for attack and interpreter runs."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass
class AttackResult:
    sample_id: str
    cwe: str | None
    original_score: float
    final_score: float
    steps_taken: int
    evaded: bool
    parse_valid: bool
    dfg_preserved: bool
    mutations_applied: list[str]
    final_code: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    n_samples: int
    n_vulnerable: int
    n_correctly_flagged: int  # vuln samples the target initially scored >= 0.5
    evasion_rate: float          # legacy: all-samples evaded / n_samples
    evasion_rate_vuln: float     # honest: correctly-flagged-and-evaded / n_correctly_flagged
    mean_confidence_drop: float
    mean_steps_to_evade: float
    parse_validity_rate: float
    dfg_preservation_rate: float
    mutation_diversity: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_valid(code: str) -> bool:
    if not isinstance(code, str):
        raise TypeError("code must be str")
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def dfg_preserved(
    original_dfg: dict[str, set[str]],
    mutated_dfg: dict[str, set[str]],
    rename_map: dict[str, str] | None = None,
) -> bool:
    """Check that two DFGs are isomorphic up to a renaming."""
    if rename_map:
        renamed: dict[str, set[str]] = {}
        for tgt, srcs in original_dfg.items():
            new_tgt = rename_map.get(tgt, tgt)
            renamed[new_tgt] = {rename_map.get(s, s) for s in srcs}
        original_dfg = renamed

    if set(original_dfg.keys()) != set(mutated_dfg.keys()):
        return False
    for k in original_dfg:
        if original_dfg[k] != mutated_dfg.get(k, set()):
            return False
    return True


def confidence_drop(original: float, final: float) -> float:
    if not (0.0 <= original <= 1.0 and 0.0 <= final <= 1.0):
        raise ValueError("scores must be in [0, 1]")
    return max(-1.0, min(1.0, original - final))


def evaded(final_score: float, threshold: float = 0.5) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return final_score < threshold


def summarise(results: Sequence[AttackResult]) -> BenchmarkSummary:
    """Aggregate per-sample results into a benchmark summary.

    Two evasion rates are reported:

    - ``evasion_rate``      naive: all results with final_score < 0.5
    - ``evasion_rate_vuln`` honest: of the *vulnerable* samples the
                            target initially flagged correctly
                            (original_score >= 0.5), the fraction the
                            attacker drove below 0.5

    The honest rate is what should be quoted in the paper; the naive
    rate is kept for back-compat and to surface if a safe control is
    accidentally counted as "evaded" (e.g., score was already < 0.5).
    """
    if not results:
        raise ValueError("empty results")

    n = len(results)
    successes = [r for r in results if r.evaded]
    diversity: dict[str, int] = {}
    for r in results:
        for m in r.mutations_applied:
            diversity[m] = diversity.get(m, 0) + 1

    vuln_results = [r for r in results if r.cwe and r.cwe.upper() != "NONE"]
    correctly_flagged = [r for r in vuln_results if r.original_score >= 0.5]
    real_evasions = [r for r in correctly_flagged if r.final_score < 0.5]

    return BenchmarkSummary(
        n_samples=n,
        n_vulnerable=len(vuln_results),
        n_correctly_flagged=len(correctly_flagged),
        evasion_rate=len(successes) / n,
        evasion_rate_vuln=(
            len(real_evasions) / len(correctly_flagged)
            if correctly_flagged else float("nan")
        ),
        mean_confidence_drop=sum(r.original_score - r.final_score for r in results) / n,
        mean_steps_to_evade=(
            sum(r.steps_taken for r in real_evasions) / len(real_evasions)
            if real_evasions else float("nan")
        ),
        parse_validity_rate=sum(1 for r in results if r.parse_valid) / n,
        dfg_preservation_rate=sum(1 for r in results if r.dfg_preserved) / n,
        mutation_diversity=diversity,
    )


def head_importance_logit_diff(
    clean_logit: float,
    ablated_logits: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Per-head logit_diff = clean - ablated."""
    if not isinstance(clean_logit, (int, float)):
        raise TypeError("clean_logit must be numeric")
    out: dict[tuple[int, int], float] = {}
    for (layer, head), ablated in ablated_logits.items():
        if not (isinstance(layer, int) and isinstance(head, int)):
            raise TypeError("keys must be (int, int)")
        out[(layer, head)] = float(clean_logit - ablated)
    return out
