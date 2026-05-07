"""End-to-end attack benchmark harness."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codesign.attacker import RLAdversary
from codesign.dataset import Sample, load_samples
from codesign.metrics import (
    AttackResult,
    BenchmarkSummary,
    dfg_preserved,
    evaded,
    parse_valid,
    summarise,
)
from codesign.parser import ProgramGraphExtractor

log = logging.getLogger(__name__)


def _recover_rename_map(original: str, mutated: str, names: list[str]) -> dict[str, str]:
    # Best-effort: VRTG appends `_advNNN` so we can find the new name by searching.
    out: dict[str, str] = {}
    for var in sorted(set(names), key=len, reverse=True):
        idx = mutated.find(f"{var}_adv")
        if idx >= 0:
            cand = mutated[idx:idx + len(var) + len("_adv") + 3]
            if cand.startswith(f"{var}_adv"):
                out[var] = cand
    return out


def run(
    score_fn: Any,
    samples: list[Sample] | None = None,
    *,
    dataset_root: str | Path = "data/cwe_samples",
    max_steps: int = 15,
    seed: int = 0,
    output: str | Path = "experiments/results/benchmark.json",
    epsilon: float = 0.3,
) -> tuple[BenchmarkSummary, list[AttackResult]]:
    if samples is None:
        samples = load_samples(dataset_root)
    extractor = ProgramGraphExtractor()
    results: list[AttackResult] = []

    for s in samples:
        log.info("attacking %s (%s)", s.id, s.cwe)
        graph = extractor.build(s.code)
        adv = RLAdversary(score_fn, epsilon=epsilon, seed=seed)
        original = float(score_fn(s.code))
        adv_code, trace = adv.attack(s.code, graph.all_variable_nodes, max_steps=max_steps)

        ok = parse_valid(adv_code)
        try:
            mutated_dfg = extractor.extract_dfg(adv_code) if ok else {}
        except Exception:
            mutated_dfg = {}
        rename = _recover_rename_map(s.code, adv_code, list(graph.variables.keys()))
        preserved = dfg_preserved(graph.dfg, mutated_dfg, rename_map=rename) if ok else False

        results.append(AttackResult(
            sample_id=s.id,
            cwe=s.cwe,
            original_score=original,
            final_score=trace.best_score,
            steps_taken=len(trace.steps),
            evaded=evaded(trace.best_score),
            parse_valid=ok,
            dfg_preserved=preserved,
            mutations_applied=[step.strategy for step in trace.steps],
            final_code=adv_code,
        ))

    summary = summarise(results)
    _persist(summary, results, output)
    return summary, results


def _persist(summary: BenchmarkSummary, results: list[AttackResult], output: str | Path) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary.to_dict(),
        "results": [asdict(r) for r in results],
    }
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("wrote %s", p)


def render_markdown(summary: BenchmarkSummary, results: list[AttackResult]) -> str:
    vuln_pct = (
        f"{summary.evasion_rate_vuln * 100:.1f}%"
        if summary.n_correctly_flagged else "n/a"
    )
    rows = [
        "## Benchmark Results",
        "",
        f"- Samples: **{summary.n_samples}** "
        f"(vulnerable: {summary.n_vulnerable}, "
        f"correctly flagged at start: {summary.n_correctly_flagged})",
        f"- **Evasion rate (correctly-flagged vuln subset): {vuln_pct}**",
        f"- Naive rate (all samples below threshold): "
        f"{summary.evasion_rate * 100:.1f}%",
        f"- Mean confidence drop: **{summary.mean_confidence_drop:+.3f}**",
        f"- Mean steps to evade: **{summary.mean_steps_to_evade:.1f}**",
        f"- Parse validity: **{summary.parse_validity_rate * 100:.1f}%**",
        f"- DFG preservation: **{summary.dfg_preservation_rate * 100:.1f}%**",
        "",
        "| Sample | CWE | Orig | Final | Drop | Steps | Evaded | Parse | DFG |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        d = r.original_score - r.final_score
        rows.append(
            f"| `{r.sample_id}` | {r.cwe} | {r.original_score:.3f} | "
            f"{r.final_score:.3f} | {d:+.3f} | {r.steps_taken} | "
            f"{'y' if r.evaded else 'n'} | "
            f"{'y' if r.parse_valid else 'n'} | "
            f"{'y' if r.dfg_preserved else 'n'} |"
        )
    return "\n".join(rows)
