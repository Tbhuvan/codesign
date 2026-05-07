"""exp01: MAB attacker over the CWE corpus, optionally across multiple seeds.

Single-seed mode keeps the existing behaviour: writes one
``exp01_attack.json`` summary plus per-sample records.

Multi-seed mode (``--seeds 0 1 2 3 4``) reruns the harness with each
seed and aggregates evasion-rate mean/std across seeds. The per-seed
JSONs are written to ``--output-dir``; an aggregate file with the
mean/std summary lands at ``<output-dir>/aggregate.json``. This is
the form quoted in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from codesign import benchmark
from codesign.cli import SVDTargetModel

EXP_LOG = Path("experiments.json")
log = logging.getLogger(__name__)


def _append_log(record: dict) -> None:
    items: list[dict] = []
    if EXP_LOG.exists():
        try:
            items = json.loads(EXP_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            items = []
    items.append(record)
    EXP_LOG.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _run_single(args, seed: int, out_path: Path) -> dict:
    target = SVDTargetModel(spec=args.target)
    summary, results = benchmark.run(
        score_fn=target.detect_vulnerability_score,
        dataset_root=args.dataset,
        max_steps=args.max_steps,
        seed=seed,
        output=str(out_path),
        epsilon=args.epsilon,
    )
    print(benchmark.render_markdown(summary, results))
    return {
        "seed": seed,
        "summary": summary.to_dict(),
        "result_path": str(out_path),
    }


def _aggregate(per_seed: list[dict]) -> dict:
    """Mean / std across seeds for the headline metrics."""
    if not per_seed:
        return {}

    def collect(key: str, only_finite: bool = False) -> list[float]:
        out = []
        for ps in per_seed:
            v = ps["summary"].get(key)
            if v is None:
                continue
            if isinstance(v, float) and v != v:  # nan
                if only_finite:
                    continue
                else:
                    out.append(float("nan"))
            else:
                out.append(float(v))
        return out

    def stats(values: list[float], finite_only: bool = True) -> dict:
        clean = [v for v in values if not (isinstance(v, float) and v != v)] if finite_only else values
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": statistics.fmean(clean),
            "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
            "n": len(clean),
        }

    return {
        "evasion_rate":       stats(collect("evasion_rate")),
        "evasion_rate_vuln":  stats(collect("evasion_rate_vuln")),
        "mean_confidence_drop": stats(collect("mean_confidence_drop")),
        "parse_validity_rate": stats(collect("parse_validity_rate")),
        "dfg_preservation_rate": stats(collect("dfg_preservation_rate")),
        "mean_steps_to_evade": stats(collect("mean_steps_to_evade")),
        "n_seeds": len(per_seed),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="ollama:qwen3:8b",
                   help="Target spec; pass 'hf:<model_id>' for a HuggingFace classifier")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0,
                   help="Single seed; ignored if --seeds is given")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Multi-seed mode: e.g. --seeds 0 1 2 3 4")
    p.add_argument("--output", default="experiments/results/exp01_attack.json",
                   help="Single-seed output path; ignored if --seeds is given")
    p.add_argument("--output-dir", default="experiments/results/exp01_multiseed",
                   help="Multi-seed output dir")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.seeds is None:
        # legacy single-seed path
        single = _run_single(args, args.seed, Path(args.output))
        _append_log({
            "experiment": "exp01_attack_evaluation",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_model": args.target,
            "dataset": args.dataset,
            "max_steps": args.max_steps,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "evasion_rate_vuln": single["summary"].get("evasion_rate_vuln"),
            "result_path": single["result_path"],
        })
        return 0

    # multi-seed path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict] = []
    for s in args.seeds:
        log.info("=== seed=%d ===", s)
        path = out_dir / f"seed_{s}.json"
        per_seed.append(_run_single(args, s, path))

    agg = _aggregate(per_seed)

    aggregate_path = out_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps({
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": args.target,
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "epsilon": args.epsilon,
        "seeds": args.seeds,
        "aggregate": agg,
        "per_seed": per_seed,
    }, indent=2, default=str), encoding="utf-8")

    print()
    print(f"Aggregated across {len(args.seeds)} seed(s):")
    if agg.get("evasion_rate_vuln"):
        ev = agg["evasion_rate_vuln"]
        print(f"  evasion_rate_vuln: {ev['mean']:.3f} ± {ev['std']:.3f} (n={ev['n']})")
    if agg.get("mean_confidence_drop"):
        cd = agg["mean_confidence_drop"]
        print(f"  mean_confidence_drop: {cd['mean']:+.3f} ± {cd['std']:.3f}")
    if agg.get("dfg_preservation_rate"):
        dp = agg["dfg_preservation_rate"]
        print(f"  dfg_preservation_rate: {dp['mean']:.3f} ± {dp['std']:.3f}")
    print(f"Written to {aggregate_path}")

    _append_log({
        "experiment": "exp01_multiseed",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": args.target,
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "epsilon": args.epsilon,
        "seeds": args.seeds,
        "evasion_rate_vuln_mean":
            agg["evasion_rate_vuln"]["mean"] if agg.get("evasion_rate_vuln") else None,
        "evasion_rate_vuln_std":
            agg["evasion_rate_vuln"]["std"] if agg.get("evasion_rate_vuln") else None,
        "result_path": str(aggregate_path),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
