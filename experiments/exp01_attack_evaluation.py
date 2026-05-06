"""exp01: run the MAB attacker against CodeBERT over the CWE corpus."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from codesign import benchmark
from codesign.cli import SVDTargetModel

EXP_LOG = Path("experiments.json")


def _append(record: dict) -> None:
    log: list[dict] = []
    if EXP_LOG.exists():
        try:
            log = json.loads(EXP_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []
    log.append(record)
    EXP_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="mrm8488/codebert-base-finetuned-detect-insecure-code")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/results/exp01_attack.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = SVDTargetModel(model_id=args.target)

    summary, results = benchmark.run(
        score_fn=target.detect_vulnerability_score,
        dataset_root=args.dataset,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        epsilon=args.epsilon,
    )
    print(benchmark.render_markdown(summary, results))

    _append({
        "experiment": "exp01_attack_evaluation",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": args.target,
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "n_samples": summary.n_samples,
        "evasion_rate": summary.evasion_rate,
        "mean_confidence_drop": summary.mean_confidence_drop,
        "mutation_diversity": summary.mutation_diversity,
        "result_path": args.output,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
