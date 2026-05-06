"""exp03: scaffolding for the robustness-persistence study.

Records pre-attack and post-attack scores. The fine-tuning loops are
not run here (need GPU); see notebooks/robustness_forgetting.ipynb.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from codesign.attacker import RLAdversary
from codesign.cli import SVDTargetModel
from codesign.dataset import load_samples
from codesign.metrics import AttackResult, evaded, parse_valid, summarise
from codesign.parser import ProgramGraphExtractor


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="mrm8488/codebert-base-finetuned-detect-insecure-code")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/results/exp03_persistence.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = SVDTargetModel(model_id=args.target)
    score_fn = target.detect_vulnerability_score

    samples = load_samples(args.dataset)
    extractor = ProgramGraphExtractor()
    baseline = {s.id: float(score_fn(s.code)) for s in samples}

    results: list[AttackResult] = []
    for s in samples:
        g = extractor.build(s.code)
        adv = RLAdversary(score_fn, epsilon=args.epsilon, seed=args.seed)
        adv_code, trace = adv.attack(s.code, g.all_variable_nodes, max_steps=args.max_steps)
        results.append(AttackResult(
            sample_id=s.id,
            cwe=s.cwe,
            original_score=baseline[s.id],
            final_score=trace.best_score,
            steps_taken=len(trace.steps),
            evaded=evaded(trace.best_score),
            parse_valid=parse_valid(adv_code),
            dfg_preserved=False,
            mutations_applied=[step.strategy for step in trace.steps],
            final_code=adv_code,
        ))

    summary = summarise(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment": "exp03_robustness_persistence_poc",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": args.target,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "baseline_scores": baseline,
        "summary": summary.to_dict(),
        "results": [asdict(r) for r in results],
        "todo": [
            "robustness fine-tune (needs GPU)",
            "continual fine-tune on non-security task",
            "re-score post-continual",
        ],
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
