"""exp00: does the target classifier actually have signal?

Before treating any classifier as ground truth for the MAB attacker,
we run it against a hand-curated probe set of unambiguous safe and
vulnerable Python snippets and compute calibration metrics. A target
that scores < ~0.7 accuracy on this set is producing noise; using
it as the SVD signal would make the rest of the experiment chain
meaningless.

This is itself a methodological contribution: many SVD-attack papers
report evasion rates without showing that the target classifier
labels their probe inputs correctly to begin with. We make this step
explicit and ship the data.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from codesign.targets import make_target

log = logging.getLogger(__name__)


def load_probe(path: str | Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["vulnerable"], data["safe"]


def calibrate(
    target_spec: str,
    probe_path: str | Path = "data/calibration_probe.json",
    threshold: float = 0.5,
) -> dict:
    target = make_target(target_spec)
    vuln_items, safe_items = load_probe(probe_path)

    rows: list[dict] = []
    tp = fp = tn = fn = 0
    for item in vuln_items:
        s = target.score(item["code"])
        is_pred_vuln = s >= threshold
        if is_pred_vuln:
            tp += 1
        else:
            fn += 1
        rows.append({
            "id": item["id"],
            "true_label": "vuln",
            "score": s,
            "predicted_vuln": is_pred_vuln,
        })
    for item in safe_items:
        s = target.score(item["code"])
        is_pred_vuln = s >= threshold
        if is_pred_vuln:
            fp += 1
        else:
            tn += 1
        rows.append({
            "id": item["id"],
            "true_label": "safe",
            "score": s,
            "predicted_vuln": is_pred_vuln,
        })

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "target": target_spec,
        "info": target.info(),
        "n_total": n,
        "n_vuln": tp + fn,
        "n_safe": tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "items": rows,
    }


def render_table(reports: list[dict]) -> str:
    rows = [
        "| Target | Accuracy | Recall | Precision | Specificity | F1 | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        verdict = "usable" if r["accuracy"] >= 0.7 else "noise"
        rows.append(
            f"| `{r['target']}` | {r['accuracy']:.2f} | {r['recall']:.2f} | "
            f"{r['precision']:.2f} | {r['specificity']:.2f} | {r['f1']:.2f} | {verdict} |"
        )
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        nargs="+",
        default=[
            "ollama:qwen3:8b",
            "ollama:qwen2.5:3b",
            "hf:mrm8488/codebert-base-finetuned-detect-insecure-code",
        ],
        help="One or more target specs to evaluate",
    )
    p.add_argument("--probe", default="data/calibration_probe.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", default="experiments/results/exp00_calibration.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reports: list[dict] = []
    for spec in args.targets:
        log.info("calibrating %s", spec)
        try:
            r = calibrate(spec, args.probe, threshold=args.threshold)
        except Exception as e:
            log.warning("  failed: %s", e)
            reports.append({"target": spec, "error": str(e)})
            continue
        log.info(
            "  accuracy=%.2f  recall=%.2f  precision=%.2f  specificity=%.2f",
            r["accuracy"], r["recall"], r["precision"], r["specificity"],
        )
        reports.append(r)

    summary = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_path": str(args.probe),
        "threshold": args.threshold,
        "reports": reports,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out)

    print()
    print(render_table([r for r in reports if "error" not in r]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
