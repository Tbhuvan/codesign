"""exp02: DFG-guided activation patching, one report per sample."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from codesign.dataset import load_samples
from codesign.interpreter import VulnerabilityCircuitProbe
from codesign.parser import ProgramGraphExtractor
from codesign.visualizer import plot_head_heatmap


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--out-dir", default="experiments/results/exp02_circuits")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = ProgramGraphExtractor()
    probe = VulnerabilityCircuitProbe(model_name=args.model, device=args.device)
    probe.load_model()

    records: list[dict] = []
    for s in load_samples(args.dataset):
        if not s.vulnerable:
            continue
        logging.info("patching %s (%s)", s.id, s.cwe)
        g = extractor.build(s.code)
        report = probe.run_dfg_guided_patching(
            s.code, g.dfg, top_k=args.top_k, max_layers=args.max_layers
        )

        json_path = out / f"{s.id}.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        png_path = out / f"{s.id}.png"
        plot_head_heatmap(report, png_path)

        records.append({
            "sample_id": s.id,
            "cwe": s.cwe,
            "top_head": (
                f"L{report.top_k_heads[0].layer}H{report.top_k_heads[0].head}"
                if report.top_k_heads else "n/a"
            ),
            "max_logit_diff": (report.top_k_heads[0].logit_diff if report.top_k_heads else 0.0),
            "json": str(json_path),
            "figure": str(png_path),
        })

    summary = out / "summary.json"
    summary.write_text(json.dumps({
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_samples": len(records),
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
