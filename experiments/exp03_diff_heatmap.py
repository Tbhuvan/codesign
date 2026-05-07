"""exp03: clean-vs-adversarial diff heatmap.

For each sample we:
  1. parse clean code, run the activation patching sweep against the
     loaded probe model
  2. run the MAB attack against a real SVD classifier (CodeBERT)
  3. parse the adversarial code, re-run the same sweep
  4. compute the per-head |logit_diff_clean| - |logit_diff_adv| matrix
  5. save figures and a JSON record

The point is to ask: when the MAB succeeds, does the SVD circuit at
some L_i H_j go *dark* in the adversarial heatmap? If so, that's
preliminary causal evidence that L_i H_j carried the SVD decision.

The probe model is loaded once and reused across the clean and
adversarial passes (both for the same sample, and across samples in
one run). This is the only practical way to keep the wall-clock
under an hour on CPU.
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
from codesign.interpreter import VulnerabilityCircuitProbe
from codesign.parser import ProgramGraphExtractor
from codesign.visualizer import plot_diff_heatmap, plot_head_heatmap

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--svd-target", default="ollama:qwen3:8b",
                   help="Target spec; pass 'hf:<model_id>' for a HuggingFace classifier")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--samples", nargs="+", default=None,
                   help="Sample ids to run; default = first --n-samples vulnerable ones")
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--max-layers", type=int, default=14,
                   help="Cap probe sweep depth to keep CPU runtime tractable")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out-dir", default="experiments/results/exp03_diff")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pick the subset to run on
    all_samples = load_samples(args.dataset)
    if args.samples:
        wanted = set(args.samples)
        chosen = [s for s in all_samples if s.id in wanted]
    else:
        chosen = [s for s in all_samples if s.vulnerable][: args.n_samples]
    if not chosen:
        log.error("no samples matched")
        return 1

    log.info("running on %d sample(s): %s", len(chosen), [s.id for s in chosen])

    # 2. Load the probe (one time, ~1 minute)
    log.info("loading probe model %s", args.probe_model)
    t0 = time.time()
    probe = VulnerabilityCircuitProbe(model_name=args.probe_model, device=args.device)
    probe.load_model()
    log.info("probe loaded in %.1fs", time.time() - t0)

    # 3. Load the SVD target (CodeBERT classifier for the attack reward)
    log.info("loading SVD target %s", args.svd_target)
    t0 = time.time()
    target = SVDTargetModel(spec=args.svd_target)
    # Force-load the pipeline now so the first attack iteration doesn't
    # silently pay the cost.
    _ = target.detect_vulnerability_score("def f():\n    return 1\n")
    log.info("svd target ready in %.1fs", time.time() - t0)

    extractor = ProgramGraphExtractor()
    summary_records: list[dict] = []

    for s in chosen:
        log.info("=== %s (%s) ===", s.id, s.cwe)
        sample_dir = out_dir / s.id
        sample_dir.mkdir(exist_ok=True)

        # Clean-code circuit pass
        log.info("[%s] clean pass", s.id)
        t0 = time.time()
        clean_graph = extractor.build(s.code)
        clean_report = probe.run_dfg_guided_patching(
            s.code, clean_graph.dfg, top_k=args.top_k, max_layers=args.max_layers
        )
        log.info("[%s] clean pass done in %.1fs", s.id, time.time() - t0)

        # MAB attack against the real SVD model
        log.info("[%s] attacking", s.id)
        t0 = time.time()
        adv = RLAdversary(
            target.detect_vulnerability_score, epsilon=args.epsilon, seed=args.seed
        )
        adv_code, trace = adv.attack(
            s.code, clean_graph.all_variable_nodes, max_steps=args.max_steps
        )
        log.info(
            "[%s] attack done in %.1fs; score %.3f -> %.3f over %d step(s)",
            s.id,
            time.time() - t0,
            trace.steps[0].score_before if trace.steps else float("nan"),
            trace.best_score,
            len(trace.steps),
        )

        # Adversarial-code circuit pass (only if the attack actually changed anything)
        if adv_code == s.code:
            log.warning("[%s] attack produced no change; skipping adv pass", s.id)
            continue

        log.info("[%s] adv pass", s.id)
        t0 = time.time()
        adv_graph = extractor.build(adv_code)
        adv_report = probe.run_dfg_guided_patching(
            adv_code, adv_graph.dfg, top_k=args.top_k, max_layers=args.max_layers
        )
        log.info("[%s] adv pass done in %.1fs", s.id, time.time() - t0)

        # Plots
        clean_png = plot_head_heatmap(clean_report, sample_dir / "clean_heatmap.png")
        adv_png = plot_head_heatmap(adv_report, sample_dir / "adv_heatmap.png")
        diff_png = plot_diff_heatmap(
            clean_report, adv_report, sample_dir / "diff_heatmap.png",
            title=f"{s.id} ({s.cwe}): clean - adv |logit_diff|",
        )

        # Per-sample JSON record
        record = {
            "sample_id": s.id,
            "cwe": s.cwe,
            "probe_model": args.probe_model,
            "svd_target": args.svd_target,
            "max_layers": args.max_layers,
            "score_clean": (
                trace.steps[0].score_before if trace.steps else None
            ),
            "score_adv": trace.best_score,
            "evaded": trace.best_score < 0.5,
            "attack_steps": len(trace.steps),
            "mutations_applied": [step.strategy for step in trace.steps],
            "clean_top_heads": [
                {"layer": h.layer, "head": h.head, "logit_diff": h.logit_diff}
                for h in clean_report.top_k_heads
            ],
            "adv_top_heads": [
                {"layer": h.layer, "head": h.head, "logit_diff": h.logit_diff}
                for h in adv_report.top_k_heads
            ],
            "clean_ablation_mode": clean_report.ablation_mode,
            "adv_ablation_mode": adv_report.ablation_mode,
            "clean_sink_positions": clean_report.sink_token_positions,
            "adv_sink_positions": adv_report.sink_token_positions,
            "clean_report": clean_report.to_dict(),
            "adv_report": adv_report.to_dict(),
            "trace": asdict(trace),
            "figures": {
                "clean": str(clean_png),
                "adv": str(adv_png),
                "diff": str(diff_png),
            },
        }
        (sample_dir / "record.json").write_text(json.dumps(record, indent=2, default=str))
        summary_records.append({
            "sample_id": s.id,
            "cwe": s.cwe,
            "score_clean": record["score_clean"],
            "score_adv": record["score_adv"],
            "evaded": record["evaded"],
            "attack_steps": record["attack_steps"],
        })

    summary = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_model": args.probe_model,
        "svd_target": args.svd_target,
        "max_layers": args.max_layers,
        "n_samples_run": len(summary_records),
        "samples": summary_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
