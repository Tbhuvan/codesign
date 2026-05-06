"""codesign command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from codesign import benchmark as bench_mod
from codesign import visualizer
from codesign.attacker import RLAdversary
from codesign.parser import ProgramGraphExtractor

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


class SVDTargetModel:
    """HuggingFace pipeline wrapped as a `code -> [0,1]` callable."""

    def __init__(self, model_id: str = "mrm8488/codebert-base-finetuned-detect-insecure-code") -> None:
        self.model_id = model_id
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ImportError("install with `pip install codesign` to get torch+transformers") from e
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("text-classification", model=self.model_id, device=device)

    def detect_vulnerability_score(self, code: str) -> float:
        self._ensure_loaded()
        # CodeBERT max length is 512.
        out = self._pipe(code[:512])[0]  # type: ignore[index]
        label = out["label"].lower()
        score = float(out["score"])
        if "label_1" in label or "unsafe" in label or "insecure" in label:
            return score
        return 1.0 - score


def cmd_scan(args: argparse.Namespace) -> int:
    from codesign.interpreter import VulnerabilityCircuitProbe

    code = Path(args.file).read_text(encoding="utf-8")
    extractor = ProgramGraphExtractor()
    g = extractor.build(code)
    console.print(
        f"[bold cyan][*][/] {len(g.variables)} unique identifier(s); "
        f"DFG {len(g.dfg)} chain(s); {len(g.sinks)} sink call(s)."
    )

    console.print(f"[bold cyan][*][/] loading {args.model}")
    probe = VulnerabilityCircuitProbe(model_name=args.model, device=args.device)
    probe.load_model()
    report = probe.run_dfg_guided_patching(
        code, g.dfg, top_k=args.top_k, max_layers=args.max_layers
    )
    console.print("[bold green][!][/] interpretability report:")
    for k, v in report.to_dict().items():
        console.print(f"    [dim]{k}[/]: {v}")

    if args.figure:
        path = visualizer.plot_head_heatmap(report, args.figure)
        console.print(f"[green]    -> {path}[/]")
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    code = Path(args.file).read_text(encoding="utf-8")
    extractor = ProgramGraphExtractor()
    g = extractor.build(code)
    console.print(f"[bold cyan][*][/] {len(g.all_variable_nodes)} variable nodes")

    console.print(f"[bold cyan][*][/] loading SVD model {args.target}")
    target = SVDTargetModel(model_id=args.target)

    adv = RLAdversary(
        target_model_score_fn=target.detect_vulnerability_score,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    console.print(f"[bold cyan][*][/] running MAB for {args.max_steps} step(s)")
    adv_code, trace = adv.attack(code, g.all_variable_nodes, max_steps=args.max_steps)

    console.rule("[bold]adversarial sample")
    console.print(adv_code)
    console.rule()
    console.print(f"original: [bold]{trace.steps[0].score_before:.3f}[/]")
    console.print(f"final:    [bold]{trace.best_score:.3f}[/]")
    console.print(f"q-table:  {trace.q_table}")

    if args.trace_json:
        out = Path(args.trace_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"trace": asdict(trace)}, indent=2, default=str))
        console.print(f"[green]    -> {out}[/]")
    if args.figure:
        visualizer.plot_attack_trace(trace, args.figure)
        console.print(f"[green]    -> {args.figure}[/]")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    target = SVDTargetModel(model_id=args.target)
    summary, results = bench_mod.run(
        score_fn=target.detect_vulnerability_score,
        dataset_root=args.dataset,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        epsilon=args.epsilon,
    )
    console.rule("[bold]benchmark summary")
    console.print(bench_mod.render_markdown(summary, results))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    console.rule("[bold]benchmark report")
    for k, v in payload["summary"].items():
        console.print(f"[dim]{k}[/]: {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codesign")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan")
    scan.add_argument("--file", "-f", required=True)
    scan.add_argument("--model", default="EleutherAI/pythia-160m")
    scan.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    scan.add_argument("--top-k", type=int, default=10)
    scan.add_argument("--max-layers", type=int, default=None)
    scan.add_argument("--figure", default=None)
    scan.set_defaults(func=cmd_scan)

    atk = sub.add_parser("attack")
    atk.add_argument("--file", "-f", required=True)
    atk.add_argument("--target", default="mrm8488/codebert-base-finetuned-detect-insecure-code")
    atk.add_argument("--max-steps", type=int, default=15)
    atk.add_argument("--epsilon", type=float, default=0.3)
    atk.add_argument("--seed", type=int, default=0)
    atk.add_argument("--trace-json", default=None)
    atk.add_argument("--figure", default=None)
    atk.set_defaults(func=cmd_attack)

    b = sub.add_parser("benchmark")
    b.add_argument("--dataset", default="data/cwe_samples")
    b.add_argument("--target", default="mrm8488/codebert-base-finetuned-detect-insecure-code")
    b.add_argument("--max-steps", type=int, default=15)
    b.add_argument("--epsilon", type=float, default=0.3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--output", default="experiments/results/benchmark.json")
    b.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("report")
    r.add_argument("--input", default="experiments/results/benchmark.json")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
