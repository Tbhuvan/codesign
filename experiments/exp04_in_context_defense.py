"""exp04: does in-context learning defense raise the bar for problem-space attacks?

We treat an LLM (served via Ollama) as the SVD classifier. For each
sample we run the MAB attack twice: once with a 0-shot prompt, once
with a 3-shot prompt that includes a fixed set of (code, label) pairs.
If ICL hardens the model, the 3-shot variant should require more
attack steps to evade, or fail to evade at all.

This is the third bullet of WP3 in the position description:
"in-context learning techniques to overcome the limited input windows
of LLMs". We don't claim a publishable defense — only that the
methodology is wired in and produces measurable deltas.

The Ollama API is used directly via HTTP rather than via an Ollama
client library, to avoid adding another dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from codesign.attacker import RLAdversary
from codesign.dataset import load_samples
from codesign.parser import ProgramGraphExtractor
from codesign.targets import OllamaPromptClassifier

log = logging.getLogger(__name__)


# Three (code, label) demonstrations for the few-shot prompt.
# Kept short and concrete; the model anchors on these examples.
SHOTS: list[tuple[str, str]] = [
    (
        'import os\n'
        'def run(user_input):\n'
        '    os.system("ping " + user_input)\n',
        "VULN",
    ),
    (
        'def add(a, b):\n'
        '    return a + b\n',
        "SAFE",
    ),
    (
        'import sqlite3\n'
        'def lookup(name):\n'
        '    conn = sqlite3.connect("u.db")\n'
        '    conn.execute("SELECT * FROM u WHERE n = ?", (name,))\n',
        "SAFE",
    ),
]


def make_classifier(
    model: str,
    n_shots: int,
    base_url: str = "http://localhost:11434",
    timeout: float = 60.0,
) -> Callable[[str], float]:
    """Return a callable code -> [0, 1] vuln confidence."""
    if not 0 <= n_shots <= len(SHOTS):
        raise ValueError(f"n_shots must be in [0, {len(SHOTS)}]")
    clf = OllamaPromptClassifier(
        model=model, base_url=base_url, timeout=timeout,
        n_shots=n_shots, shots=SHOTS,
    )
    return clf.score


def run_one(
    sample,
    model: str,
    n_shots: int,
    extractor: ProgramGraphExtractor,
    max_steps: int,
    epsilon: float,
    seed: int,
) -> dict:
    score_fn = make_classifier(model, n_shots)
    graph = extractor.build(sample.code)
    t0 = time.time()
    base_score = score_fn(sample.code)
    adv = RLAdversary(score_fn, epsilon=epsilon, seed=seed)
    _, trace = adv.attack(sample.code, graph.all_variable_nodes, max_steps=max_steps)
    elapsed = time.time() - t0

    evaded_step: int | None = None
    for i, step in enumerate(trace.steps):
        if step.score_after < 0.5:
            evaded_step = i
            break

    return {
        "sample_id": sample.id,
        "cwe": sample.cwe,
        "model": model,
        "n_shots": n_shots,
        "base_score": base_score,
        "final_score": trace.best_score,
        "evaded": trace.best_score < 0.5,
        "first_evade_step": evaded_step,
        "total_steps": len(trace.steps),
        "elapsed_seconds": round(elapsed, 1),
        "trace_q_table": trace.q_table,
        "mutations": [s.strategy for s in trace.steps],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--dataset", default="data/cwe_samples")
    p.add_argument("--samples", nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--shots", nargs="+", type=int, default=[0, 3])
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="experiments/results/exp04_icl.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    all_samples = load_samples(args.dataset)
    if args.samples:
        wanted = set(args.samples)
        chosen = [s for s in all_samples if s.id in wanted]
    else:
        chosen = [s for s in all_samples if s.vulnerable][: args.n_samples]
    if not chosen:
        log.error("no samples matched")
        return 1

    log.info(
        "running %d sample(s) x %d shot config(s) against %s",
        len(chosen), len(args.shots), args.model,
    )
    extractor = ProgramGraphExtractor()
    results: list[dict] = []
    for s in chosen:
        for k in args.shots:
            log.info("[%s] n_shots=%d", s.id, k)
            r = run_one(s, args.model, k, extractor,
                        args.max_steps, args.epsilon, args.seed)
            log.info(
                "  -> base=%.2f final=%.2f evaded=%s steps=%d",
                r["base_score"], r["final_score"], r["evaded"], r["total_steps"],
            )
            results.append(r)

    # Aggregate: per-shot evasion rate and mean steps to first evade.
    by_shot: dict[int, dict] = {}
    for r in results:
        b = by_shot.setdefault(r["n_shots"], {"n": 0, "evaded": 0, "steps": []})
        b["n"] += 1
        if r["evaded"]:
            b["evaded"] += 1
            if r["first_evade_step"] is not None:
                b["steps"].append(r["first_evade_step"])

    summary = {
        "model": args.model,
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by_shot": {
            str(k): {
                "n": v["n"],
                "evasion_rate": v["evaded"] / v["n"] if v["n"] else 0.0,
                "mean_first_evade_step": (
                    sum(v["steps"]) / len(v["steps"]) if v["steps"] else None
                ),
            }
            for k, v in by_shot.items()
        },
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out)
    log.info("by_shot: %s", json.dumps(summary["by_shot"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
