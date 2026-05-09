# CodeSign

Adversarial attacks and mechanistic interpretability for code-LLM
vulnerability detectors.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Reproduction recipes are in [`experiments/`](experiments/README.md).

## What it does

Two analyses, one structural ground truth.

- **Attacker** (`codesign attack`). Multi-armed bandit over eight
  semantics-preserving mutators that cover the natural-developer-
  practice space the WP3 description calls for:
    1. variable renaming with `_advNNN` suffixes (byte-precise via tree-sitter)
    2. natural identifier renaming (context-likely names like `idx`, `command`, `secret`)
    3. dead-code insertion (no-op assignment after every `def`)
    4. control-flow flattening (`if True:` wrap)
    5. equivalent expression substitution (`not (a == b)` <-> `a != b`, `a + 0` <-> `a`, ...)
    6. docstring insertion (neutral one-line developer phrasing)
    7. inline comment insertion (neutral developer-style trailing comments)
    8. type annotations added (`: str` on untyped parameters)

  Default target is an LLM-as-classifier (`ollama:qwen3:8b`); a
  HuggingFace discriminative classifier can be selected with
  `--target hf:<model_id>`. Targets are gated by the
  [calibration probe (exp00)](experiments/exp00_target_calibration.py)
  before they are used as ground truth.
- **Interpreter** (`codesign scan`). Loads a code-trained model via
  TransformerLens (default `Qwen/Qwen2.5-Coder-1.5B-Instruct`),
  runs a `(layer, head)` zero-ablation sweep with per-head ablation
  restricted to DFG-anchored token positions when alignment succeeds,
  and ranks heads by `clean - ablated` logit difference. Emits the
  layer×head heatmap.

Both share a tree-sitter parser that produces an AST, an
assignment-based DFG, a coarse CFG, and a list of dangerous-call sinks.
The DFG is the structural primitive the attacker preserves and the
interpreter anchors on.

## Quickstart

```bash
git clone https://github.com/Tbhuvan/codesign.git
cd codesign
pip install -e ".[dev,notebook]"

# DFG extraction + activation patching
codesign scan -f data/cwe_samples/cwe_078_os_command_injection.py \
              --top-k 10 --figure figures/heatmap_cwe78.png

# MAB attack against real CodeBERT
codesign attack -f data/cwe_samples/cwe_089_sql_injection.py \
                --max-steps 15 --figure figures/attack_cwe89.png

# Full benchmark over the shipped corpus
codesign benchmark --output experiments/results/benchmark.json
```

## Layout

```
codesign/
├── data/
│   ├── cwe_samples/              12 CWE-labelled Python samples
│   └── calibration_probe.json    20-item safe/vuln probe for exp00
├── experiments/                  exp00-04 reproducible scripts
├── notebooks/                    robustness_forgetting.ipynb
├── src/codesign/
│   ├── parser.py                 tree-sitter AST/DFG/CFG/sinks
│   ├── attacker.py               MAB + 8 mutators
│   ├── interpreter.py            DFG-guided activation patching
│   ├── targets.py                target abstraction (Ollama / HF / heuristic)
│   ├── metrics.py                evasion / preservation / head-importance
│   ├── benchmark.py              end-to-end harness
│   ├── visualizer.py             matplotlib helpers (incl. diff heatmap)
│   ├── dataset.py                sample loader
│   └── cli.py                    `codesign scan|attack|benchmark|report`
└── tests/                        pytest suite
```

## Experiments

| ID | Question | Wall-clock |
|---|---|---|
| **exp00** | **Does the candidate target classifier label our probe set correctly?** | ~3 min for 3 targets |
| exp01 | MAB evasion rate against the calibrated target | ~20 min on 12 samples |
| exp02 | Head-importance heatmap on clean code | ~10–30 min/sample CPU |
| exp03 | Clean-vs-adversarial diff heatmap | ~30 min/sample CPU |
| exp04 | Does k-shot ICL harden Ollama-served LLMs against the MAB? | ~15 min for 4 samples × 2 shot configs |

**exp00 is gating** - running attacks without first verifying the target
isn't noise produces meaningless evasion rates. The current
calibration scoreboard is in
[`experiments/results/exp00_calibration.json`](experiments/results/exp00_calibration.json).

## Threat model

White-box query access to a vulnerability classifier returning a score
in `[0, 1]`. The adversary cannot retrain the model; queries are
unbounded. Goal: drive the score below 0.5 while keeping (a) the AST
valid and (b) the DFG isomorphic up to a renaming. Compile- and
runtime-equivalence are out of scope; this matches the threat model
used by ALERT (arXiv:2201.08698) and VRTG.

## Reproducibility

| Flag | Default | Notes |
|---|---|---|
| `--seed` | 0 | Pins the MAB RNG. |
| `--max-steps` | 15 | Attack budget per sample. |
| `--epsilon` | 0.3 | Exploration rate. |
| `--target` | `ollama:qwen3:8b` | SVD classifier spec; also accepts `hf:<model_id>` and `heuristic`. Calibrated via exp00. |
| `--model` (scan) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Probe model - code-trained Qwen, ~3 GB on CPU. |

The CodeBERT-driven attack benchmark on 12 samples runs in ~5 min on
CPU. The full TransformerLens head-sweep on Qwen-Coder-1.5B is much
slower on CPU (~5–10 sec per ablation × 28 layers × 12 heads); use
`--max-layers 8` to constrain it for laptop runs, or use the smaller
`Qwen/Qwen2.5-Coder-0.5B-Instruct` variant.

## Numbers

This README ships no pre-computed results. Run `codesign benchmark`
locally; the JSON schema is stable. Reasoning: hard-coded numbers in a
README rot on every change to the harness, and a result that depends
on a model download and a seed deserves to be reproduced.

## Comparison

| | ALERT | MHM | VRTG | CodeSign |
|---|---|---|---|---|
| Problem-space mutations | y | y | y | y |
| AST-safe rewriting | n | partial | y | y |
| DFG-equivalence post-check | n | n | y | y |
| RL/MAB scheduling | n | n | GA | MAB |
| Mech-interp coupling | n | n | n | y |

## Tests

```bash
pytest -m "not slow"                      # ~2s, no model downloads
pytest -m slow                            # full sweep, needs transformer_lens
ruff check src tests experiments
```

## Limitations

- DFG-equivalence is a structural check, not behavioural. A mutator
  could in principle preserve the DFG but break runtime semantics.
  Same gap as in ALERT/MHM/VRTG.
- **The most popular public HuggingFace "insecure code" classifier
  is out-of-distribution on Python.** Our calibration probe (exp00)
  scored `mrm8488/codebert-base-finetuned-detect-insecure-code` at
  0.50 accuracy / 0.0 specificity on a 20-item balanced Python
  probe. Spot-checks confirm it scores C/C++ defects sensibly
  (`*null = 5;` -> 0.995 vuln) but mislabels `def add(a, b):
  return a + b` at 0.69 vuln and `eval(input())` at 0.35 vuln. The
  classifier isn't broken; it's a C/C++-trained model being asked
  to label Python. The probe doesn't claim "broken classifier";
  it claims "this isn't a usable SVD target for this corpus
  without language-matched fine-tuning". We default to
  `ollama:qwen3:8b` (0.80 accuracy on the same probe) and leave
  the HF target accessible via `--target hf:<id>` for users with
  a calibrated discriminative classifier. See
  `experiments/results/exp00_calibration.json` for the full
  scoreboard.
- The default probe is a generative code LLM, not a binary SVD
  classifier. The patching metric is max-logit at the final position
  rather than a class-logit. For full WP2 fidelity, swap in a
  fine-tuned SVD classifier of the same architecture; the
  `FINE_TUNE_TO_BASE` override pattern in `interpreter.py` makes this
  one line of config.
- Sample corpus is small. Point `--dataset` at a Devign / BigVul /
  DiverseVul export with the same `.py` + `.json` convention to scale.

## Ethics

CodeSign is a defensive research artifact. The goal is *measuring*
the robustness of LLM-based SVD pipelines, not facilitating evasion
of deployed systems.

The mutators we ship are well-documented attack primitives in the
academic literature (ALERT, MHM, VRTG, CODEBREAKER). Releasing an
open implementation does not extend the attack surface; it makes
evaluation reproducible.

We do not ship techniques that push the artifact toward
malware-flavored obfuscation (Unicode homoglyphs, string-encoding
obfuscation, dead-branch exploit scaffolding, tokenizer-evasion).
Those exist in the literature and may be appropriate in a
separate red-team-only tool with coordinated-disclosure norms; not
here.

The calibration probe (exp00) is itself a defensive contribution:
it tells SVD-pipeline operators when their classifier is at chance
level on a balanced probe set. Our finding that the most-popular
public Python SVD classifier scores 0.50 / 0.0 specificity is more
useful to defenders than to attackers.

If a CodeSign run on a specific production SVD product reveals a
systematic evasion path: coordinated disclosure to the vendor, not
public reporting.

## License

Apache 2.0.
