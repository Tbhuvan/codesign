# CodeSign

Adversarial attacks and mechanistic interpretability for code-LLM
vulnerability detectors.

[![CI](https://github.com/Tbhuvan/codesign/actions/workflows/ci.yml/badge.svg)](https://github.com/Tbhuvan/codesign/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

The long-form write-up is in [`paper/codesign.md`](paper/codesign.md).
Reproduction recipes are in [`experiments/`](experiments/README.md).

## What it does

Two analyses, one structural ground truth.

- **Attacker** (`codesign attack`). Multi-armed bandit over three
  semantics-preserving mutators: variable renaming via byte-precise
  tree-sitter ranges, dead-code insertion, control-flow flattening.
  Validated against a real HuggingFace classifier
  (`mrm8488/codebert-base-finetuned-detect-insecure-code`).
- **Interpreter** (`codesign scan`). Loads a small TransformerLens model
  (default `EleutherAI/pythia-160m`), runs an exhaustive
  `(layer, head)` zero-ablation sweep, and ranks heads by
  `clean - ablated` logit difference. Emits the layer×head heatmap
  used in the paper.

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
├── data/cwe_samples/             12 CWE-labelled Python samples
├── experiments/                  exp01-03 reproducible scripts
├── notebooks/                    robustness_forgetting.ipynb
├── paper/codesign.md             long-form write-up
├── src/codesign/
│   ├── parser.py                 tree-sitter AST/DFG/CFG/sinks
│   ├── attacker.py               MAB + 3 mutators
│   ├── interpreter.py            DFG-guided activation patching
│   ├── metrics.py                evasion / preservation / head-importance
│   ├── benchmark.py              end-to-end harness
│   ├── visualizer.py             matplotlib helpers
│   ├── dataset.py                sample loader
│   └── cli.py                    `codesign scan|attack|benchmark|report`
└── tests/                        pytest suite
```

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
| `--target` | `mrm8488/codebert-base-finetuned-detect-insecure-code` | SVD classifier. |
| `--model` (scan) | `EleutherAI/pythia-160m` | Probe model — small, CPU-friendly. |

A full benchmark over the shipped 12-sample corpus runs in ~30s with
the heuristic fallback and ~5min with CodeBERT loaded.

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
- The default probe (`pythia-160m`) is not code-trained. It's a
  smoke-test substrate; reproduce on `deepseek-coder-1.3b` for the
  real WP2 claim.
- Sample corpus is small. Point `--dataset` at a Devign / BigVul /
  DiverseVul export with the same `.py` + `.json` convention to scale.

## Ethics

Defensive-research artifact. The mutators evade a classifier; they
don't author exploits. Disclosure aids defenders (the WP2 hooks help
locate vulnerable circuits) and the attack space (renaming,
dead-code, CFG flattening) is well-known in the literature.

## License

Apache 2.0.
