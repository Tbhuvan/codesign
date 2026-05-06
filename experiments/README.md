# Experiments

Each script writes JSON to `experiments/results/` and (where useful) appends
a one-line record to `experiments.json` at the project root.

| ID | Script | Question |
|---|---|---|
| exp01 | `exp01_attack_evaluation.py` | What evasion rate does the MAB get on real CodeBERT? |
| exp02 | `exp02_circuit_discovery.py` | Which heads carry the SVD signal under DFG-guided patching? |
| exp03 | `exp03_robustness_persistence.py` | Pre/post-attack scores; scaffolding for the forgetting study. |

## Run

```bash
python -m experiments.exp01_attack_evaluation
python -m experiments.exp02_circuit_discovery
python -m experiments.exp03_robustness_persistence
```

Default `--seed=0`. The MAB is deterministic given seed + inputs. First
run downloads the HuggingFace models (a few minutes); subsequent runs
hit the cache.
