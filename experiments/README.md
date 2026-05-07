# Experiments

Each script writes JSON to `experiments/results/` and (where useful) appends
a one-line record to `experiments.json` at the project root.

| ID | Script | Question |
|---|---|---|
| **exp00** | `exp00_target_calibration.py` | **Does the candidate target classifier label our probe set correctly?** Gating step. |
| exp01 | `exp01_attack_evaluation.py` | What evasion rate does the MAB get against the calibrated target? |
| exp02 | `exp02_circuit_discovery.py` | Which heads carry the SVD signal under DFG-guided patching on clean code? |
| exp03 | `exp03_diff_heatmap.py` | What changes inside the model when an attack succeeds? Per-head diff between clean and adversarial heatmaps. |
| exp04 | `exp04_in_context_defense.py` | Do k-shot ICL prompts harden an Ollama-served LLM against the MAB attack? |

## Run

```bash
# 0. calibrate the target (gating step - run this first!)
python -m experiments.exp00_target_calibration \
  --targets ollama:qwen3:8b ollama:qwen2.5:3b hf:mrm8488/codebert-base-finetuned-detect-insecure-code \
  --output experiments/results/exp00_calibration.json

# 1. attack benchmark (default target ollama:qwen3:8b, ~20 min)
python -m experiments.exp01_attack_evaluation

# 2. clean-code circuit sweep on 1 sample, half the layers (~10 min CPU)
python -m experiments.exp02_circuit_discovery \
  --max-layers 14 \
  --out-dir experiments/results/exp02_circuits

# 3. clean-vs-adversarial diff heatmap (~30 min/sample CPU)
python -m experiments.exp03_diff_heatmap \
  --samples cwe-78-os-command-injection cwe-89-sql-injection \
  --max-layers 14 \
  --out-dir experiments/results/exp03_diff

# 4. ICL defense (Ollama qwen3:8b, ~15 min for 4 samples × 2 shot configs)
python -m experiments.exp04_in_context_defense \
  --model qwen3:8b \
  --shots 0 3 \
  --n-samples 4 \
  --output experiments/results/exp04_icl.json
```

Default `--seed=0`. The MAB is deterministic given seed + inputs. First
run downloads the HuggingFace models (a few minutes); subsequent runs
hit the cache.

## Probe model note

exp02 and exp03 default to `Qwen/Qwen2.5-Coder-1.5B-Instruct`. The Coder
fine-tunes are not in TransformerLens's official registry, so the loader in
`interpreter.py` uses the architecture config of the registered base
(`Qwen/Qwen2.5-1.5B-Instruct`) and overrides the weights via the `hf_model=`
parameter. This is the standard TL pattern for using fine-tuned variants.

For laptops, swap to `Qwen/Qwen2.5-Coder-0.5B-Instruct` (~1 GB) which is
also handled by the override mapping.

## Ollama setup

exp04 requires an Ollama server reachable at `http://localhost:11434`.
Defaults to `qwen3:8b` (5.2 GB). The script issues HTTP `POST` calls
directly; no Ollama Python client dependency.

## What gets written

Every experiment produces:

- A `summary.json` with the run config, schema version, and aggregate
  metrics
- Per-sample records under `experiments/results/<exp>/<sample_id>/` with
  raw scores, traces, and any plots
- For exp02/exp03: PNG heatmaps using `matplotlib`
