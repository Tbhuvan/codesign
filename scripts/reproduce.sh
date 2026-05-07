#!/usr/bin/env bash
# Reproduce the experimental results in the paper.
# Run from the repository root.
#
# Prerequisites:
#   - Python 3.10+
#   - Ollama running locally with `qwen3:8b` pulled
#     (`ollama pull qwen3:8b`)
#   - `pip install -e ".[dev,notebook]"`
#
# Wall-clock guidance (CPU laptop):
#   exp00: ~3 min
#   exp01 single seed: ~20 min
#   exp01 three seeds:  ~60 min
#   exp02 (max_layers=8 on one sample): ~15 min
#   exp03 (max_layers=8 on one sample): ~30 min
#   exp04 (4 samples, two shot configs): ~15 min

set -euo pipefail

mkdir -p experiments/results figures

echo "=== exp00: target calibration ==="
python -m experiments.exp00_target_calibration \
    --targets ollama:qwen3:8b ollama:qwen2.5:3b \
              hf:mrm8488/codebert-base-finetuned-detect-insecure-code \
    --output experiments/results/exp00_calibration.json

echo
echo "=== exp01: MAB attack benchmark, three seeds ==="
python -m experiments.exp01_attack_evaluation \
    --target ollama:qwen3:8b \
    --seeds 0 1 2 \
    --max-steps 10 \
    --output-dir experiments/results/exp01_multiseed

echo
echo "=== exp02: clean-code circuit sweep on cwe-78 ==="
python -m experiments.exp02_circuit_discovery \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --max-layers 8 \
    --out-dir experiments/results/exp02_circuits

echo
echo "=== exp03: clean-vs-adversarial diff heatmap on cwe-78 ==="
python -m experiments.exp03_diff_heatmap \
    --probe-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --svd-target ollama:qwen3:8b \
    --samples cwe-78-os-command-injection \
    --max-layers 8 \
    --max-steps 8 \
    --out-dir experiments/results/exp03_diff

echo
echo "=== exp04: ICL defense, 0-shot vs 3-shot ==="
python -m experiments.exp04_in_context_defense \
    --model qwen3:8b \
    --shots 0 3 \
    --n-samples 4 \
    --max-steps 8 \
    --output experiments/results/exp04_icl.json

echo
echo "All experiments complete. JSON results under experiments/results/."
echo "Figures (PNG) under experiments/results/exp02_circuits/ and exp03_diff/<sample>/."
