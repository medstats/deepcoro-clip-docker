#!/bin/bash
set -e

# ==============================================================================
# Vast.ai Launch & Inference Helper for DeepCORO-CLIP
# ==============================================================================

echo "================================================================================"
echo "  DeepCORO-CLIP Vast.ai Inference Runner"
echo "  Research model only — not validated for clinical decision-making."
echo "================================================================================"

DATA_DIR="${1:-/workspace/data}"
OUTPUT_DIR="${2:-/workspace/results}"
GPU_ID="${3:-0}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

echo "[INFO] Input Directory:  $DATA_DIR"
echo "[INFO] Output Directory: $OUTPUT_DIR"
echo "[INFO] GPU Device:       $GPU_ID"

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "[INFO] Starting DeepCORO-CLIP inference..."
python /workspace/run_deepcoro.py \
    --input "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --model stenosis \
    --gpu "$GPU_ID"

echo "[SUCCESS] Processing finished. Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
