#!/bin/bash
# Hybrid pipeline: CPU for data generation, TPU for training.
# Qwen3-1.7B: fast on CPU for generation, fits on TPU for training.
set -e
cd /home/smitop2/ao-self-distill

echo "=== Step 1: Generate data on CPU ==="
echo "Start: $(date)"

PYTHONUNBUFFERED=1 python3 scripts/full_pipeline_1_7b.py --mode generate 2>&1

echo ""
echo "=== Step 2: Train on TPU ==="
echo "Start: $(date)"

export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export PJRT_DEVICE=TPU

PYTHONUNBUFFERED=1 python3 scripts/full_pipeline_1_7b.py --mode train 2>&1

echo ""
echo "=== Pipeline complete at $(date) ==="
