#!/bin/bash
# Generate training data for the self-distillation activation oracle
# Runs on CPU (TPU generation is too slow due to XLA recompilation)
#
# Usage: ./scripts/generate_data.sh [num_samples] [output_dir]

set -e

NUM_SAMPLES=${1:-500}
OUTPUT_DIR=${2:-data}

cd /home/smitop2/ao-self-distill

echo "Starting data generation: ${NUM_SAMPLES} samples -> ${OUTPUT_DIR}"
echo "Start time: $(date)"

python3 run_pipeline.py generate \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --device cpu \
    2>&1 | tee "${OUTPUT_DIR}/generation.log"

echo "Data generation complete at $(date)"

# Count results
if [ -f "${OUTPUT_DIR}/dataset.jsonl" ]; then
    NUM_EXAMPLES=$(wc -l < "${OUTPUT_DIR}/dataset.jsonl")
    echo "Generated ${NUM_EXAMPLES} training examples"
fi

# Auto-commit the data
cd /home/smitop2/ao-self-distill
git add "${OUTPUT_DIR}/dataset.jsonl" "${OUTPUT_DIR}/generation.log" 2>/dev/null || true
git commit -m "Add generated training data (${NUM_EXAMPLES} examples)" 2>/dev/null || true
git push origin master 2>/dev/null || true
