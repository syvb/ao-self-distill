#!/bin/bash
# Run training after data generation is complete.
# Usage: ./scripts/run_training.sh [data_dir] [output_dir]
set -e

DATA_DIR=${1:-data}
OUTPUT_DIR=${2:-checkpoints}

cd /home/smitop2/ao-self-distill

# Check data
if [ ! -f "${DATA_DIR}/dataset.jsonl" ]; then
    echo "ERROR: No dataset found at ${DATA_DIR}/dataset.jsonl"
    exit 1
fi

NUM_EXAMPLES=$(wc -l < "${DATA_DIR}/dataset.jsonl")
echo "Training with ${NUM_EXAMPLES} examples from ${DATA_DIR}"

mkdir -p ${OUTPUT_DIR}

# Run training on CPU (TPU for training requires careful setup)
cd src
python3 train_tpu.py \
    --data_dir "../${DATA_DIR}" \
    --output_dir "../${OUTPUT_DIR}" \
    --device cpu \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --grad_accum 2 \
    --save_every 50 \
    --log_every 5 \
    2>&1 | tee "../${OUTPUT_DIR}/training.log"

echo "Training complete!"
echo "Checkpoints in ${OUTPUT_DIR}/"

# Commit
cd /home/smitop2/ao-self-distill
git add "${OUTPUT_DIR}/training.log" docs/ 2>/dev/null || true
git commit -m "Training complete with ${NUM_EXAMPLES} examples" 2>/dev/null || true
git push origin master 2>/dev/null || true
