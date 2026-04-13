#!/bin/bash
# Auto-start training when enough data has been generated.
# Polls data/dataset.jsonl and starts training when it has MIN_EXAMPLES entries.
#
# Usage: ./scripts/auto_train.sh [min_examples] [data_dir]

set -e

MIN_EXAMPLES=${1:-100}
DATA_DIR=${2:-data}
POLL_INTERVAL=60  # seconds

cd /home/smitop2/ao-self-distill

echo "Waiting for at least ${MIN_EXAMPLES} examples in ${DATA_DIR}/dataset.jsonl..."
echo "Polling every ${POLL_INTERVAL}s"

MAX_WAIT=28800  # 8 hours max
WAITED=0

while true; do
    if [ -f "${DATA_DIR}/dataset.jsonl" ]; then
        COUNT=$(wc -l < "${DATA_DIR}/dataset.jsonl")
        echo "$(date '+%H:%M:%S') - ${COUNT} examples generated"

        if [ "$COUNT" -ge "$MIN_EXAMPLES" ]; then
            echo ""
            echo "Sufficient data! Starting training..."
            break
        fi
    else
        echo "$(date '+%H:%M:%S') - No dataset file yet"
    fi

    WAITED=$((WAITED + POLL_INTERVAL))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out waiting for data after ${MAX_WAIT}s"
        # Start anyway with whatever we have
        if [ -f "${DATA_DIR}/dataset.jsonl" ]; then
            COUNT=$(wc -l < "${DATA_DIR}/dataset.jsonl")
            if [ "$COUNT" -gt 10 ]; then
                echo "Starting training with ${COUNT} examples anyway..."
                break
            fi
        fi
        exit 1
    fi

    sleep ${POLL_INTERVAL}
done

# Start training
echo ""
echo "============================================"
echo "Starting training at $(date)"
echo "============================================"

# Use TPU for training
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3
export PJRT_DEVICE=TPU

cd /home/smitop2/ao-self-distill/src
python3 train_tpu.py \
    --data_dir "../${DATA_DIR}" \
    --output_dir ../checkpoints \
    --device xla \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --grad_accum 4 \
    --save_every 200 \
    --log_every 10 \
    2>&1 | tee ../checkpoints/training.log

echo ""
echo "Training complete at $(date)"

# Commit results
cd /home/smitop2/ao-self-distill
git add checkpoints/training.log PLAN.md docs/ 2>/dev/null || true
git commit -m "Training complete - $(wc -l < ${DATA_DIR}/dataset.jsonl) examples" 2>/dev/null || true
git push origin master 2>/dev/null || true
