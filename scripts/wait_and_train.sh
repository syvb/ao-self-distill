#!/bin/bash
# Wait for data generation to finish, then start training automatically.
set -e
cd /home/smitop2/ao-self-distill

GEN_PID=$(cat data/.generation_pid 2>/dev/null)

echo "Waiting for generation process $GEN_PID to finish..."
if [ -n "$GEN_PID" ] && kill -0 "$GEN_PID" 2>/dev/null; then
    # Use wait if possible, otherwise poll infrequently
    tail --pid="$GEN_PID" -f /dev/null 2>/dev/null || {
        while kill -0 "$GEN_PID" 2>/dev/null; do
            sleep 30
            echo "$(date '+%H:%M') - still waiting, $(wc -l < data/dataset.jsonl) examples"
        done
    }
fi

echo ""
echo "Generation complete at $(date)"
echo "Final dataset: $(wc -l < data/dataset.jsonl) examples"

# Commit data
git add data/dataset.jsonl data/descriptions.jsonl data/simple_gen.log 2>/dev/null
git commit -m "Complete: $(wc -l < data/dataset.jsonl) training examples from 50 texts" 2>/dev/null || true
git push origin master 2>/dev/null || true

echo ""
echo "Starting training..."
mkdir -p checkpoints

PYTHONUNBUFFERED=1 python3 src/train_tpu.py \
    --data_dir data \
    --output_dir checkpoints \
    --device cpu \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --grad_accum 2 \
    --save_every 50 \
    --log_every 5 \
    2>&1 | tee checkpoints/training.log

echo "Training complete at $(date)"

# Commit checkpoints
git add checkpoints/training.log 2>/dev/null
git commit -m "Training complete" 2>/dev/null || true
git push origin master 2>/dev/null || true
