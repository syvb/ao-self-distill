#!/usr/bin/env python3
"""
Orchestration script: monitors data generation and auto-starts training.

This runs as a long-lived process that:
1. Polls for data generation completion
2. Starts training on TPU when enough data is available
3. Runs evaluation after training
4. Commits results to git
"""

import json
import os
import subprocess
import sys
import time

DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
EVAL_DIR = "eval_results"
MIN_EXAMPLES = 30  # Start training with at least this many examples
POLL_INTERVAL = 30  # seconds
MAX_WAIT = 14400  # 4 hours

os.chdir("/home/smitop2/ao-self-distill")


def count_examples():
    """Count training examples in dataset.jsonl."""
    path = os.path.join(DATA_DIR, "dataset.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def is_generation_running():
    """Check if data generation process is running."""
    pid_file = os.path.join(DATA_DIR, ".generation_pid")
    if not os.path.exists(pid_file):
        return False
    with open(pid_file) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        return True
    except ProcessError:
        return False
    except Exception:
        return False


def wait_for_data():
    """Wait for sufficient training data."""
    print(f"Waiting for at least {MIN_EXAMPLES} training examples...")
    waited = 0

    while True:
        n = count_examples()
        running = is_generation_running()
        status = "RUNNING" if running else "STOPPED"
        print(f"  [{waited//60}min] {n} examples, generation {status}")

        if n >= MIN_EXAMPLES:
            print(f"\nSufficient data! {n} examples available.")
            return True

        if not running and n > 0:
            print(f"\nGeneration stopped with {n} examples.")
            if n >= 10:
                print("Starting with what we have...")
                return True
            else:
                print("Not enough examples. Exiting.")
                return False

        if waited >= MAX_WAIT:
            if n >= 10:
                print(f"\nTimeout! Starting with {n} examples.")
                return True
            return False

        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL


def start_training():
    """Start training on TPU (or CPU if TPU unavailable)."""
    n = count_examples()
    print(f"\n{'='*60}")
    print(f"Starting training with {n} examples")
    print(f"{'='*60}")

    # Try TPU first, fall back to CPU
    env = os.environ.copy()
    env["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    env["TPU_HOST_BOUNDS"] = "1,1,1"
    env["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
    env["PJRT_DEVICE"] = "TPU"

    cmd = [
        sys.executable, "src/train_tpu.py",
        "--data_dir", DATA_DIR,
        "--output_dir", CHECKPOINT_DIR,
        "--device", "cpu",  # Start with CPU for reliability
        "--epochs", "3",
        "--batch_size", "2",
        "--learning_rate", "2e-4",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--grad_accum", "2",
        "--save_every", "100",
        "--log_every", "5",
    ]

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, cwd="/home/smitop2/ao-self-distill")

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return False

    print("Training complete!")
    return True


def run_evaluation():
    """Run evaluation on the trained model."""
    print(f"\n{'='*60}")
    print("Running evaluation")
    print(f"{'='*60}")

    final_checkpoint = os.path.join(CHECKPOINT_DIR, "final")
    if not os.path.exists(final_checkpoint):
        # Find latest checkpoint
        checkpoints = sorted(
            [d for d in os.listdir(CHECKPOINT_DIR)
             if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))],
        )
        if checkpoints:
            final_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        else:
            print("No checkpoint found!")
            return

    cmd = [
        sys.executable, "src/evaluate.py",
        "--checkpoint", final_checkpoint,
        "--data_dir", DATA_DIR,
        "--output_dir", EVAL_DIR,
        "--device", "cpu",
        "--num_eval", "20",
    ]

    result = subprocess.run(cmd, cwd="/home/smitop2/ao-self-distill")
    return result.returncode == 0


def git_commit(message):
    """Commit and push results."""
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "master"], check=True)
        print(f"Committed: {message}")
    except Exception as e:
        print(f"Git commit failed: {e}")


def main():
    print("="*60)
    print("Self-Distillation AO Pipeline Orchestrator")
    print("="*60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Min examples: {MIN_EXAMPLES}")
    print()

    # Step 1: Wait for data
    if not wait_for_data():
        print("Insufficient data. Exiting.")
        return

    # Step 2: Train
    success = start_training()
    if success:
        git_commit(f"Training complete - {count_examples()} examples")

    # Step 3: Evaluate
    if success:
        eval_success = run_evaluation()
        if eval_success:
            git_commit("Add evaluation results")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
