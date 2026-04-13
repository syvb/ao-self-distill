#!/usr/bin/env python3
"""
Train the Self-Supervised Activation Oracle.

This script fine-tunes Qwen3-8B with LoRA to produce semantic descriptions
from injected activation vectors.

Usage:
    python scripts/train_ao.py --data-dir data/train --output-dir checkpoints/ao_v1
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model_and_tokenizer, DEFAULT_SOURCE_LAYERS
from src.data import AODataset
from src.train import setup_lora, train_ao


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/ao_v1")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--source-layers", type=int, nargs="+", default=DEFAULT_SOURCE_LAYERS)
    parser.add_argument("--val-split", type=float, default=0.1)
    return parser.parse_args()


def find_model_path():
    from pathlib import Path
    from src.model import MODEL_NAME
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "models--Qwen--Qwen3-8B",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-8B",
    ]
    for cache_dir in cache_dirs:
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            versions = list(snapshots.iterdir())
            if versions:
                return str(versions[0])
    return MODEL_NAME


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = args.model_path or find_model_path()
    print(f"Model: {model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    # Determine device
    device = args.device
    if device == "auto":
        try:
            import torch_xla.core.xla_model as xm
            os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
            os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
            device = xm.xla_device()
        except Exception:
            device = "cpu"
    elif device == "xla":
        import torch_xla.core.xla_model as xm
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
        os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
        device = xm.xla_device()

    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device="cpu")

    # Apply LoRA
    print("Setting up LoRA...")
    model = setup_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)

    # Load dataset
    print("Loading dataset...")
    act_dir = os.path.join(args.data_dir, "activations")
    desc_path = os.path.join(args.data_dir, "descriptions.jsonl")

    full_dataset = AODataset(
        activation_dir=act_dir,
        descriptions_path=desc_path,
        tokenizer=tokenizer,
        source_layers=args.source_layers,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # Train
    print("\nStarting training...")
    model = train_ao(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
        device=device,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
