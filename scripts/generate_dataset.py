#!/usr/bin/env python3
"""
Generate the self-supervised AO training dataset.

This script:
1. Loads diverse text samples
2. Runs Qwen3-8B on each text to collect residual stream activations
3. Generates semantic descriptions of each text using the same model
4. Saves (activation, description) pairs for AO training

Usage:
    python scripts/generate_dataset.py --num-samples 5000 --output-dir data/train
"""

import argparse
import json
import os
import sys
import time
import random
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    load_model_and_tokenizer,
    ActivationCollector,
    DEFAULT_SOURCE_LAYERS,
    MODEL_NAME,
)
from src.data import (
    load_simple_text_samples,
    load_text_samples,
    generate_description,
    DESCRIPTION_PROMPTS,
    _save_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="data/train")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model (default: auto-detect from cache)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cpu, xla, auto")
    parser.add_argument("--source-layers", type=int, nargs="+", default=DEFAULT_SOURCE_LAYERS)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-desc-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simple-data", action="store_true",
                       help="Use simple/small datasets (wikitext) for quick testing")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing data")
    parser.add_argument("--save-every", type=int, default=50)
    # For collecting activations at specific token positions
    parser.add_argument("--token-strategy", type=str, default="random_subset",
                       choices=["all", "last", "random_subset", "middle"],
                       help="Which token positions to collect activations from")
    parser.add_argument("--max-num-tokens", type=int, default=10,
                       help="Max number of tokens to collect activations from (for random_subset)")
    return parser.parse_args()


def find_model_path():
    """Auto-detect the model path from HuggingFace cache."""
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


def select_token_positions(seq_len: int, strategy: str, max_tokens: int) -> list[int]:
    """Select which token positions to collect activations from."""
    if strategy == "all":
        return list(range(seq_len))
    elif strategy == "last":
        return [seq_len - 1]
    elif strategy == "middle":
        mid = seq_len // 2
        start = max(0, mid - max_tokens // 2)
        end = min(seq_len, start + max_tokens)
        return list(range(start, end))
    elif strategy == "random_subset":
        n = min(max_tokens, seq_len)
        return sorted(random.sample(range(seq_len), n))
    else:
        return list(range(seq_len))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    act_dir = os.path.join(args.output_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)
    desc_path = os.path.join(args.output_dir, "descriptions.jsonl")
    meta_path = os.path.join(args.output_dir, "metadata.json")

    # Load existing data if resuming
    existing_count = 0
    if args.resume and os.path.exists(desc_path):
        with open(desc_path) as f:
            existing_count = sum(1 for _ in f)
        print(f"Resuming from {existing_count} existing examples")

    # Find model path
    model_path = args.model_path or find_model_path()
    print(f"Using model: {model_path}")

    # Determine device
    device = args.device
    if device == "auto":
        try:
            import torch_xla.core.xla_model as xm
            device = "xla"
        except Exception:
            device = "cpu"

    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    t0 = time.time()

    if device == "xla":
        import torch_xla.core.xla_model as xm
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
        os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
        model, tokenizer = load_model_and_tokenizer(model_path, device="xla")
    else:
        model, tokenizer = load_model_and_tokenizer(model_path, device=device)

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load text samples
    print("Loading text samples...")
    if args.simple_data:
        texts = load_simple_text_samples(args.num_samples, seed=args.seed)
    else:
        texts = load_text_samples(args.num_samples, seed=args.seed)

    if len(texts) < args.num_samples:
        print(f"Warning: Only got {len(texts)} samples (requested {args.num_samples})")

    # Save metadata
    metadata = {
        "model": model_path,
        "num_samples": len(texts),
        "source_layers": args.source_layers,
        "token_strategy": args.token_strategy,
        "max_num_tokens": args.max_num_tokens,
        "max_text_length": args.max_text_length,
        "seed": args.seed,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Process each text
    collector = ActivationCollector(model, args.source_layers)
    desc_file = open(desc_path, "a" if args.resume else "w")

    print(f"\nProcessing {len(texts)} texts...")
    t_start = time.time()

    for i, text in enumerate(texts):
        if i < existing_count:
            continue

        try:
            # Step 1: Collect activations
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=args.max_text_length
            )
            input_ids = inputs["input_ids"].to(model.device)
            seq_len = input_ids.shape[1]

            positions = select_token_positions(
                seq_len, args.token_strategy, args.max_num_tokens
            )

            with collector.collect():
                with torch.no_grad():
                    model(input_ids=input_ids)

                # Extract activations at selected positions
                act_dict = {}
                for layer_idx in args.source_layers:
                    acts = collector.get_activations(layer_idx, positions)
                    act_dict[layer_idx] = acts.cpu()

            # Save activations
            act_path = os.path.join(act_dir, f"activations_{i}.pt")
            torch.save(act_dict, act_path)

            # Step 2: Generate semantic description
            prompt_idx = i % len(DESCRIPTION_PROMPTS)
            description = generate_description(
                model, tokenizer, text,
                max_new_tokens=args.max_desc_tokens,
                prompt_idx=prompt_idx,
            )

            # Save description
            desc_item = {
                "idx": i,
                "text": text,
                "description": description,
                "prompt_idx": prompt_idx,
                "seq_len": seq_len,
                "num_positions": len(positions),
                "positions": positions,
            }
            desc_file.write(json.dumps(desc_item, ensure_ascii=False) + "\n")

            if (i + 1) % args.save_every == 0:
                desc_file.flush()
                elapsed = time.time() - t_start
                rate = (i + 1 - existing_count) / elapsed
                eta = (len(texts) - i - 1) / max(rate, 0.01)
                print(f"  [{i+1}/{len(texts)}] Rate: {rate:.1f} ex/s | ETA: {eta/60:.1f} min")
                print(f"    Last description: {description[:100]}...")

        except Exception as e:
            print(f"  Error processing text {i}: {e}")
            continue

    desc_file.close()
    elapsed = time.time() - t_start
    print(f"\nDone! Processed {len(texts)} texts in {elapsed/60:.1f} minutes")
    print(f"Activations saved to: {act_dir}")
    print(f"Descriptions saved to: {desc_path}")


if __name__ == "__main__":
    main()
