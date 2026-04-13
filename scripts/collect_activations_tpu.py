#!/usr/bin/env python3
"""
Collect activations using the full model on TPU.

This script runs only forward passes (no generation), which works well on TPU.
The descriptions are generated separately using GGUF on CPU.

Usage:
    python scripts/collect_activations_tpu.py --text-file data/train/texts.json \
        --output-dir data/train/activations --batch-size 4
"""

import argparse
import json
import os
import sys
import time
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=str, required=True,
                       help="JSON file with list of texts, or JSONL with 'text' field")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--source-layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--max-act-tokens", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    return parser.parse_args()


def find_model_path():
    from pathlib import Path
    for base in [
        Path.home() / ".cache" / "huggingface" / "models--Qwen--Qwen3-8B",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-8B",
    ]:
        snapshots = base / "snapshots"
        if snapshots.exists():
            for v in snapshots.iterdir():
                return str(v)
    return "Qwen/Qwen3-8B"


def load_texts(path: str) -> list[str]:
    """Load texts from JSON or JSONL file."""
    if path.endswith(".jsonl"):
        texts = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])
        return texts
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            return [d["text"] for d in data]
        raise ValueError(f"Unexpected format in {path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set TPU env vars
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"

    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print(f"TPU device: {dev}")

    # Load model
    model_path = args.model_path or find_model_path()
    print(f"Loading model: {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(dev)
    model.eval()
    xm.mark_step()
    print("Model loaded on TPU")

    # Load texts
    texts = load_texts(args.text_file)
    end_idx = args.end_idx if args.end_idx > 0 else len(texts)
    texts = texts[args.start_idx:end_idx]
    print(f"Processing {len(texts)} texts (indices {args.start_idx}-{end_idx})")

    # Warmup forward pass
    print("Warming up...")
    warmup_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        model(input_ids=warmup_ids)
    xm.mark_step()
    print("Warmup done")

    # Collect activations
    t_start = time.time()
    meta_path = os.path.join(args.output_dir, "activation_meta.jsonl")
    meta_file = open(meta_path, "w")

    for i, text in enumerate(texts):
        global_idx = args.start_idx + i
        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=args.max_seq_length
            )
            input_ids = inputs["input_ids"].to(dev)
            seq_len = input_ids.shape[1]

            # Select token positions
            n_pos = min(args.max_act_tokens, seq_len)
            positions = sorted(random.sample(range(seq_len), n_pos))

            # Set up hooks
            activations = {}
            def make_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    # Extract at selected positions and move to CPU
                    activations[li] = h[:, positions, :].detach().cpu()
                return hook_fn

            hooks = []
            for li in args.source_layers:
                hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

            with torch.no_grad():
                model(input_ids=input_ids)
            xm.mark_step()

            for h in hooks:
                h.remove()

            # Save activations
            save_dict = {li: activations[li].squeeze(0) for li in args.source_layers}
            torch.save(save_dict, os.path.join(args.output_dir, f"activations_{global_idx}.pt"))

            # Save metadata
            meta = {
                "idx": global_idx,
                "seq_len": seq_len,
                "positions": positions,
                "source_layers": args.source_layers,
            }
            meta_file.write(json.dumps(meta) + "\n")

            if (i + 1) % 10 == 0:
                meta_file.flush()
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(texts) - i - 1) / max(rate, 0.01)
                print(f"  [{i+1}/{len(texts)}] Rate: {rate:.2f} ex/s | ETA: {eta:.0f}s")

        except Exception as e:
            print(f"  Error at index {global_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    meta_file.close()
    elapsed = time.time() - t_start
    print(f"\nDone! Processed {len(texts)} texts in {elapsed:.1f}s ({len(texts)/elapsed:.2f} ex/s)")


if __name__ == "__main__":
    main()
