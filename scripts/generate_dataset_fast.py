#!/usr/bin/env python3
"""
Fast dataset generation pipeline using GGUF (CPU) for descriptions
and torch (CPU/TPU) for activation collection.

This can be distributed across multiple workers in the TPU pod.
Each worker processes a shard of the data.

Usage:
    # Single worker (all data):
    python scripts/generate_dataset_fast.py --num-samples 5000 --output-dir data/train

    # Distributed across workers (run on each worker):
    python scripts/generate_dataset_fast.py --num-samples 5000 --output-dir data/train \
        --worker-id 0 --num-workers 16
"""

import argparse
import json
import os
import sys
import time
import random
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DESCRIPTION_PROMPTS = [
    (
        "Describe the semantic content of the following text in detail. "
        "Cover: the language used, the topic, grammatical structures, "
        "sentiment/tone, and likely continuations.\n\nText: \"{text}\" /no_think"
    ),
    (
        "Analyze this text passage. What language is it in? What is the subject matter? "
        "What is the writing style and register? What might come next?\n\n"
        "Text: \"{text}\" /no_think"
    ),
    (
        "You are examining a text passage. Describe what you observe about it: "
        "its language, topic, structure, tone, and what information it conveys. "
        "Also predict what might follow.\n\nText: \"{text}\" /no_think"
    ),
    (
        "Provide a comprehensive semantic analysis of this text. Include: "
        "(1) language identification, (2) topic/domain, (3) key entities or concepts, "
        "(4) grammatical features, (5) pragmatic intent, (6) likely continuations.\n\n"
        "Text: \"{text}\" /no_think"
    ),
    (
        "What can you tell about this text? Describe its content, language, style, "
        "and meaning as thoroughly as you can.\n\nText: \"{text}\" /no_think"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="data/train")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-desc-tokens", type=int, default=200)
    parser.add_argument("--n-threads", type=int, default=32)
    parser.add_argument("--source-layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--max-act-tokens", type=int, default=10,
                       help="Max tokens to collect activations from per example")
    parser.add_argument("--gguf-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--skip-activations", action="store_true",
                       help="Only generate descriptions (for workers without the full model)")
    parser.add_argument("--skip-descriptions", action="store_true",
                       help="Only collect activations (for workers without GGUF)")
    return parser.parse_args()


def find_gguf_path():
    cache = Path.home() / ".cache" / "huggingface" / "models--Qwen--Qwen3-8B-GGUF"
    for p in cache.rglob("*.gguf"):
        if "Q4_K_M" in p.name:
            return str(p)
    return None


def find_model_path():
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
    return "Qwen/Qwen3-8B"


def load_text_samples(num_samples: int, seed: int = 42) -> list[str]:
    """Load text samples from wikitext (always available, fast)."""
    from datasets import load_dataset
    random.seed(seed)
    samples = []

    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:
                words = text.split()
                if len(words) > 20:
                    start = random.randint(0, max(0, len(words) - 50))
                    chunk = " ".join(words[start:start + 50])
                    if len(chunk) > 80:
                        samples.append(chunk)
                        if len(samples) >= num_samples * 2:  # Over-sample then trim
                            break
    except Exception as e:
        print(f"Error loading wikitext: {e}")

    random.shuffle(samples)
    return samples[:num_samples]


def generate_description_gguf(llm, text: str, prompt_idx: int, max_tokens: int = 200) -> str:
    """Generate a description using GGUF model via llama.cpp."""
    template = DESCRIPTION_PROMPTS[prompt_idx % len(DESCRIPTION_PROMPTS)]
    user_msg = template.format(text=text[:400])

    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.9, echo=False,
                 stop=["<|im_end|>", "<|im_start|>"])
    text_out = output["choices"][0]["text"]

    # Strip <think> tags if present
    if "<think>" in text_out and "</think>" in text_out:
        think_end = text_out.index("</think>") + len("</think>")
        text_out = text_out[think_end:].strip()
    elif text_out.startswith("<think>"):
        # Incomplete think block
        text_out = text_out.split("</think>")[-1].strip() if "</think>" in text_out else text_out

    return text_out.strip()


def collect_activations_for_text(
    model, tokenizer, text: str, source_layers: list[int],
    max_tokens: int = 10, max_length: int = 256,
) -> tuple[dict, list[int], int]:
    """Collect activations from the full model on CPU."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Select token positions
    n_pos = min(max_tokens, seq_len)
    positions = sorted(random.sample(range(seq_len), n_pos))

    activations = {}
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            activations[li] = h[:, positions, :].detach().clone()
        return hook_fn

    hooks = []
    for li in source_layers:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return activations, positions, seq_len


def main():
    args = parse_args()
    random.seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    act_dir = os.path.join(args.output_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)

    # Load text samples
    print(f"Loading {args.num_samples} text samples...")
    all_texts = load_text_samples(args.num_samples, seed=args.seed)
    print(f"  Got {len(all_texts)} samples")

    # Shard for this worker
    shard_size = len(all_texts) // args.num_workers
    start_idx = args.worker_id * shard_size
    end_idx = start_idx + shard_size if args.worker_id < args.num_workers - 1 else len(all_texts)
    texts = all_texts[start_idx:end_idx]
    print(f"Worker {args.worker_id}/{args.num_workers}: processing indices {start_idx}-{end_idx} ({len(texts)} samples)")

    # Load GGUF model for descriptions
    llm = None
    if not args.skip_descriptions:
        gguf_path = args.gguf_path or find_gguf_path()
        if gguf_path:
            from llama_cpp import Llama
            print(f"Loading GGUF model: {gguf_path}")
            llm = Llama(gguf_path, n_ctx=2048, n_threads=args.n_threads, verbose=False)
        else:
            print("WARNING: No GGUF model found. Skipping descriptions.")
            args.skip_descriptions = True

    # Load full model for activation collection
    full_model = None
    tokenizer = None
    if not args.skip_activations:
        model_path = args.model_path or find_model_path()
        print(f"Loading full model for activations: {model_path}")
        os.environ["PJRT_DEVICE"] = "CPU"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        full_model.eval()
        print("  Full model loaded")

    # Process samples
    desc_path = os.path.join(args.output_dir, f"descriptions_worker{args.worker_id}.jsonl")
    desc_file = open(desc_path, "w")
    t_start = time.time()
    n_processed = 0

    for local_idx, text in enumerate(texts):
        global_idx = start_idx + local_idx

        try:
            # Generate description
            description = None
            if not args.skip_descriptions and llm is not None:
                prompt_idx = global_idx % len(DESCRIPTION_PROMPTS)
                description = generate_description_gguf(
                    llm, text, prompt_idx, max_tokens=args.max_desc_tokens
                )

            # Collect activations
            act_dict = None
            positions = None
            seq_len = 0
            if not args.skip_activations and full_model is not None:
                act_dict, positions, seq_len = collect_activations_for_text(
                    full_model, tokenizer, text, args.source_layers,
                    max_tokens=args.max_act_tokens,
                )
                # Save activations
                save_dict = {li: v.cpu() for li, v in act_dict.items()}
                torch.save(save_dict, os.path.join(act_dir, f"activations_{global_idx}.pt"))

            # Save description
            desc_item = {
                "idx": global_idx,
                "text": text,
                "description": description,
                "prompt_idx": global_idx % len(DESCRIPTION_PROMPTS),
                "seq_len": seq_len,
                "positions": positions,
            }
            desc_file.write(json.dumps(desc_item, ensure_ascii=False) + "\n")
            n_processed += 1

            if n_processed % 10 == 0:
                desc_file.flush()
                elapsed = time.time() - t_start
                rate = n_processed / elapsed
                eta = (len(texts) - n_processed) / max(rate, 0.01)
                print(f"  Worker {args.worker_id}: [{n_processed}/{len(texts)}] "
                      f"Rate: {rate:.2f} ex/s | ETA: {eta/60:.1f} min")

        except Exception as e:
            print(f"  Error at index {global_idx}: {e}")
            continue

    desc_file.close()
    elapsed = time.time() - t_start
    print(f"\nWorker {args.worker_id} done! Processed {n_processed} samples in {elapsed/60:.1f} min")
    print(f"  Descriptions: {desc_path}")
    if not args.skip_activations:
        print(f"  Activations: {act_dir}")


if __name__ == "__main__":
    main()
