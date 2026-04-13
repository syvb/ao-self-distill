#!/usr/bin/env python3
"""
Fast data generation pipeline.

Strategy:
1. Load texts and generate ALL descriptions first (slow, CPU)
2. Then collect ALL activations (fast, forward pass only)
3. Combine into training dataset

Optimizations:
- Shorter descriptions (128 tokens)
- Only layer 18 by default (primary layer)
- Greedy decoding (faster than sampling)
- Efficient batching of activation collection
"""

import json
import os
import sys
import time
import random
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_model_and_tokenizer, ActivationCollector, DEFAULT_SOURCE_LAYERS
from data import load_text_samples, load_simple_text_samples, DESCRIPTION_PROMPTS


def generate_descriptions_fast(
    model, tokenizer, texts: list, max_new_tokens: int = 128,
    save_path: str = None,
):
    """Generate descriptions for all texts. Resumes from save file if exists."""
    results = []

    # Resume support
    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            for line in f:
                results.append(json.loads(line))
        print(f"Resuming from {len(results)} existing descriptions")

    existing = {r["text"][:100] for r in results}
    save_file = open(save_path, "a") if save_path else None

    print(f"Generating descriptions for {len(texts)} texts...")
    start = time.time()

    for i, text in enumerate(texts):
        if text[:100] in existing:
            continue

        if i > 0 and i % 5 == 0:
            elapsed = time.time() - start
            rate = (i - len(existing)) / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i}/{len(texts)}] {rate:.2f}/sec, "
                  f"ETA: {eta:.0f}min, done: {len(results)}")

        try:
            # Use a simple prompt
            template = random.choice(DESCRIPTION_PROMPTS)
            prompt = template.format(text=text[:400])

            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            inputs = tokenizer(formatted, return_tensors="pt",
                             truncation=True, max_length=768)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy is faster
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            description = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Strip think tags
            if "<think>" in description:
                import re
                description = re.sub(r'<think>.*?</think>', '', description, flags=re.DOTALL).strip()
                if "<think>" in description:
                    description = description.split("</think>")[-1].strip()

            if len(description) < 20:
                continue

            result = {
                "text": text[:500],
                "description": description,
                "text_idx": i,
            }
            results.append(result)

            if save_file:
                save_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                save_file.flush()

        except Exception as e:
            print(f"  Error on text {i}: {e}")
            continue

    if save_file:
        save_file.close()

    elapsed = time.time() - start
    print(f"Generated {len(results)} descriptions in {elapsed/60:.1f}min")
    return results


def collect_activations_fast(
    model, tokenizer, texts: list, descriptions: list,
    source_layers: list = None, output_dir: str = "data",
    max_positions: int = 3,
):
    """Collect activations for all texts. This is a fast forward-pass only step."""
    source_layers = source_layers or [18]  # Only primary layer by default

    act_dir = os.path.join(output_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)

    collector = ActivationCollector(model, source_layers)
    collector.register_hooks()

    dataset = []
    total_examples = 0

    print(f"Collecting activations for {len(descriptions)} texts, "
          f"layers {source_layers}...")
    start = time.time()

    for desc_item in descriptions:
        text = desc_item["text"]
        description = desc_item["description"]
        text_idx = desc_item["text_idx"]

        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                continue

            # Select token positions (prefer content words)
            candidates = []
            for pos in range(1, seq_len - 1):
                tok_str = tokenizer.decode([input_ids[0, pos].item()])
                if len(tok_str.strip()) > 2:
                    candidates.append(pos)
            if not candidates:
                candidates = list(range(1, seq_len - 1))

            num_pos = min(max_positions, len(candidates))
            positions = sorted(random.sample(candidates, num_pos))

            # Forward pass (fast!)
            collector.activations = {}
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            # Save activations per layer
            for layer_idx in source_layers:
                if layer_idx not in collector.activations:
                    continue

                acts = collector.activations[layer_idx]  # (1, seq_len, hidden)
                position_acts = acts[0, positions, :].cpu()  # (num_pos, hidden)

                act_file = f"act_{total_examples}_L{layer_idx}.pt"
                torch.save(position_acts, os.path.join(act_dir, act_file))

                example = {
                    "idx": total_examples,
                    "text": text[:500],
                    "description": description,
                    "layer": layer_idx,
                    "positions": positions,
                    "num_activations": num_pos,
                    "activation_file": act_file,
                    "seq_len": seq_len,
                    "text_idx": text_idx,
                }
                dataset.append(example)
                total_examples += 1

        except Exception as e:
            print(f"  Error on text_idx {text_idx}: {e}")
            continue

        if len(dataset) % 100 == 0 and len(dataset) > 0:
            elapsed = time.time() - start
            rate = len(dataset) / elapsed
            print(f"  {len(dataset)} examples, {rate:.1f}/sec")

    collector.clear()
    elapsed = time.time() - start
    print(f"Collected {len(dataset)} examples in {elapsed:.1f}sec")

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {dataset_path}")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_desc_tokens", type=int, default=128)
    parser.add_argument("--layers", type=str, default="18",
                       help="Comma-separated layer indices")
    parser.add_argument("--max_positions", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    source_layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading Qwen3-8B on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(device=args.device)
    print(f"Model loaded on {model.device}")

    # Load texts
    print(f"Loading {args.num_texts} texts...")
    try:
        texts = load_text_samples(args.num_texts)
    except Exception:
        texts = load_simple_text_samples(args.num_texts)
    print(f"Got {len(texts)} texts")

    # Step 1: Generate descriptions (slow)
    desc_path = os.path.join(args.output_dir, "descriptions.jsonl")
    descriptions = generate_descriptions_fast(
        model, tokenizer, texts,
        max_new_tokens=args.max_desc_tokens,
        save_path=desc_path,
    )

    # Step 2: Collect activations (fast)
    dataset = collect_activations_fast(
        model, tokenizer, texts, descriptions,
        source_layers=source_layers,
        output_dir=args.output_dir,
        max_positions=args.max_positions,
    )

    print(f"\nDone! {len(dataset)} training examples in {args.output_dir}/")

    # Auto-commit
    try:
        os.system(f"cd /home/smitop2/ao-self-distill && "
                  f"git add {args.output_dir}/dataset.jsonl "
                  f"{args.output_dir}/descriptions.jsonl && "
                  f"git commit -m 'Add generated training data ({len(dataset)} examples)' && "
                  f"git push origin master")
    except Exception:
        pass


if __name__ == "__main__":
    main()
