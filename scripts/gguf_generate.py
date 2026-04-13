#!/usr/bin/env python3
"""
Fast data generation using GGUF for descriptions + full model for activations.

Strategy:
1. Use GGUF quantized Qwen3-8B for fast description generation (10x faster)
2. Use full bf16 model for activation collection (forward pass only, no generation)
3. TPU for activation collection, GGUF on CPU for descriptions

This is still self-distillation because:
- The same architecture generates descriptions (quantized ≈ original)
- The full model produces the activations used for training
"""

import json
import os
import sys
import time
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

GGUF_PATH = "/home/smitop2/.cache/huggingface/hub/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf"

DESCRIPTION_PROMPTS = [
    (
        "Describe the semantic content of the following text in detail. "
        "Cover: the language used, the topic, grammatical structures, "
        "sentiment/tone, and likely continuations.\n\nText: \"{text}\"\n\n"
        "Analysis:"
    ),
    (
        "Analyze this text passage. What language is it in? What is the subject matter? "
        "What is the writing style and register? What might come next?\n\nText: \"{text}\"\n\n"
        "Analysis:"
    ),
    (
        "Provide a comprehensive semantic analysis of this text. Include: "
        "(1) language identification, (2) topic/domain, (3) key entities or concepts, "
        "(4) grammatical features, (5) pragmatic intent, (6) likely continuations.\n\n"
        "Text: \"{text}\"\n\nAnalysis:"
    ),
    (
        "What can you tell about this text? Describe its content, language, style, "
        "and meaning as thoroughly as you can.\n\nText: \"{text}\"\n\nAnalysis:"
    ),
    (
        "You are examining a text passage. Describe what you observe about it: "
        "its language, topic, structure, tone, and what information it conveys. "
        "Also predict what might follow.\n\nText: \"{text}\"\n\nObservations:"
    ),
]


def load_texts(num_texts: int) -> list:
    """Load text samples from wikitext (always available)."""
    from datasets import load_dataset

    random.seed(42)
    samples = []

    # wikitext-103 (large, diverse)
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:
                words = text.split()
                if len(words) > 20:
                    start = random.randint(0, max(0, len(words) - 60))
                    chunk = " ".join(words[start:start + 60])
                    if len(chunk) > 80:
                        samples.append(chunk)
                        if len(samples) >= num_texts:
                            break
    except Exception as e:
        print(f"wikitext failed: {e}")

    # If not enough, add FineWeb
    if len(samples) < num_texts:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                            split="train", streaming=True)
            for item in ds:
                text = item["text"].strip()
                if len(text) > 100:
                    words = text.split()
                    if len(words) > 30:
                        start = random.randint(0, max(0, len(words) - 60))
                        chunk = " ".join(words[start:start + 60])
                        samples.append(chunk)
                        if len(samples) >= num_texts:
                            break
        except Exception as e:
            print(f"FineWeb failed: {e}")

    random.shuffle(samples)
    return samples[:num_texts]


def generate_descriptions_gguf(texts: list, max_tokens: int = 128,
                                save_path: str = None,
                                n_threads: int = 64) -> list:
    """Generate descriptions using GGUF model (much faster than full model)."""
    from llama_cpp import Llama

    # Resume support
    results = []
    existing = set()
    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                existing.add(r["text"][:80])
        print(f"Resuming from {len(results)} existing descriptions")

    print(f"Loading GGUF model ({n_threads} threads)...")
    llm = Llama(GGUF_PATH, n_ctx=2048, n_threads=n_threads, verbose=False)
    print("GGUF model loaded")

    save_file = open(save_path, "a") if save_path else None

    print(f"Generating {len(texts)} descriptions...")
    start = time.time()
    skipped = 0

    for i, text in enumerate(texts):
        if text[:80] in existing:
            skipped += 1
            continue

        try:
            template = DESCRIPTION_PROMPTS[i % len(DESCRIPTION_PROMPTS)]
            prompt = template.format(text=text[:400])

            output = llm(prompt, max_tokens=max_tokens, temperature=0.7,
                        top_p=0.9, stop=["\n\n\n"])
            description = output["choices"][0]["text"].strip()

            # Strip think tags if present
            if "<think>" in description:
                import re
                description = re.sub(r'<think>.*?</think>', '', description,
                                    flags=re.DOTALL).strip()

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
                if len(results) % 10 == 0:
                    save_file.flush()

        except Exception as e:
            print(f"  Error on text {i}: {e}")
            continue

        if (i - skipped) > 0 and (i - skipped) % 20 == 0:
            elapsed = time.time() - start
            rate = (len(results) - (len(existing) if existing else 0)) / elapsed
            eta = (len(texts) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i}/{len(texts)}] {rate:.1f}/sec, "
                  f"ETA: {eta:.0f}min, done: {len(results)}")

    if save_file:
        save_file.close()

    elapsed = time.time() - start
    new_count = len(results) - (len(existing) if existing else 0)
    print(f"Generated {new_count} new descriptions in {elapsed/60:.1f}min "
          f"(total: {len(results)})")

    del llm  # Free memory
    return results


def collect_activations_tpu(texts: list, descriptions: list,
                            source_layers: list, output_dir: str,
                            max_positions: int = 3, device: str = "cpu"):
    """Collect activations using the full model (optionally on TPU)."""
    from model import load_model_and_tokenizer, ActivationCollector

    act_dir = os.path.join(output_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)

    print(f"\nLoading full model for activation collection on {device}...")
    model, tokenizer = load_model_and_tokenizer(device=device)

    collector = ActivationCollector(model, source_layers)
    collector.register_hooks()

    dataset = []
    total = 0

    # Build text lookup
    text_to_desc = {}
    for d in descriptions:
        text_to_desc[d["text"][:80]] = d

    print(f"Collecting activations for {len(descriptions)} texts, "
          f"layers {source_layers}...")
    start = time.time()

    for desc_item in descriptions:
        text = desc_item["text"]
        description = desc_item["description"]

        try:
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                continue

            # Select content-word positions
            candidates = []
            for pos in range(1, seq_len - 1):
                tok_str = tokenizer.decode([input_ids[0, pos].item()])
                if len(tok_str.strip()) > 2:
                    candidates.append(pos)
            if not candidates:
                candidates = list(range(1, seq_len - 1))

            num_pos = min(max_positions, len(candidates))
            positions = sorted(random.sample(candidates, num_pos))

            # Forward pass
            collector.activations = {}
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            # Save per-layer activations
            for layer_idx in source_layers:
                if layer_idx not in collector.activations:
                    continue

                acts = collector.activations[layer_idx]
                position_acts = acts[0, positions, :].cpu()

                act_file = f"act_{total}_L{layer_idx}.pt"
                torch.save(position_acts, os.path.join(act_dir, act_file))

                example = {
                    "idx": total,
                    "text": text[:500],
                    "description": description,
                    "layer": layer_idx,
                    "positions": positions,
                    "num_activations": num_pos,
                    "activation_file": act_file,
                    "seq_len": seq_len,
                }
                dataset.append(example)
                total += 1

        except Exception as e:
            print(f"  Error: {e}")
            continue

        if len(dataset) % 100 == 0 and len(dataset) > 0:
            elapsed = time.time() - start
            print(f"  {len(dataset)} examples in {elapsed:.0f}s")

    collector.clear()
    elapsed = time.time() - start
    print(f"Collected {len(dataset)} examples in {elapsed:.0f}s")

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset: {dataset_path}")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_desc_tokens", type=int, default=128)
    parser.add_argument("--layers", type=str, default="9,18,27")
    parser.add_argument("--max_positions", type=int, default=3)
    parser.add_argument("--n_threads", type=int, default=64)
    parser.add_argument("--act_device", type=str, default="cpu",
                       help="Device for activation collection (cpu or xla)")
    args = parser.parse_args()

    source_layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0: Load texts
    print(f"Loading {args.num_texts} texts...")
    texts = load_texts(args.num_texts)
    print(f"Got {len(texts)} texts")

    # Step 1: Fast description generation with GGUF
    desc_path = os.path.join(args.output_dir, "descriptions.jsonl")
    descriptions = generate_descriptions_gguf(
        texts,
        max_tokens=args.max_desc_tokens,
        save_path=desc_path,
        n_threads=args.n_threads,
    )

    # Step 2: Activation collection with full model
    dataset = collect_activations_tpu(
        texts, descriptions,
        source_layers=source_layers,
        output_dir=args.output_dir,
        max_positions=args.max_positions,
        device=args.act_device,
    )

    print(f"\nDone! {len(dataset)} training examples")

    # Auto commit
    os.system(
        f"cd /home/smitop2/ao-self-distill && "
        f"git add data/dataset.jsonl data/descriptions.jsonl && "
        f"git commit -m 'Generated {len(dataset)} training examples' && "
        f"git push origin master"
    )


if __name__ == "__main__":
    main()
