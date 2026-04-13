#!/usr/bin/env python3
"""
TPU-friendly data generation using a custom fixed-shape generation loop.

The key insight: torch_xla is slow with model.generate() because each step
has different shapes (growing sequence length). By pre-allocating to max
length and using attention masks, we can get XLA to compile once.

Pipeline:
1. Prepare all prompts with padding to a fixed length
2. Use TPU for batch forward passes with KV-cache workaround
3. Collect activations in the same forward passes
"""

import json
import os
import sys
import time
import random
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_texts(num_texts: int) -> list:
    """Load text samples."""
    from datasets import load_dataset
    random.seed(42)
    samples = []

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

    if len(samples) < num_texts:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                            split="train", streaming=True)
            for item in ds:
                text = item["text"].strip()
                if 100 < len(text) < 1000:
                    words = text.split()
                    if len(words) > 20:
                        start = random.randint(0, max(0, len(words) - 60))
                        chunk = " ".join(words[start:start + 60])
                        samples.append(chunk)
                        if len(samples) >= num_texts:
                            break
        except Exception as e:
            print(f"FineWeb failed: {e}")

    random.shuffle(samples)
    return samples[:num_texts]


DESCRIPTION_PROMPTS = [
    "Describe the semantic content of the following text in detail. Cover: language, topic, grammar, sentiment, and likely continuations.\n\nText: \"{text}\"\n\nAnalysis:",
    "Analyze this text passage. What language? What subject? What style?\n\nText: \"{text}\"\n\nAnalysis:",
    "What can you tell about this text? Describe content, language, style, meaning.\n\nText: \"{text}\"\n\nAnalysis:",
    "Provide a semantic analysis: language, topic, entities, grammar, continuations.\n\nText: \"{text}\"\n\nAnalysis:",
]


def greedy_generate_fixed(model, input_ids, attention_mask, max_new_tokens=64,
                          eos_token_id=None, pad_token_id=0):
    """Custom generation loop with fixed tensor shapes for TPU compatibility.

    Instead of growing tensors, we pre-allocate max length and use masks.
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + max_new_tokens

    # Pre-allocate output buffer
    output_ids = torch.full((batch_size, total_len), pad_token_id,
                           dtype=input_ids.dtype, device=input_ids.device)
    output_ids[:, :prompt_len] = input_ids

    output_mask = torch.zeros(batch_size, total_len,
                             dtype=attention_mask.dtype, device=attention_mask.device)
    output_mask[:, :prompt_len] = attention_mask

    # Track which sequences are done
    done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    # Generate one token at a time
    # NOTE: This is still autoregressive but with fixed output tensor shapes
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            # First step: use full prompt
            outputs = model(
                input_ids=output_ids[:, :prompt_len],
                attention_mask=output_mask[:, :prompt_len],
            )
        else:
            # Subsequent steps: just the last token + KV cache from past
            cur_pos = prompt_len + step - 1
            outputs = model(
                input_ids=output_ids[:, cur_pos:cur_pos+1],
                attention_mask=output_mask[:, :cur_pos+1],
                past_key_values=past_key_values,
            )

        past_key_values = outputs.past_key_values

        # Get next token (greedy)
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Place in output
        gen_pos = prompt_len + step
        output_ids[:, gen_pos] = next_tokens
        output_mask[:, gen_pos] = 1

        # Check for EOS
        if eos_token_id is not None:
            done = done | (next_tokens == eos_token_id)
            if done.all():
                break

    return output_ids


def generate_all_data(model, tokenizer, texts, device, output_dir,
                      source_layers=[9, 18, 27], max_positions=3,
                      max_desc_tokens=64):
    """Generate descriptions and collect activations for all texts."""
    from model import ActivationCollector

    os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

    collector = ActivationCollector(model, source_layers)
    collector.register_hooks()

    dataset = []
    total = 0
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"Processing {len(texts)} texts...")
    start = time.time()

    for i, text in enumerate(texts):
        try:
            # Step 1: Generate description
            template = DESCRIPTION_PROMPTS[i % len(DESCRIPTION_PROMPTS)]
            prompt = template.format(text=text[:300])
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

            prompt_inputs = tokenizer(formatted, return_tensors="pt",
                                     truncation=True, max_length=512,
                                     padding="max_length", pad_side="left")
            prompt_ids = prompt_inputs["input_ids"].to(device)
            prompt_mask = prompt_inputs["attention_mask"].to(device)

            # Generate description
            with torch.no_grad():
                gen_ids = greedy_generate_fixed(
                    model, prompt_ids, prompt_mask,
                    max_new_tokens=max_desc_tokens,
                    eos_token_id=eos_id, pad_token_id=pad_id,
                )

            # Decode generated text
            prompt_len = prompt_mask.sum().item()
            new_ids = gen_ids[0, prompt_len:]
            description = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            if len(description) < 20:
                continue

            # Step 2: Collect activations on original text
            text_inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=256)
            text_ids = text_inputs["input_ids"].to(device)
            text_mask = text_inputs["attention_mask"].to(device)
            seq_len = text_ids.shape[1]

            if seq_len < 5:
                continue

            # Select positions
            candidates = []
            for pos in range(1, seq_len - 1):
                tok_str = tokenizer.decode([text_ids[0, pos].item()])
                if len(tok_str.strip()) > 2:
                    candidates.append(pos)
            if not candidates:
                candidates = list(range(1, seq_len - 1))

            num_pos = min(max_positions, len(candidates))
            positions = sorted(random.sample(candidates, num_pos))

            # Forward pass for activations
            collector.activations = {}
            with torch.no_grad():
                model(input_ids=text_ids, attention_mask=text_mask)

            # Save per-layer
            for layer_idx in source_layers:
                if layer_idx not in collector.activations:
                    continue

                acts = collector.activations[layer_idx]
                pos_acts = acts[0, positions, :].cpu()

                act_file = f"act_{total}_L{layer_idx}.pt"
                torch.save(pos_acts, os.path.join(output_dir, "activations", act_file))

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
            print(f"  Error on text {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(texts)}] {rate:.2f}/sec, "
                  f"ETA: {eta:.0f}min, examples: {total}")

    collector.clear()
    elapsed = time.time() - start
    print(f"\nGenerated {total} examples in {elapsed/60:.1f}min")

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also save descriptions separately
    desc_path = os.path.join(output_dir, "descriptions.jsonl")
    seen = set()
    with open(desc_path, "w") as f:
        for ex in dataset:
            key = ex["text"][:80]
            if key not in seen:
                seen.add(key)
                f.write(json.dumps({
                    "text": ex["text"],
                    "description": ex["description"],
                    "text_idx": ex["idx"],
                }, ensure_ascii=False) + "\n")

    print(f"Dataset: {dataset_path} ({total} examples)")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_desc_tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default="9,18,27")
    parser.add_argument("--max_positions", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    source_layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Qwen3-8B on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.device == "xla":
        import torch_xla
        device = torch_xla.device()
        model = model.to(device)
    else:
        device = torch.device(args.device)
        model = model.to(device)

    model.eval()
    print(f"Model on {device}")

    # Load texts
    texts = load_texts(args.num_texts)
    print(f"Loaded {len(texts)} texts")

    # Generate all data
    dataset = generate_all_data(
        model, tokenizer, texts, device, args.output_dir,
        source_layers=source_layers,
        max_positions=args.max_positions,
        max_desc_tokens=args.max_desc_tokens,
    )

    # Auto-commit
    os.system(
        f"cd /home/smitop2/ao-self-distill && "
        f"git add data/dataset.jsonl data/descriptions.jsonl && "
        f"git commit -m 'Generated {len(dataset)} training examples' && "
        f"git push origin master"
    )


if __name__ == "__main__":
    main()
