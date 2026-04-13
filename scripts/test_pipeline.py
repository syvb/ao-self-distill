#!/usr/bin/env python3
"""
Quick end-to-end test of the full AO pipeline.

Tests activation collection, description generation, and a single training step
on a handful of examples. Runs on CPU.
"""

import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    load_model_and_tokenizer,
    ActivationCollector,
    ActivationInjector,
    build_oracle_prompt,
    get_placeholder_token_id,
    DEFAULT_SOURCE_LAYERS,
)
from src.data import generate_description, DESCRIPTION_PROMPTS
from src.train import setup_lora, create_injection_hook, collate_ao_batch


def find_model_path():
    from pathlib import Path
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


def main():
    model_path = find_model_path()
    print(f"Model: {model_path}")

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_path, device="cpu")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Test texts
    texts = [
        "Le chat est assis sur le tapis.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process data through multiple layers of computation.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    print("\n=== Test 1: Activation Collection ===")
    collector = ActivationCollector(model, [9, 18, 27])
    all_activations = {}

    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        with collector.collect():
            with torch.no_grad():
                model(input_ids=input_ids)
            acts = {}
            for layer in [9, 18, 27]:
                # Take a subset of positions
                n_pos = min(5, seq_len)
                positions = list(range(0, seq_len, max(1, seq_len // n_pos)))[:n_pos]
                acts[layer] = collector.get_activations(layer, positions).cpu()
            all_activations[i] = acts

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        print(f"  Text {i}: {text[:50]}... ({seq_len} tokens)")
        for layer in [9, 18, 27]:
            print(f"    Layer {layer}: {acts[layer].shape}, norm={acts[layer].norm(dim=-1).mean():.1f}")

    print("\n=== Test 2: Description Generation ===")
    descriptions = []
    for i, text in enumerate(texts[:2]):  # Only 2 to save time
        print(f"  Generating description for text {i}...")
        t0 = time.time()
        desc = generate_description(model, tokenizer, text, max_new_tokens=128, prompt_idx=0)
        descriptions.append(desc)
        print(f"    [{time.time()-t0:.1f}s] {desc[:150]}...")

    print("\n=== Test 3: Oracle Prompt Construction ===")
    for num_acts in [1, 3, 5]:
        prompt, positions, tokens = build_oracle_prompt(
            tokenizer, num_acts, source_layer=18,
            question="Describe the semantic content of this text."
        )
        print(f"  {num_acts} acts -> prompt: {prompt[:80]}...")
        print(f"    Placeholder positions: {positions}")

    print("\n=== Test 4: Activation Injection ===")
    # Build an oracle prompt with 3 placeholder tokens
    prompt_text, placeholder_pos, prompt_tokens = build_oracle_prompt(
        tokenizer, 3, source_layer=18,
        question="Describe the semantic content of this text."
    )
    prompt_ids = prompt_tokens["input_ids"].to(model.device)

    # Get 3 activation vectors from text 0
    source_acts = all_activations[0][18][:3]  # (3, 4096)

    # Test injection
    injector = ActivationInjector(model)
    with injector.inject(placeholder_pos, source_acts):
        with torch.no_grad():
            outputs = model(input_ids=prompt_ids)
            logits = outputs.logits
    print(f"  Injection output logits shape: {logits.shape}")

    # Decode the next token prediction
    next_token = logits[0, -1, :].argmax()
    print(f"  Next token after injection: '{tokenizer.decode([next_token])}'")

    print("\n=== Test 5: Training Step (LoRA) ===")
    # Set up LoRA
    model.train()
    model_lora = setup_lora(model, rank=8, alpha=16)

    # Create a fake batch
    fake_batch = []
    for i in range(2):
        acts = all_activations[i % len(all_activations)][18][:3]
        fake_batch.append({
            "oracle_text": f"Layer 18: ? ? ? Describe the semantic content of this text.",
            "target_text": descriptions[i % len(descriptions)] if descriptions else "This is a test description.",
            "activations": acts,
            "source_layer": 18,
            "num_activations": 3,
        })

    collated = collate_ao_batch(fake_batch, tokenizer, max_length=256)
    print(f"  Batch input_ids: {collated['input_ids'].shape}")
    print(f"  Batch labels: {collated['labels'].shape}")
    print(f"  Batch activations: {collated['activations'].shape}")
    print(f"  Placeholder positions: {collated['placeholder_positions']}")

    # Set up injection hook
    base_model = model_lora.base_model.model
    hook_fn, hook_container = create_injection_hook(2)
    hook_handle = base_model.model.layers[2].register_forward_hook(hook_fn)

    hook_container["positions"] = collated["placeholder_positions"]
    hook_container["vectors"] = collated["activations"]
    hook_container["active"] = True

    # Forward + backward
    outputs = model_lora(
        input_ids=collated["input_ids"],
        attention_mask=collated["attention_mask"],
        labels=collated["labels"],
    )
    loss = outputs.loss
    print(f"  Loss: {loss.item():.4f}")

    loss.backward()
    print("  Backward pass successful!")

    # Check gradients exist on LoRA params
    grad_count = sum(1 for p in model_lora.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_trainable = sum(1 for p in model_lora.parameters() if p.requires_grad)
    print(f"  Parameters with gradients: {grad_count}/{total_trainable}")

    hook_handle.remove()

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
