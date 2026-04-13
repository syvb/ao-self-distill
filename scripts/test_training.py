#!/usr/bin/env python3
"""
Quick end-to-end test of the training pipeline.
Creates a tiny dataset and runs a few training steps.
"""

import os
import sys
import json
import time
import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    load_model_and_tokenizer, ActivationCollector, ActivationInjector,
    build_oracle_prompt, collect_activations_for_text, get_placeholder_token_id,
)
from data import AODataset, generate_description, ORACLE_QUESTIONS
from train_tpu import setup_lora, InjectionHookManager, collate_ao_batch, train_ao


def create_test_data(model, tokenizer, output_dir: str = "test_data", n: int = 10):
    """Create a tiny test dataset."""
    os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

    texts = [
        "The quantum computer achieved error correction using a new topological approach.",
        "Le chat est assis sur le tapis, regardant par la fenetre.",
        "Machine learning models require large amounts of training data.",
        "The GDP growth rate of 3.2% exceeded analysts' expectations.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "She walked through the empty streets, the rain falling steadily.",
        "Photosynthesis converts carbon dioxide and water into glucose.",
        "The board of directors approved the merger with a 7-2 vote.",
        "Der Baum im Garten tragt dieses Jahr besonders viele Apfel.",
        "According to Einstein's theory of general relativity, massive objects warp spacetime.",
    ][:n]

    dataset = []
    layer = 18

    print(f"Creating test data ({n} texts)...")
    collector = ActivationCollector(model, [layer])
    collector.register_hooks()

    for i, text in enumerate(texts):
        # Collect activations
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(model.device)

        seq_len = input_ids.shape[1]
        positions = [min(3, seq_len - 2)]  # Simple: just one position

        collector.activations = {}
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=inputs["attention_mask"].to(model.device))

        acts = collector.activations[layer]
        position_acts = acts[0, positions, :].cpu()

        # Generate a simple description
        description = generate_description(model, tokenizer, text, max_new_tokens=80)
        if not description or len(description) < 10:
            description = f"This is a text about: {text[:100]}"

        act_file = f"act_{i}_L{layer}.pt"
        torch.save(position_acts, os.path.join(output_dir, "activations", act_file))

        example = {
            "idx": i,
            "text": text,
            "description": description,
            "layer": layer,
            "positions": positions,
            "num_activations": len(positions),
            "activation_file": act_file,
            "seq_len": seq_len,
            "text_idx": i,
        }
        dataset.append(example)
        print(f"  [{i+1}/{n}] {text[:40]}... -> {description[:40]}...")

    collector.clear()

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Test data saved: {len(dataset)} examples in {output_dir}/")
    return dataset_path


def test_training(model, tokenizer, dataset_path: str, data_dir: str,
                  device, num_steps: int = 5):
    """Run a few training steps to verify everything works."""
    print(f"\n{'='*50}")
    print("Testing training pipeline")
    print(f"{'='*50}")

    # Apply LoRA
    model = setup_lora(model, rank=16, alpha=32, dropout=0.0)

    # Load dataset
    activations_dir = os.path.join(data_dir, "activations")
    dataset = AODataset(
        activation_dir=activations_dir,
        descriptions_path=dataset_path,
        tokenizer=tokenizer,
        source_layers=[18],
    )

    print(f"Dataset: {len(dataset)} examples")

    # Quick test: load one example
    sample = dataset[0]
    print(f"Sample oracle: {sample['oracle_text'][:80]}...")
    print(f"Sample target: {sample['target_text'][:80]}...")
    print(f"Sample activations: {sample['activations'].shape}")

    # Run a few training steps
    output_dir = os.path.join(data_dir, "test_checkpoints")
    model = train_ao(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        save_every=999,
        log_every=1,
        device=device,
        use_tpu=False,
    )

    print("\nTraining test PASSED!")
    return model


def test_inference(model, tokenizer, data_dir: str):
    """Test oracle inference after training."""
    print(f"\n{'='*50}")
    print("Testing oracle inference")
    print(f"{'='*50}")

    # Load a test activation
    activations_dir = os.path.join(data_dir, "activations")
    act_files = list(Path(activations_dir).glob("*.pt"))
    if not act_files:
        print("No activation files found!")
        return

    # Load the dataset to get metadata
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    with open(dataset_path) as f:
        first_example = json.loads(f.readline())

    activations = torch.load(act_files[0], weights_only=True)
    layer = first_example["layer"]
    num_acts = activations.shape[0]

    # Build oracle prompt
    oracle_prompt, ph_positions, tokens = build_oracle_prompt(
        tokenizer, num_acts, layer,
    )

    print(f"Oracle prompt: {oracle_prompt}")
    print(f"Placeholder positions: {ph_positions}")

    # Inject and generate
    injector = ActivationInjector(model)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    with injector.inject(ph_positions, activations):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nOriginal text: {first_example['text'][:100]}")
    print(f"Ground truth:  {first_example['description'][:150]}")
    print(f"Generated:     {generated[:150]}")
    print("\nInference test PASSED!")


from pathlib import Path

def main():
    test_dir = "test_data"
    device_str = "cpu"

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(device=device_str)
    device = model.device

    # Step 1: Create test data
    dataset_path = create_test_data(model, tokenizer, test_dir, n=6)

    # Step 2: Test training
    model = test_training(model, tokenizer, dataset_path, test_dir, device)

    # Step 3: Test inference
    test_inference(model, tokenizer, test_dir)

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)


if __name__ == "__main__":
    main()
