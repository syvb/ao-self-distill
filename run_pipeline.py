#!/usr/bin/env python3
"""
Main pipeline for Self-Distillation Activation Oracle.

This script orchestrates the full pipeline:
1. Load text data
2. Generate semantic descriptions (self-distillation labels)
3. Collect activations from the model
4. Train the activation oracle with LoRA
5. Evaluate

Usage:
    # Generate data (CPU - uses model for both activations and descriptions)
    python run_pipeline.py generate --num_samples 1000 --device cpu

    # Train (TPU)
    python run_pipeline.py train --device xla

    # Evaluate
    python run_pipeline.py eval --checkpoint checkpoints/final
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def cmd_generate(args):
    """Generate training data: text passages, descriptions, and activations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model import (
        ActivationCollector, load_model_and_tokenizer,
        DEFAULT_SOURCE_LAYERS, collect_activations_for_text,
    )
    from src.data import (
        load_text_samples, load_simple_text_samples,
        generate_description, ORACLE_QUESTIONS,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "activations"), exist_ok=True)

    # Load model
    print(f"Loading model on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(device=args.device)
    print(f"Model loaded on {model.device}")

    # Load text samples
    print(f"\nLoading {args.num_samples} text samples...")
    try:
        samples = load_text_samples(args.num_samples)
    except Exception as e:
        print(f"Full dataset loading failed ({e}), falling back to simple samples...")
        samples = load_simple_text_samples(args.num_samples)

    if not samples:
        print("ERROR: No text samples loaded. Generating minimal synthetic data.")
        samples = _synthetic_samples(args.num_samples)

    print(f"Got {len(samples)} text samples")

    # Process each sample: generate description + collect activations
    dataset_path = os.path.join(args.output_dir, "dataset.jsonl")
    existing_count = 0

    # Resume support
    if os.path.exists(dataset_path):
        with open(dataset_path) as f:
            existing_count = sum(1 for _ in f)
        print(f"Resuming from {existing_count} existing examples")

    dataset_file = open(dataset_path, "a")
    source_layers = DEFAULT_SOURCE_LAYERS
    total_generated = existing_count

    print(f"\nGenerating descriptions and collecting activations...")
    start_time = time.time()

    for i, text in enumerate(samples):
        if i < existing_count:
            continue

        if i > 0 and i % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i - existing_count) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i}/{len(samples)}] {rate:.2f} samples/sec, "
                  f"ETA: {eta:.1f}min, total: {total_generated}")

        try:
            # Step 1: Generate description
            description = generate_description(model, tokenizer, text, max_new_tokens=200)
            if not description or len(description) < 20:
                continue

            # Step 2: Collect activations at selected positions
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            seq_len = inputs["input_ids"].shape[1]

            # Select 1-3 interesting token positions
            num_positions = min(random.randint(1, 3), seq_len - 2)
            if num_positions < 1:
                continue

            # Prefer content-word positions (longer tokens, not at edges)
            candidates = []
            for pos in range(1, seq_len - 1):
                tok_str = tokenizer.decode([inputs["input_ids"][0, pos].item()])
                if len(tok_str.strip()) > 2:
                    candidates.append(pos)

            if not candidates:
                candidates = list(range(1, seq_len - 1))

            positions = sorted(random.sample(candidates, min(num_positions, len(candidates))))

            # Collect activations
            act_data, _, _ = collect_activations_for_text(
                model, tokenizer, text,
                source_layers=source_layers,
                token_positions=positions,
                max_length=256,
            )

            # Save each layer's activations and create training examples
            for layer_idx in source_layers:
                if layer_idx not in act_data:
                    continue

                acts = act_data[layer_idx]  # (num_positions, hidden_size)

                # Save activations
                act_filename = f"activations_{total_generated}_L{layer_idx}.pt"
                torch.save(acts, os.path.join(args.output_dir, "activations", act_filename))

                # Create training example
                example = {
                    "idx": total_generated,
                    "text": text[:500],
                    "description": description,
                    "layer": layer_idx,
                    "positions": positions,
                    "num_activations": len(positions),
                    "activation_file": act_filename,
                    "seq_len": seq_len,
                }
                dataset_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_generated += 1

            # Flush periodically
            if i % 50 == 0:
                dataset_file.flush()

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    dataset_file.close()
    elapsed = time.time() - start_time
    print(f"\nData generation complete!")
    print(f"  Total examples: {total_generated}")
    print(f"  Time: {elapsed/60:.1f}min")
    print(f"  Dataset: {dataset_path}")


def cmd_train(args):
    """Train the activation oracle."""
    from src.model import load_model_and_tokenizer, INJECTION_LAYER
    from src.train import setup_lora, create_injection_hook, collate_ao_batch, train_ao
    from src.data import AODataset, ORACLE_QUESTIONS

    print(f"Loading model on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(device=args.device)

    # Apply LoRA
    model = setup_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.05,
    )

    # Load dataset
    dataset_path = os.path.join(args.data_dir, "dataset.jsonl")
    activations_dir = os.path.join(args.data_dir, "activations")

    dataset = AODataset(
        activation_dir=activations_dir,
        descriptions_path=dataset_path,
        tokenizer=tokenizer,
        source_layers=[9, 18, 27],
    )

    print(f"Training with {len(dataset)} examples")

    # Train
    model = train_ao(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.grad_accum,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
    )

    print("Training complete!")


def cmd_eval(args):
    """Evaluate the trained activation oracle."""
    from src.model import (
        load_model_and_tokenizer, ActivationCollector, ActivationInjector,
        build_oracle_prompt, collect_activations_for_text, DEFAULT_SOURCE_LAYERS,
    )
    from src.data import load_jsonl, generate_description
    from peft import PeftModel

    print(f"Loading base model...")
    model, tokenizer = load_model_and_tokenizer(device=args.device)

    # Load LoRA weights
    if args.checkpoint:
        print(f"Loading LoRA from {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)

    model.eval()

    # Load eval data
    dataset_path = os.path.join(args.data_dir, "dataset.jsonl")
    examples = load_jsonl(dataset_path)
    random.shuffle(examples)
    eval_examples = examples[:args.num_eval]

    print(f"\nEvaluating on {len(eval_examples)} examples...")

    results = []
    injector = ActivationInjector(model)

    for i, ex in enumerate(eval_examples):
        try:
            # Load activations
            act_path = os.path.join(args.data_dir, "activations", ex["activation_file"])
            if not os.path.exists(act_path):
                continue

            activations = torch.load(act_path, weights_only=True)
            layer = ex["layer"]
            num_acts = activations.shape[0]

            # Build oracle prompt
            oracle_prompt, ph_positions, tokens = build_oracle_prompt(
                tokenizer, num_acts, layer,
                question="Describe the semantic content of this text."
            )

            # Inject activations and generate
            input_ids = tokens["input_ids"].to(model.device)
            attention_mask = tokens["attention_mask"].to(model.device)

            with injector.inject(ph_positions, activations):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            result = {
                "idx": i,
                "text": ex["text"][:200],
                "ground_truth": ex["description"][:300],
                "generated": generated[:300],
                "layer": layer,
            }
            results.append(result)

            if i < 5:
                print(f"\n--- Example {i} ---")
                print(f"Text: {ex['text'][:100]}...")
                print(f"Ground truth: {ex['description'][:150]}...")
                print(f"Generated: {generated[:150]}...")

        except Exception as e:
            print(f"  Error evaluating example {i}: {e}")
            continue

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nEvaluation complete! {len(results)} examples evaluated")
    print(f"Results saved to {output_path}")


def _synthetic_samples(n):
    """Generate minimal synthetic text samples."""
    templates = [
        "The quantum computer achieved error correction using a new topological approach to stabilize qubits at room temperature.",
        "Le chat est assis sur le tapis, regardant par la fenetre avec curiosite. Il attend son maitre.",
        "In 2024, global renewable energy capacity exceeded fossil fuel capacity for the first time in history.",
        "The patient presented with acute respiratory distress and bilateral infiltrates on chest X-ray.",
        "Mix flour, sugar, and butter until crumbly. Add eggs one at a time, beating well after each addition.",
        "Der Baum im Garten tragt dieses Jahr besonders viele Apfel. Die Ernte wird gut sein.",
        "Once upon a time, in a kingdom far away, there lived a wise old owl who counseled the king.",
        "The GDP growth rate of 3.2% exceeded analysts' expectations of 2.8%, driving markets higher.",
        "She walked through the empty streets, the rain falling steadily on her umbrella as she headed home.",
        "According to Einstein's theory of general relativity, massive objects warp spacetime around them.",
        "The board of directors approved the $5.2 billion merger with a 7-2 vote after months of negotiations.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy captured by chlorophyll.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)  # recursive implementation",
        "The Tokyo Olympics were postponed to 2021 due to the global pandemic that started in early 2020.",
        "Machine learning models require large amounts of training data and significant computational resources.",
    ]
    result = []
    for i in range(n):
        result.append(templates[i % len(templates)])
    return result


def main():
    parser = argparse.ArgumentParser(description="Self-Distillation Activation Oracle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate data
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--num_samples", type=int, default=500)
    gen_parser.add_argument("--output_dir", type=str, default="data")
    gen_parser.add_argument("--device", type=str, default="cpu")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the activation oracle")
    train_parser.add_argument("--data_dir", type=str, default="data")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--learning_rate", type=float, default=1e-5)
    train_parser.add_argument("--lora_rank", type=int, default=64)
    train_parser.add_argument("--lora_alpha", type=int, default=128)
    train_parser.add_argument("--grad_accum", type=int, default=4)
    train_parser.add_argument("--save_every", type=int, default=200)
    train_parser.add_argument("--log_every", type=int, default=10)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate the oracle")
    eval_parser.add_argument("--data_dir", type=str, default="data")
    eval_parser.add_argument("--checkpoint", type=str, default=None)
    eval_parser.add_argument("--output_dir", type=str, default="eval_results")
    eval_parser.add_argument("--device", type=str, default="cpu")
    eval_parser.add_argument("--num_eval", type=int, default=100)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
