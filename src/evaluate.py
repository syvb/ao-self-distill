"""
Evaluation pipeline for the Self-Distillation Activation Oracle.

Tests the trained oracle by:
1. Injecting activations from held-out text
2. Generating descriptions from activations alone
3. Comparing to the ground-truth text-based descriptions
4. Scoring with automated metrics and qualitative analysis
"""

import json
import os
import random
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from model import (
    ActivationCollector, ActivationInjector,
    build_oracle_prompt, collect_activations_for_text,
    DEFAULT_SOURCE_LAYERS, load_model_and_tokenizer,
    get_placeholder_token_id,
)
from data import load_jsonl, generate_description


def load_oracle_model(checkpoint_path: str, device: str = "cpu"):
    """Load the base model + LoRA oracle weights."""
    print(f"Loading base model...")
    model, tokenizer = load_model_and_tokenizer(device=device)

    print(f"Loading LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    return model, tokenizer


def generate_from_activations(
    model, tokenizer, activations: torch.Tensor,
    layer: int, injector: ActivationInjector,
    question: str = "Describe the semantic content of this text.",
    max_new_tokens: int = 200,
) -> str:
    """Generate a description from activations using the oracle."""
    num_acts = activations.shape[0]

    # Build oracle prompt
    oracle_prompt, ph_positions, tokens = build_oracle_prompt(
        tokenizer, num_acts, layer, question=question,
    )

    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    # Inject and generate
    with injector.inject(ph_positions, activations):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_oracle(
    model, tokenizer,
    data_dir: str,
    num_eval: int = 100,
    output_dir: str = "eval_results",
    questions: Optional[List[str]] = None,
):
    """Run full evaluation of the oracle model."""
    os.makedirs(output_dir, exist_ok=True)

    # Load eval data
    dataset_path = os.path.join(data_dir, "dataset.jsonl")
    examples = load_jsonl(dataset_path)
    random.shuffle(examples)
    eval_examples = examples[:num_eval]

    if questions is None:
        questions = [
            "Describe the semantic content of this text.",
            "What language is this text in and what is the topic?",
            "What is the grammatical structure and meaning?",
        ]

    injector = ActivationInjector(model)
    results = []

    print(f"\nEvaluating on {len(eval_examples)} examples...")
    print(f"Questions: {len(questions)}")

    for i, ex in enumerate(eval_examples):
        act_path = os.path.join(data_dir, "activations", ex["activation_file"])
        if not os.path.exists(act_path):
            continue

        activations = torch.load(act_path, weights_only=True)
        layer = ex["layer"]

        for q_idx, question in enumerate(questions):
            try:
                generated = generate_from_activations(
                    model, tokenizer, activations, layer, injector,
                    question=question,
                )

                result = {
                    "idx": i,
                    "question_idx": q_idx,
                    "question": question,
                    "text": ex["text"][:300],
                    "ground_truth": ex["description"][:400],
                    "generated": generated[:400],
                    "layer": layer,
                    "num_activations": ex.get("num_activations", activations.shape[0]),
                }
                results.append(result)

            except Exception as e:
                print(f"  Error on example {i}, question {q_idx}: {e}")
                continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_examples)}] {len(results)} results so far")

        # Print a few examples
        if i < 3:
            print(f"\n--- Example {i} (Layer {layer}) ---")
            print(f"Text: {ex['text'][:100]}...")
            print(f"Ground truth: {ex['description'][:120]}...")
            if results:
                print(f"Generated: {results[-1]['generated'][:120]}...")

    # Save results
    output_path = os.path.join(output_dir, "eval_results.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute basic statistics
    stats = analyze_results(results)

    stats_path = os.path.join(output_dir, "eval_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"  Results: {output_path}")
    print(f"  Stats: {stats_path}")
    print(f"\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return results, stats


def analyze_results(results: List[Dict]) -> Dict:
    """Compute basic statistics on evaluation results."""
    if not results:
        return {"error": "no results"}

    stats = {
        "num_examples": len(results),
        "avg_generated_length": sum(len(r["generated"]) for r in results) / len(results),
        "avg_ground_truth_length": sum(len(r["ground_truth"]) for r in results) / len(results),
    }

    # Check for language identification accuracy
    # Simple heuristic: does the generated text mention the same language as ground truth?
    lang_keywords = ["english", "french", "german", "spanish", "chinese",
                     "japanese", "arabic", "code", "python", "sql"]
    lang_matches = 0
    lang_total = 0
    for r in results:
        gt_langs = [l for l in lang_keywords if l in r["ground_truth"].lower()]
        gen_langs = [l for l in lang_keywords if l in r["generated"].lower()]
        if gt_langs:
            lang_total += 1
            if any(l in gen_langs for l in gt_langs):
                lang_matches += 1

    if lang_total > 0:
        stats["language_identification_accuracy"] = round(lang_matches / lang_total, 3)

    # Check for topic keyword overlap
    topic_overlap_scores = []
    for r in results:
        gt_words = set(r["ground_truth"].lower().split())
        gen_words = set(r["generated"].lower().split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "for", "of", "with", "and", "or", "but", "this",
                      "that", "it", "its", "be", "as", "by", "from"}
        gt_content = gt_words - stop_words
        gen_content = gen_words - stop_words
        if gt_content:
            overlap = len(gt_content & gen_content) / len(gt_content)
            topic_overlap_scores.append(overlap)

    if topic_overlap_scores:
        stats["avg_keyword_overlap"] = round(
            sum(topic_overlap_scores) / len(topic_overlap_scores), 3
        )

    # Per-layer stats
    layers = set(r["layer"] for r in results)
    for layer in sorted(layers):
        layer_results = [r for r in results if r["layer"] == layer]
        layer_overlap = []
        for r in layer_results:
            gt_words = set(r["ground_truth"].lower().split()) - {"the", "a", "an", "is"}
            gen_words = set(r["generated"].lower().split()) - {"the", "a", "an", "is"}
            if gt_words:
                layer_overlap.append(len(gt_words & gen_words) / len(gt_words))
        if layer_overlap:
            stats[f"layer_{layer}_keyword_overlap"] = round(
                sum(layer_overlap) / len(layer_overlap), 3
            )

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Self-Distillation AO")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to LoRA checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_eval", type=int, default=50)
    args = parser.parse_args()

    model, tokenizer = load_oracle_model(args.checkpoint, device=args.device)

    results, stats = evaluate_oracle(
        model, tokenizer,
        data_dir=args.data_dir,
        num_eval=args.num_eval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
