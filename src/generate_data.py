"""Generate self-distillation training data.

Pipeline:
1. Load diverse text passages
2. Run Qwen3-8B on each passage, extract activations at selected positions
3. Prompt Qwen3-8B to describe the semantic content of each passage
4. Save (text, activations, description) triples as training data
"""

import json
import os
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from config import ModelConfig, DataConfig
from activation_utils import ActivationExtractor


# Diverse description prompts to avoid formulaic outputs
DESCRIPTION_PROMPTS = [
    # Rich semantic descriptions
    (
        "Analyze the following text passage in detail. Describe: the language, topic, "
        "grammatical structures, what the text communicates, what continuations are likely, "
        "and any notable features.\n\nText: \"{text}\"\n\nDetailed analysis:"
    ),
    # Focus on a specific token
    (
        "Describe the semantic content of the following text, focusing on what is happening "
        "at and around the word '{token}' (position {pos}/{total}).\n\n"
        "Text: \"{text}\"\n\nDescription covering language, topic, grammar, meaning, "
        "and likely continuations:"
    ),
    # Model beliefs framing
    (
        "Consider the following text: \"{text}\"\n\n"
        "What would a language model's internal representation at the word '{token}' "
        "likely encode? Describe the semantic, syntactic, and contextual information "
        "that would be present in the model's activations at this position."
    ),
    # Concise summary style
    (
        "Briefly characterize this text passage:\n\"{text}\"\n\n"
        "Cover: language, domain/topic, register/style, key entities, and what "
        "information a reader would extract from it."
    ),
    # Context prediction framing
    (
        "Given this text: \"{text}\"\n\n"
        "Describe what a language model would know about the context at the word "
        "'{token}'. What has come before? What is likely to follow? What is the "
        "overall meaning being conveyed?"
    ),
]


def load_text_passages(config: DataConfig, max_passages: int = None) -> List[str]:
    """Load diverse text passages from multiple sources."""
    passages = []
    target = max_passages or config.num_passages

    print(f"Loading text passages (target: {target})...")

    # Source 1: FineWeb (English web text) - use a streaming subset
    try:
        print("  Loading FineWeb samples...")
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                         split="train", streaming=True)
        count = 0
        for example in ds:
            text = example.get("text", "")
            if text and 100 < len(text) < 2000:
                passages.append(text[:1500])
                count += 1
                if count >= target // 2:
                    break
                if count % 1000 == 0:
                    print(f"    Loaded {count} FineWeb passages")
        print(f"  Got {count} FineWeb passages")
    except Exception as e:
        print(f"  FineWeb loading failed: {e}")

    # Source 2: Wikipedia (multilingual)
    try:
        print("  Loading Wikipedia samples...")
        for lang in ["en", "fr", "de", "es", "zh"]:
            try:
                ds = load_dataset("wikipedia", f"20220301.{lang}",
                                 split="train", streaming=True)
                count = 0
                for example in ds:
                    text = example.get("text", "")
                    if text and 100 < len(text) < 2000:
                        passages.append(text[:1500])
                        count += 1
                        if count >= target // 10:
                            break
                print(f"    Got {count} {lang} Wikipedia passages")
            except Exception as e:
                print(f"    Wikipedia {lang} failed: {e}")
    except Exception as e:
        print(f"  Wikipedia loading failed: {e}")

    # If we don't have enough, generate some synthetic diverse text
    if len(passages) < target // 2:
        print(f"  Generating synthetic passages to fill gap...")
        synthetic = generate_synthetic_passages(target - len(passages))
        passages.extend(synthetic)

    random.shuffle(passages)
    passages = passages[:target]
    print(f"Total passages loaded: {len(passages)}")
    return passages


def generate_synthetic_passages(n: int) -> List[str]:
    """Generate synthetic diverse passages for bootstrapping."""
    # These are short seed passages spanning different domains/languages
    seeds = [
        "The quantum computer achieved error correction using a new topological approach.",
        "Le chat est assis sur le tapis, regardant par la fenetre avec curiosite.",
        "In 2024, global renewable energy capacity exceeded fossil fuel capacity for the first time.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "The patient presented with acute respiratory distress and bilateral infiltrates.",
        "Mix flour, sugar, and butter until crumbly. Add eggs one at a time.",
        "Der Baum im Garten tragt dieses Jahr besonders viele Apfel.",
        "Once upon a time, in a kingdom far away, there lived a wise old owl.",
        "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name;",
        "The GDP growth rate of 3.2% exceeded analysts' expectations of 2.8%.",
        "She walked through the empty streets, the rain falling steadily on her umbrella.",
        "According to Einstein's theory of general relativity, massive objects warp spacetime.",
        "The board of directors approved the merger with a 7-2 vote.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
        "The Tokyo Olympics were postponed to 2021 due to the global pandemic.",
    ]
    result = []
    for i in range(n):
        result.append(seeds[i % len(seeds)])
    return result


def select_token_positions(input_ids: torch.Tensor, tokenizer,
                          min_pos: int = 1, max_pos: int = 5) -> List[int]:
    """Select semantically interesting token positions for activation extraction.

    Prefers content words over function words/punctuation.
    """
    seq_len = input_ids.shape[1]
    if seq_len <= 2:
        return [0]

    # Decode each token to check if it's a content word
    content_positions = []
    function_positions = []

    for i in range(1, seq_len - 1):  # Skip BOS/EOS
        token_str = tokenizer.decode([input_ids[0, i].item()])
        stripped = token_str.strip()
        # Simple heuristic: content words are longer and not punctuation
        if len(stripped) > 2 and stripped.isalpha():
            content_positions.append(i)
        elif stripped:
            function_positions.append(i)

    # Prefer content words, fall back to any position
    candidates = content_positions if content_positions else function_positions
    if not candidates:
        candidates = list(range(1, seq_len - 1))

    num_positions = random.randint(min_pos, min(max_pos, len(candidates)))
    return sorted(random.sample(candidates, num_positions))


def generate_description(model, tokenizer, text: str, token: str,
                        pos: int, total: int,
                        config: DataConfig) -> str:
    """Generate a semantic description of a text passage using the model itself."""
    # Pick a random prompt template
    template = random.choice(DESCRIPTION_PROMPTS)
    prompt = template.format(text=text[:500], token=token, pos=pos, total=total)

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                      max_length=1024)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_description_tokens,
            temperature=config.description_temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][input_ids.shape[1]:]
    description = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return description.strip()


def process_passage(model, tokenizer, extractor: ActivationExtractor,
                   text: str, passage_idx: int,
                   model_config: ModelConfig,
                   data_config: DataConfig) -> List[Dict]:
    """Process a single text passage: extract activations and generate descriptions."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=data_config.max_passage_tokens)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    seq_len = input_ids.shape[1]
    if seq_len < 5:
        return []

    # Select token positions
    positions = select_token_positions(
        input_ids, tokenizer,
        data_config.min_positions_per_passage,
        data_config.max_positions_per_passage
    )

    # Extract activations at all selected layers
    activations = extractor.extract(input_ids, attention_mask, positions)

    examples = []
    for layer_idx in model_config.extraction_layers:
        if layer_idx not in activations:
            continue

        layer_acts = activations[layer_idx]  # [1, num_positions, hidden_size]

        for pos_i, pos in enumerate(positions):
            token_str = tokenizer.decode([input_ids[0, pos].item()])

            # Generate description using the model
            description = generate_description(
                model, tokenizer, text, token_str.strip(),
                pos, seq_len, data_config
            )

            if not description or len(description) < 20:
                continue

            # Extract the single activation vector
            act_vector = layer_acts[0, pos_i, :].cpu().numpy()

            example = {
                "passage_idx": passage_idx,
                "text": text[:500],
                "token": token_str.strip(),
                "token_position": pos,
                "sequence_length": seq_len,
                "layer": layer_idx,
                "description": description,
                "activation_file": f"act_{passage_idx}_{layer_idx}_{pos}.npy",
            }
            examples.append((example, act_vector))

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_passages", type=int, default=500,
                       help="Number of passages to process")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting passage index (for parallel runs)")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, xla, cuda)")
    args = parser.parse_args()

    model_config = ModelConfig()
    data_config = DataConfig()
    data_config.num_passages = args.num_passages
    data_config.data_dir = args.output_dir

    # Create output directories
    os.makedirs(f"{args.output_dir}/activations", exist_ok=True)
    os.makedirs(f"{args.output_dir}/descriptions", exist_ok=True)

    print(f"Loading model {model_config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "xla":
        import torch_xla
        device = torch_xla.device()
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=args.device,
        )

    model.eval()
    print(f"Model loaded on {model.device}")

    # Set up activation extractor
    extractor = ActivationExtractor(model, model_config.extraction_layers)
    extractor.register_hooks()

    # Load text passages
    passages = load_text_passages(data_config, args.num_passages)

    # Process passages
    all_examples = []
    dataset_file = f"{args.output_dir}/training_dataset_{args.start_idx}.jsonl"

    print(f"\nProcessing {len(passages)} passages...")
    start_time = time.time()

    for i, text in enumerate(passages):
        global_idx = args.start_idx + i

        if i > 0 and i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (len(passages) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(passages)}] {rate:.1f} passages/sec, "
                  f"ETA: {eta/60:.1f}min, examples so far: {len(all_examples)}")

        try:
            examples = process_passage(
                model, tokenizer, extractor, text, global_idx,
                model_config, data_config
            )

            for example_dict, act_vector in examples:
                # Save activation vector
                act_path = f"{args.output_dir}/activations/{example_dict['activation_file']}"
                np.save(act_path, act_vector)

                all_examples.append(example_dict)

                # Periodically save the dataset
                if len(all_examples) % 100 == 0:
                    _save_dataset(all_examples, dataset_file)

        except Exception as e:
            print(f"  Error processing passage {global_idx}: {e}")
            continue

    # Final save
    _save_dataset(all_examples, dataset_file)
    extractor.clear_hooks()

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(all_examples)} training examples in {elapsed/60:.1f}min")
    print(f"Dataset saved to {dataset_file}")


def _save_dataset(examples: List[Dict], path: str):
    """Save examples to JSONL file."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
