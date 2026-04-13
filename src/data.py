"""
Data pipeline for self-supervised Activation Oracle training.

Handles:
- Loading diverse text corpora
- Generating semantic descriptions using the model
- Creating (activation, description) training pairs
- Saving/loading datasets to disk
"""

import json
import os
import random
import torch
from pathlib import Path
from typing import Optional


# Description generation prompts - varied to get diverse descriptions
DESCRIPTION_PROMPTS = [
    (
        "Describe the semantic content of the following text in detail. "
        "Cover: the language used, the topic, grammatical structures, "
        "sentiment/tone, and likely continuations.\n\nText: \"{text}\""
    ),
    (
        "Analyze this text passage. What language is it in? What is the subject matter? "
        "What is the writing style and register? What might come next?\n\nText: \"{text}\""
    ),
    (
        "You are examining a text passage. Describe what you observe about it: "
        "its language, topic, structure, tone, and what information it conveys. "
        "Also predict what might follow.\n\nText: \"{text}\""
    ),
    (
        "Provide a comprehensive semantic analysis of this text. Include: "
        "(1) language identification, (2) topic/domain, (3) key entities or concepts, "
        "(4) grammatical features, (5) pragmatic intent, (6) likely continuations.\n\n"
        "Text: \"{text}\""
    ),
    (
        "What can you tell about this text? Describe its content, language, style, "
        "and meaning as thoroughly as you can.\n\nText: \"{text}\""
    ),
]

# Oracle question templates used during training
ORACLE_QUESTIONS = [
    "Describe the semantic content of this text.",
    "What is this text about? Describe it in detail.",
    "Analyze the content represented by these activations.",
    "What language, topic, and meaning are encoded here?",
    "Describe what information is contained in these activations.",
]


def load_text_samples(
    num_samples: int = 10000,
    max_length: int = 256,
    seed: int = 42,
) -> list[str]:
    """Load diverse text samples from HuggingFace datasets.

    Returns a list of text passages.
    """
    from datasets import load_dataset

    random.seed(seed)
    samples = []

    # Use wikitext (always available, no auth needed)
    try:
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        count = 0
        for item in wiki:
            text = item["text"].strip()
            if len(text) > 100:
                words = text.split()
                if len(words) > 20:
                    start = random.randint(0, max(0, len(words) - 60))
                    chunk = " ".join(words[start:start + 60])
                    if len(chunk) > 80:
                        samples.append(chunk)
                        count += 1
                        if count >= num_samples // 2:
                            break
        print(f"  Loaded {count} wikitext samples")
    except Exception as e:
        print(f"Warning: Could not load wikitext: {e}")

    # Try FineWeb for web text diversity
    try:
        fineweb = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                               split="train", streaming=True)
        count = 0
        for item in fineweb:
            text = item["text"].strip()
            if len(text) > 100:
                words = text.split()
                if len(words) > 30:
                    start = random.randint(0, max(0, len(words) - 60))
                    chunk = " ".join(words[start:start + 60])
                    samples.append(chunk)
                    count += 1
                    if count >= num_samples // 2:
                        break
        print(f"  Loaded {count} FineWeb samples")
    except Exception as e:
        print(f"Warning: Could not load FineWeb: {e}")

    random.shuffle(samples)
    print(f"Loaded {len(samples)} text samples")
    return samples[:num_samples]


def load_simple_text_samples(num_samples: int = 5000, seed: int = 42) -> list[str]:
    """Load text samples from easily available datasets as a fallback."""
    from datasets import load_dataset

    random.seed(seed)
    samples = []

    # Use wikitext which is small and always available
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
                        if len(samples) >= num_samples:
                            break
    except Exception as e:
        print(f"Warning: Could not load wikitext: {e}")

    # Also use tiny_shakespeare or similar simple datasets
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        for item in ds:
            text = item["text"].strip()
            if len(text) > 80:
                samples.append(text[:300])
                if len(samples) >= num_samples:
                    break
    except Exception:
        pass

    random.shuffle(samples)
    print(f"Loaded {len(samples)} text samples (simple)")
    return samples[:num_samples]


def generate_description(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 256,
    prompt_idx: int = None,
) -> str:
    """Generate a semantic description of the given text using the model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to describe
        max_new_tokens: Maximum tokens in the description
        prompt_idx: Which prompt template to use (random if None)

    Returns:
        The generated description string
    """
    if prompt_idx is None:
        prompt_idx = random.randint(0, len(DESCRIPTION_PROMPTS) - 1)

    prompt_template = DESCRIPTION_PROMPTS[prompt_idx]
    # Truncate text if too long
    text_truncated = text[:500]
    prompt = prompt_template.format(text=text_truncated)

    # Format as chat - disable Qwen3 thinking mode for direct descriptions
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

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

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

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    description = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip any remaining think tags if present
    if "<think>" in description:
        # Remove everything between <think> and </think>
        import re
        description = re.sub(r'<think>.*?</think>', '', description, flags=re.DOTALL).strip()
        # If still has unclosed think tag, take content after it
        if "<think>" in description:
            description = description.split("</think>")[-1].strip()

    return description


def generate_descriptions_batch(
    model,
    tokenizer,
    texts: list[str],
    max_new_tokens: int = 256,
    save_path: Optional[str] = None,
    save_every: int = 100,
) -> list[dict]:
    """Generate descriptions for a batch of texts.

    Returns list of dicts with 'text' and 'description' keys.
    """
    results = []
    existing = set()

    # Load existing results if resuming
    if save_path and os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                item = json.loads(line)
                results.append(item)
                existing.add(item["text"][:100])
        print(f"Resuming from {len(results)} existing descriptions")

    for i, text in enumerate(texts):
        if text[:100] in existing:
            continue

        try:
            prompt_idx = i % len(DESCRIPTION_PROMPTS)
            description = generate_description(
                model, tokenizer, text,
                max_new_tokens=max_new_tokens,
                prompt_idx=prompt_idx,
            )
            result = {
                "text": text,
                "description": description,
                "prompt_idx": prompt_idx,
                "idx": len(results),
            }
            results.append(result)

            if save_path and len(results) % save_every == 0:
                _save_jsonl(results, save_path)
                print(f"  Saved {len(results)} descriptions")

        except Exception as e:
            print(f"  Error generating description for text {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Generated {len(results)}/{len(texts)} descriptions")

    if save_path:
        _save_jsonl(results, save_path)

    return results


def _save_jsonl(data: list[dict], path: str):
    """Save a list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class AODataset(torch.utils.data.Dataset):
    """Dataset for Activation Oracle training.

    Each example from dataset.jsonl has:
    - text: source text
    - description: self-generated description (training target)
    - layer: which layer activations came from
    - activation_file: filename in activations/ dir (e.g. activations_0_L9.pt)
    - num_activations: number of activation vectors
    - positions: token positions where activations were collected

    The activation file is a tensor of shape (num_positions, hidden_size).
    """

    def __init__(
        self,
        activation_dir: str,
        descriptions_path: str,
        tokenizer,
        max_num_activations: int = 10,
        max_target_length: int = 256,
        source_layers: list[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_num_activations = max_num_activations
        self.max_target_length = max_target_length
        self.source_layers = source_layers or [9, 18, 27]
        self.activation_dir = activation_dir

        # Load and validate examples
        all_examples = load_jsonl(descriptions_path)
        self.examples = []
        for ex in all_examples:
            act_path = Path(activation_dir) / ex["activation_file"]
            if act_path.exists():
                self.examples.append(ex)

        print(f"AODataset: {len(self.examples)} valid examples "
              f"(of {len(all_examples)} total)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Load activation tensor
        act_path = Path(self.activation_dir) / ex["activation_file"]
        activations = torch.load(act_path, weights_only=True)

        # activations shape: (num_positions, hidden_size)
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)

        source_layer = ex["layer"]

        # Subsample activations if too many
        num_acts = min(activations.shape[0], self.max_num_activations)
        if activations.shape[0] > num_acts:
            indices = sorted(random.sample(range(activations.shape[0]), num_acts))
            activations = activations[indices]

        # Pick a random oracle question
        question = random.choice(ORACLE_QUESTIONS)

        # Build the oracle prompt with placeholder tokens
        placeholders = " ?" * num_acts
        oracle_text = f"Layer {source_layer}:{placeholders} {question}"

        # Target is the description
        target_text = ex["description"]

        return {
            "oracle_text": oracle_text,
            "target_text": target_text,
            "activations": activations,  # (num_acts, hidden_size)
            "source_layer": source_layer,
            "num_activations": num_acts,
        }
