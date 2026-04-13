#!/usr/bin/env python3
"""
Training with HuggingFace Trainer + FSDP on TPU.
Falls back to CPU with optimizations if TPU fails.
"""

import json
import os
import sys
import random
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


class AOTrainDataset(Dataset):
    """Simplified AO dataset for HuggingFace Trainer."""

    ORACLE_QUESTIONS = [
        "Describe the semantic content of this text.",
        "What is this text about? Describe it in detail.",
        "Analyze the content represented by these activations.",
        "What language, topic, and meaning are encoded here?",
        "Describe what information is contained in these activations.",
    ]

    def __init__(self, data_dir, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = data_dir

        dataset_path = os.path.join(data_dir, "dataset.jsonl")
        act_dir = os.path.join(data_dir, "activations")

        self.examples = []
        with open(dataset_path) as f:
            for line in f:
                ex = json.loads(line)
                act_path = os.path.join(act_dir, ex["activation_file"])
                if os.path.exists(act_path):
                    self.examples.append(ex)

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = random.choice(self.ORACLE_QUESTIONS)
        layer = ex["layer"]
        num_acts = ex.get("num_activations", 1)

        # Build oracle prompt (without activation injection for now -
        # we'll train on text description pairs first as a baseline)
        placeholders = " ?" * num_acts
        oracle_text = f"Layer {layer}:{placeholders} {question}"
        target = ex["description"]

        messages = [
            {"role": "user", "content": oracle_text},
            {"role": "assistant", "content": target},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Mask prompt tokens from loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Find assistant start and mask prompt
        target_tokens = self.tokenizer.encode(target[:50], add_special_tokens=False)
        if len(target_tokens) > 3:
            for i in range(len(input_ids) - 3):
                if input_ids[i:i+3].tolist() == target_tokens[:3]:
                    labels[:i] = -100
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # Small LoRA config to minimize memory
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = AOTrainDataset(args.data_dir, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=args.bf16,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()
