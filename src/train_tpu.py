"""
TPU-optimized training loop for the Self-Distillation Activation Oracle.

Uses torch_xla for TPU acceleration. Handles:
- LoRA fine-tuning with activation injection
- TPU-compatible forward/backward passes
- Checkpointing and logging
"""

import os
import json
import time
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import (
    INJECTION_LAYER,
    PLACEHOLDER_TOKEN,
    get_placeholder_token_id,
    HIDDEN_SIZE,
)
from data import AODataset, load_jsonl, ORACLE_QUESTIONS


def setup_lora(model, rank: int = 64, alpha: int = 128, dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class InjectionHookManager:
    """Manages activation injection hooks for training."""

    def __init__(self, model, injection_layer: int = INJECTION_LAYER):
        self.model = model
        self.injection_layer = injection_layer
        self.positions = None
        self.vectors = None
        self.active = False
        self._hook = None

    def _hook_fn(self, module, input, output):
        if not self.active or self.positions is None:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        vectors = self.vectors.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Norm-matched additive steering for each example in the batch
        batch_size = hidden_states.shape[0]
        for b in range(batch_size):
            for i, pos in enumerate(self.positions[b]):
                if pos < 0:  # padding sentinel
                    continue
                h_i = hidden_states[b, pos, :]
                v_i = vectors[b, i, :]

                h_norm = torch.norm(h_i)
                v_norm = torch.norm(v_i)

                if v_norm > 1e-8:
                    hidden_states[b, pos, :] = h_i + h_norm * (v_i / v_norm)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def register(self):
        """Register the injection hook."""
        # Access layers through PEFT wrapper
        base_model = self.model
        while hasattr(base_model, 'base_model'):
            base_model = base_model.base_model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        elif hasattr(base_model, 'layers'):
            layers = base_model.layers
        else:
            raise RuntimeError("Could not find model layers for hook registration")

        self._hook = layers[self.injection_layer].register_forward_hook(self._hook_fn)
        return self

    def set(self, positions, vectors):
        """Set injection data and activate."""
        self.positions = positions
        self.vectors = vectors
        self.active = True

    def deactivate(self):
        """Deactivate injection."""
        self.active = False

    def remove(self):
        """Remove the hook."""
        if self._hook:
            self._hook.remove()
            self._hook = None


def collate_ao_batch(batch, tokenizer, max_length: int = 512):
    """Collate a batch of AO training examples."""
    placeholder_id = get_placeholder_token_id(tokenizer)

    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    all_placeholder_positions = []
    all_activations = []

    for item in batch:
        oracle_text = item["oracle_text"]
        target_text = item["target_text"]
        activations = item["activations"]

        messages = [
            {"role": "user", "content": oracle_text},
            {"role": "assistant", "content": target_text},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

        tokens = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]

        # Find placeholder positions
        ph_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()

        # Create labels: mask the prompt part
        labels = input_ids.clone()
        if ph_positions:
            target_tokens = tokenizer.encode(target_text[:50], add_special_tokens=False)
            prompt_end = max(ph_positions) + 20
            if len(target_tokens) > 3:
                for start_pos in range(len(input_ids) - len(target_tokens)):
                    if input_ids[start_pos:start_pos+3].tolist() == target_tokens[:3]:
                        prompt_end = start_pos
                        break
            labels[:prompt_end] = -100

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_mask)
        all_placeholder_positions.append(ph_positions)
        all_activations.append(activations)

    # Pad to same length
    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or 0

    padded_input_ids = []
    padded_labels = []
    padded_attention_masks = []

    for input_ids, labels, attn_mask in zip(all_input_ids, all_labels, all_attention_masks):
        pad_len = max_len - input_ids.shape[0]
        padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_id)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100)]))
        padded_attention_masks.append(torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)]))

    # Pad placeholder positions
    max_ph = max(len(p) for p in all_placeholder_positions) if all_placeholder_positions else 1
    max_ph = max(max_ph, 1)
    padded_positions = []
    for positions in all_placeholder_positions:
        padded_positions.append(positions + [-1] * (max_ph - len(positions)))

    # Pad activations
    padded_activations = []
    for acts, positions in zip(all_activations, all_placeholder_positions):
        num_acts = len(positions)
        if acts.shape[0] < max_ph:
            pad = torch.zeros(max_ph - acts.shape[0], acts.shape[1])
            acts = torch.cat([acts, pad], dim=0)
        padded_activations.append(acts[:max_ph])

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_attention_masks),
        "placeholder_positions": padded_positions,
        "activations": torch.stack(padded_activations),
    }


def train_ao(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    save_every: int = 200,
    log_every: int = 10,
    max_grad_norm: float = 1.0,
    device=None,
    use_tpu: bool = False,
):
    """Main training loop for the Activation Oracle."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up injection hook
    hook_mgr = InjectionHookManager(model, INJECTION_LAYER)
    hook_mgr.register()

    # DataLoader
    def collate_fn(batch):
        return collate_ao_batch(batch, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    # Simple linear warmup + decay schedule
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training log
    log_path = os.path.join(output_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    model.train()
    global_step = 0
    running_loss = 0
    running_count = 0

    print(f"\n{'='*60}")
    print(f"Training Activation Oracle")
    print(f"{'='*60}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} x {gradient_accumulation_steps} = "
          f"{batch_size * gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  TPU: {use_tpu}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activations = batch["activations"]
            positions = batch["placeholder_positions"]

            # Set up injection
            hook_mgr.set(positions, activations)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            loss_val = loss.item() * gradient_accumulation_steps
            running_loss += loss_val
            running_count += 1
            epoch_loss += loss_val
            epoch_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if use_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / running_count
                    running_loss = 0
                    running_count = 0
                    lr = scheduler.get_last_lr()[0]
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(avg_loss, 4),
                        "lr": round(lr, 8),
                        "time": time.time(),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    print(f"  Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                if global_step % save_every == 0:
                    save_dir = os.path.join(output_dir, f"step_{global_step}")
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"  Saved checkpoint: {save_dir}")

            hook_mgr.deactivate()

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Avg Loss: {avg_epoch_loss:.4f} | "
              f"Time: {epoch_time/60:.1f}min")

        # Save epoch checkpoint
        save_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"  Saved epoch checkpoint: {save_dir}")

    hook_mgr.remove()
    log_file.close()

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete! Final model: {final_dir}")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Self-Distillation AO")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "xla", "cuda"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    use_tpu = args.device == "xla"

    # Load model
    print(f"Loading Qwen3-8B on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_tpu:
        import torch_xla
        device = torch_xla.device()
        model = model.to(device)
    else:
        device = torch.device(args.device)
        model = model.to(device)

    # Apply LoRA
    model = setup_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # Load dataset
    dataset_path = os.path.join(args.data_dir, "dataset.jsonl")
    activations_dir = os.path.join(args.data_dir, "activations")

    dataset = AODataset(
        activation_dir=activations_dir,
        descriptions_path=dataset_path,
        tokenizer=tokenizer,
        source_layers=[9, 18, 27],
    )

    print(f"Dataset: {len(dataset)} examples")

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
        device=device,
        use_tpu=use_tpu,
    )


if __name__ == "__main__":
    main()
