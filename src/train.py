"""
Training loop for the Self-Supervised Activation Oracle.

Handles:
- LoRA setup on the model
- Custom forward pass with activation injection at layer 2
- Training on (activation, description) pairs
- Checkpointing and logging
"""

import os
import json
import time
import random
import torch
import torch.nn as nn
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_cosine_schedule_with_warmup

from .model import (
    INJECTION_LAYER,
    PLACEHOLDER_TOKEN,
    get_placeholder_token_id,
    HIDDEN_SIZE,
)


def setup_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def create_injection_hook(injection_layer_idx: int):
    """Create a hook factory for activation injection.

    Returns a hook function and a container to set injection data.
    """
    container = {"positions": None, "vectors": None, "active": False}

    def hook_fn(module, input, output):
        if not container["active"] or container["positions"] is None:
            return output

        positions = container["positions"]
        vectors = container["vectors"]

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        vectors = vectors.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Norm-matched additive steering for each example in the batch
        batch_size = hidden_states.shape[0]
        for b in range(batch_size):
            for i, pos in enumerate(positions[b]):
                if pos < 0:  # padding sentinel
                    continue
                h_i = hidden_states[b, pos, :]  # (hidden_size,)
                v_i = vectors[b, i, :]  # (hidden_size,)

                h_norm = torch.norm(h_i)
                v_norm = torch.norm(v_i)

                if v_norm > 1e-8:
                    v_normalized = v_i / v_norm
                    hidden_states[b, pos, :] = h_i + h_norm * v_normalized

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    return hook_fn, container


def collate_ao_batch(batch, tokenizer, max_length: int = 512):
    """Collate a batch of AO training examples.

    Each example has:
    - oracle_text: The oracle prompt with placeholder tokens
    - target_text: The description to generate
    - activations: (num_acts, hidden_size) tensor
    - source_layer: int
    - num_activations: int

    Returns tokenized inputs, labels, and injection metadata.
    """
    placeholder_id = get_placeholder_token_id(tokenizer)

    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    all_placeholder_positions = []  # per-example list of positions
    all_activations = []

    for item in batch:
        oracle_text = item["oracle_text"]
        target_text = item["target_text"]
        activations = item["activations"]

        # Build the full sequence: [oracle_prompt] [target_description] [eos]
        # The oracle prompt is input, the description is the target
        messages = [
            {"role": "user", "content": oracle_text},
            {"role": "assistant", "content": target_text},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
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

        # Create labels: mask the prompt part (only train on description)
        labels = input_ids.clone()
        # Find where the assistant response starts
        # Mask everything up to and including the assistant header
        # We'll use a simple heuristic: find the position after the last placeholder
        if ph_positions:
            # Mask everything before the target (description) starts
            # Find the end of the oracle question in the token sequence
            # Simple approach: mask up to the assistant content start
            assistant_marker = tokenizer.encode("assistant", add_special_tokens=False)
            # Find the last occurrence of a reasonable marker
            prompt_end = max(ph_positions) + 20  # rough estimate
            # More robust: mask everything before where the target starts
            target_tokens = tokenizer.encode(target_text[:50], add_special_tokens=False)
            if len(target_tokens) > 3:
                # Find where target_tokens start in input_ids
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
    padded_input_ids = []
    padded_labels = []
    padded_attention_masks = []

    for input_ids, labels, attn_mask in zip(all_input_ids, all_labels, all_attention_masks):
        pad_len = max_len - input_ids.shape[0]
        padded_input_ids.append(
            torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id or 0)])
        )
        padded_labels.append(
            torch.cat([labels, torch.full((pad_len,), -100)])
        )
        padded_attention_masks.append(
            torch.cat([attn_mask, torch.zeros(pad_len)])
        )

    # Pad placeholder positions (use -1 as sentinel)
    max_ph = max(len(p) for p in all_placeholder_positions)
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
        "attention_mask": torch.stack(padded_attention_masks).long(),
        "placeholder_positions": padded_positions,  # list of lists
        "activations": torch.stack(padded_activations),  # (batch, max_ph, hidden_size)
    }


def train_ao(
    model,
    tokenizer,
    train_dataset,
    val_dataset=None,
    output_dir: str = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    save_every: int = 500,
    log_every: int = 10,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
):
    """Main training loop for the Activation Oracle.

    Args:
        model: The model with LoRA adapters applied
        tokenizer: The tokenizer
        train_dataset: AODataset for training
        val_dataset: Optional AODataset for validation
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        save_every: Save checkpoint every N steps
        log_every: Log every N steps
        max_grad_norm: Max gradient norm for clipping
        device: Device to train on
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up the injection hook
    # Access the base model's layers through PEFT wrapper
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(base_model, "model"):
        layers = base_model.model.layers
    else:
        layers = base_model.layers

    hook_fn, hook_container = create_injection_hook(INJECTION_LAYER)
    hook_handle = layers[INJECTION_LAYER].register_forward_hook(hook_fn)

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

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training log
    log_path = os.path.join(output_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    model.train()
    global_step = 0
    total_loss = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activations = batch["activations"]  # keep on CPU until hook
            positions = batch["placeholder_positions"]

            # Set up injection
            hook_container["positions"] = positions
            hook_container["vectors"] = activations
            hook_container["active"] = True

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = total_loss / (log_every * gradient_accumulation_steps)
                    total_loss = 0
                    lr = scheduler.get_last_lr()[0]
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(avg_loss, 4),
                        "lr": lr,
                        "time": time.time(),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                if global_step % save_every == 0:
                    save_dir = os.path.join(output_dir, f"step_{global_step}")
                    model.save_pretrained(save_dir)
                    print(f"  Saved checkpoint to {save_dir}")

            # Disable injection for cleanliness
            hook_container["active"] = False

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_epoch_loss:.4f}")

        # Save end-of-epoch checkpoint
        save_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_dir)
        print(f"Saved epoch checkpoint to {save_dir}")

    # Cleanup
    hook_handle.remove()
    log_file.close()

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"Training complete. Final model saved to {final_dir}")

    return model
