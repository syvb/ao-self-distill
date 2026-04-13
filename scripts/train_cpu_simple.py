#!/usr/bin/env python3
"""
Simple CPU training of activation oracle with Qwen3-1.7B.
Minimal dependencies, no HF Trainer, no TPU complications.
"""

import json, os, sys, time, random, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_DIR = "data_1.7b"
CHECKPOINT_DIR = "checkpoints_1.7b"
INJECTION_LAYER = 1  # After layer 1, as in paper
PLACEHOLDER_TOKEN = " ?"

ORACLE_QUESTIONS = [
    "Describe the semantic content of this text.",
    "What is this text about?",
    "Describe what is happening in this text.",
    "What language, topic, and content does this text contain?",
    "Characterize the content encoded in these activations.",
]


def injection_hook_factory(state):
    """Create a hook that injects activations at placeholder positions."""
    def hook(module, input, output):
        if not state.get("active"):
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        positions = state["positions"]
        vectors = state["vectors"].to(hidden.device, hidden.dtype)

        for b in range(hidden.shape[0]):
            for i, pos in enumerate(positions[b]):
                if pos < 0:
                    continue
                h_i = hidden[b, pos, :]
                v_i = vectors[b, i, :]
                h_norm = torch.norm(h_i)
                v_norm = torch.norm(v_i)
                if v_norm > 1e-8:
                    hidden[b, pos, :] = h_i + h_norm * (v_i / v_norm)

        return (hidden,) + rest if rest is not None else hidden
    return hook


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    )

    # LoRA
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", lora_dropout=0.05,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    # Get placeholder token id
    ph_ids = tokenizer.encode(" ?", add_special_tokens=False)
    placeholder_id = ph_ids[-1]
    print(f"Placeholder token ID: {placeholder_id}")

    # Register injection hook on base model's layer
    base = model.base_model.model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    layers = base.layers
    print(f"Model has {len(layers)} layers, injecting after layer {INJECTION_LAYER}")

    inject_state = {"active": False, "positions": None, "vectors": None}
    hook = layers[INJECTION_LAYER].register_forward_hook(injection_hook_factory(inject_state))

    # Load examples
    examples = []
    with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
        for line in f:
            ex = json.loads(line)
            act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
            if os.path.exists(act_path):
                examples.append(ex)
    random.shuffle(examples)
    print(f"Loaded {len(examples)} training examples")

    # Optimizer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Training log
    log_file = open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "w")

    print(f"\n=== Training: {args.max_steps} steps ===")
    start = time.time()
    total_loss = 0.0
    step_count = 0

    for epoch in range(args.epochs):
        random.shuffle(examples)
        for ex in examples:
            if step_count >= args.max_steps:
                break

            try:
                # Load activation
                act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
                activations = torch.load(act_path, weights_only=True)  # (num_pos, hidden)
                if activations.dim() == 1:
                    activations = activations.unsqueeze(0)
                num_acts = activations.shape[0]

                # Build prompt with placeholder tokens
                question = random.choice(ORACLE_QUESTIONS)
                placeholders = " ?" * num_acts
                oracle_text = f"Layer {ex['layer']}:{placeholders} {question}"
                target = ex["description"]

                messages = [
                    {"role": "user", "content": oracle_text},
                    {"role": "assistant", "content": target},
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

                enc = tokenizer(
                    formatted, truncation=True, max_length=384,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                # Find placeholder positions
                ph_positions = (input_ids[0] == placeholder_id).nonzero(as_tuple=True)[0].tolist()
                if len(ph_positions) == 0:
                    continue

                # Align activations to placeholder positions
                num_ph = min(len(ph_positions), num_acts)
                ph_positions = ph_positions[:num_ph]
                acts_to_inject = activations[:num_ph].unsqueeze(0)  # (1, num_ph, hidden)
                positions_batch = [ph_positions]  # batch of 1

                # Create labels (mask prompt, keep target)
                labels = input_ids.clone()
                # Mask prompt tokens - find assistant section
                target_tokens = tokenizer.encode(target[:30], add_special_tokens=False)
                prompt_end = len(input_ids[0])
                if len(target_tokens) >= 3:
                    for i in range(len(input_ids[0]) - 3):
                        if input_ids[0, i:i+3].tolist() == target_tokens[:3]:
                            prompt_end = i
                            break
                labels[0, :prompt_end] = -100
                labels[attention_mask == 0] = -100

                # Set injection
                inject_state["active"] = True
                inject_state["positions"] = positions_batch
                inject_state["vectors"] = acts_to_inject

                # Forward + backward
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    inject_state["active"] = False
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()

                inject_state["active"] = False

                loss_val = loss.item()
                total_loss += loss_val
                step_count += 1

                if step_count % args.log_every == 0:
                    avg = total_loss / args.log_every
                    total_loss = 0.0
                    elapsed = time.time() - start
                    rate = step_count / elapsed
                    log_entry = {
                        "step": step_count, "epoch": epoch,
                        "loss": round(avg, 4),
                        "elapsed_min": round(elapsed / 60, 1),
                        "rate_per_min": round(rate * 60, 2),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    print(f"Step {step_count}/{args.max_steps} | "
                          f"Loss: {avg:.4f} | {rate*60:.1f} steps/min | "
                          f"elapsed: {elapsed/60:.1f}min")

                if step_count % args.save_every == 0:
                    save_dir = os.path.join(CHECKPOINT_DIR, f"step_{step_count}")
                    model.save_pretrained(save_dir)
                    print(f"  Saved {save_dir}")

            except Exception as e:
                inject_state["active"] = False
                print(f"Error: {e}")
                import traceback; traceback.print_exc()
                continue

        if step_count >= args.max_steps:
            break

    hook.remove()
    log_file.close()

    # Save final
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n=== Training complete: {step_count} steps in {(time.time()-start)/60:.1f}min ===")
    print(f"Final model: {final_dir}")


if __name__ == "__main__":
    main()
