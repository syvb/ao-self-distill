#!/usr/bin/env python3
"""
Train Qwen3-8B activation oracle using SPMD sharding across 4 TPU v5lite chips.
"""

import os
os.environ['XLA_USE_SPMD'] = '1'

import json, sys, time, random, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "data"  # Original 8B data
CHECKPOINT_DIR = "checkpoints_8b_spmd"
INJECTION_LAYER = 1

ORACLE_QUESTIONS = [
    "Describe the semantic content of this text.",
    "What is this text about?",
    "Describe what is happening in this text.",
    "What language, topic, and content does this text contain?",
    "Characterize the content encoded in these activations.",
]


def injection_hook_factory(state):
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

        # Build an additive update tensor, then add once (no in-place modification)
        batch, seq, hid = hidden.shape
        update = torch.zeros_like(hidden)
        for b in range(batch):
            for i, pos in enumerate(positions[b]):
                if pos < 0:
                    continue
                h_i = hidden[b, pos, :]
                v_i = vectors[b, i, :]
                h_norm = torch.norm(h_i)
                v_norm = torch.norm(v_i) + 1e-8
                # norm-matched: add h_norm * (v_i / v_norm) at position pos
                update = update.index_put(
                    (torch.tensor([b], device=hidden.device),
                     torch.tensor([pos], device=hidden.device)),
                    h_norm.unsqueeze(0) * (v_i / v_norm).unsqueeze(0),
                    accumulate=True,
                )
        new_hidden = hidden + update
        return (new_hidden,) + rest if rest is not None else new_hidden
    return hook


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # SPMD setup
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Devices: {num_devices}")
    mesh = xs.Mesh(list(range(num_devices)), (num_devices,), ('fsdp',))
    xs.set_global_mesh(mesh)

    dev = xm.xla_device()
    print(f"Device: {dev}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = model.to(dev)
    print("Model on TPU, sharding...")

    # Shard all parameters across fsdp dim
    for name, param in model.named_parameters():
        if param.dim() >= 1:
            spec = [None] * param.dim()
            spec[0] = 'fsdp'
            try:
                xs.mark_sharding(param, mesh, tuple(spec))
            except Exception:
                pass

    # LoRA
    print("Adding LoRA...")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", lora_dropout=0.05,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    # Get placeholder token
    ph_ids = tokenizer.encode(" ?", add_special_tokens=False)
    placeholder_id = ph_ids[-1]
    print(f"Placeholder token: {placeholder_id}")

    # Register injection hook on the base model's layer
    base = model.base_model.model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    layers = base.layers
    print(f"Base has {len(layers)} layers, injecting at layer {INJECTION_LAYER}")

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

    log_file = open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "w")

    print(f"\n=== Training: {args.max_steps} steps ===")
    start = time.time()
    losses = []
    step_count = 0

    for epoch in range(100):  # many epochs, break on max_steps
        random.shuffle(examples)
        for ex in examples:
            if step_count >= args.max_steps:
                break

            try:
                # Load activation
                act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
                activations = torch.load(act_path, weights_only=True)
                if activations.dim() == 1:
                    activations = activations.unsqueeze(0)
                num_acts = activations.shape[0]

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
                    formatted, truncation=True, max_length=args.max_length,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(dev)
                attention_mask = enc["attention_mask"].to(dev)

                # Find placeholder positions (on CPU to avoid sync)
                ph_positions = (enc["input_ids"][0] == placeholder_id).nonzero(as_tuple=True)[0].tolist()
                if len(ph_positions) == 0:
                    continue

                num_ph = min(len(ph_positions), num_acts)
                ph_positions = ph_positions[:num_ph]
                acts_to_inject = activations[:num_ph].unsqueeze(0).to(dev)

                # Labels: mask prompt
                labels = enc["input_ids"].clone()
                target_tokens = tokenizer.encode(target[:30], add_special_tokens=False)
                prompt_end = labels.shape[1]
                if len(target_tokens) >= 3:
                    input_list = enc["input_ids"][0].tolist()
                    for i in range(len(input_list) - 3):
                        if input_list[i:i+3] == target_tokens[:3]:
                            prompt_end = i
                            break
                labels[0, :prompt_end] = -100
                labels[enc["attention_mask"] == 0] = -100
                labels = labels.to(dev)

                inject_state["active"] = True
                inject_state["positions"] = [ph_positions]
                inject_state["vectors"] = acts_to_inject

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
                xm.mark_step()

                inject_state["active"] = False

                loss_val = loss.item()
                losses.append(loss_val)
                step_count += 1

                if step_count % args.log_every == 0:
                    avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
                    elapsed = time.time() - start
                    rate = step_count / elapsed * 60
                    entry = {
                        "step": step_count, "loss": round(avg, 4),
                        "elapsed_min": round(elapsed / 60, 1),
                        "rate_per_min": round(rate, 2),
                    }
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()
                    print(f"Step {step_count}/{args.max_steps} | Loss: {avg:.4f} | "
                          f"{rate:.1f} steps/min | {elapsed/60:.1f}min")

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

    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    elapsed = time.time() - start
    print(f"\n=== Training complete: {step_count} steps in {elapsed/60:.1f}min ===")


if __name__ == "__main__":
    main()
