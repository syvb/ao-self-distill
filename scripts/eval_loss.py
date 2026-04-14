#!/usr/bin/env python3
"""
Loss-based evaluation on TPU - no autoregressive generation needed.

Compares:
1. Oracle with CORRECT activations (should have low loss)
2. Oracle with SHUFFLED activations (should have higher loss)
3. Oracle with ZERO activations (baseline)

If the oracle has learned to read activations, loss with correct activations
should be significantly lower than with shuffled/zero.
"""

import os
os.environ['XLA_USE_SPMD'] = '1'

import json, sys, random, torch, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints_8b_spmd/final/lora_weights.pt"
INJECTION_LAYER = 1


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
        batch, seq, hid = hidden.shape
        max_pos = max(max(p) for p in positions if p)
        if seq <= max_pos:
            return output
        vectors = state["vectors"].to(hidden.device, hidden.dtype)
        update = torch.zeros_like(hidden)
        for b in range(batch):
            for i, pos in enumerate(positions[b]):
                if pos < 0 or pos >= seq:
                    continue
                h_i = hidden[b, pos, :]
                v_i = vectors[b, i, :]
                h_norm = torch.norm(h_i)
                v_norm = torch.norm(v_i) + 1e-8
                update = update.index_put(
                    (torch.tensor([b], device=hidden.device),
                     torch.tensor([pos], device=hidden.device)),
                    h_norm.unsqueeze(0) * (v_i / v_norm).unsqueeze(0),
                    accumulate=True,
                )
        new_hidden = hidden + update
        return (new_hidden,) + rest if rest is not None else new_hidden
    return hook


def compute_loss(model, tokenizer, ex, inject_state, dev, placeholder_id,
                 condition="correct", shuffled_acts=None, zero_acts=False,
                 max_length=256):
    """Compute loss on a single example with given activation condition."""
    import os
    act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
    activations = torch.load(act_path, weights_only=True)
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    num_acts = activations.shape[0]

    if condition == "shuffled":
        activations = shuffled_acts[:num_acts]
    elif condition == "zero":
        activations = torch.zeros_like(activations)
    # else "correct" keeps original

    placeholders = " ?" * num_acts
    oracle_text = f"Layer {ex['layer']}:{placeholders} Describe the semantic content of this text."
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
        formatted, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc["attention_mask"].to(dev)

    ph_positions = (enc["input_ids"][0] == placeholder_id).nonzero(as_tuple=True)[0].tolist()
    if not ph_positions:
        return None
    num_ph = min(len(ph_positions), num_acts)
    ph_positions = ph_positions[:num_ph]
    acts_to_inject = activations[:num_ph].unsqueeze(0).to(dev)

    # Mask prompt
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

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        xm.mark_step()

    inject_state["active"] = False
    return out.loss.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--output", default="eval_results/loss_results.jsonl")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    mesh = xs.Mesh(list(range(num_devices)), (num_devices,), ('fsdp',))
    xs.set_global_mesh(mesh)
    dev = xm.xla_device()
    print(f"Device: {dev} ({num_devices} chips)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = model.to(dev)

    for name, param in model.named_parameters():
        if param.dim() >= 1:
            spec = [None] * param.dim()
            spec[0] = 'fsdp'
            try:
                xs.mark_sharding(param, mesh, tuple(spec))
            except Exception:
                pass

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", lora_dropout=0.0,
    )
    model = get_peft_model(model, lora)

    print(f"Loading LoRA from {args.checkpoint}...")
    lora_state = torch.load(args.checkpoint, weights_only=True)
    current_state = dict(model.named_parameters())
    loaded = 0
    for name, w in lora_state.items():
        if name in current_state:
            current_state[name].data.copy_(w.to(dev))
            loaded += 1
    print(f"Loaded {loaded}/{len(lora_state)} LoRA weights")
    model.eval()

    base = model.base_model.model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    layers = base.layers
    inject_state = {"active": False, "positions": None, "vectors": None}
    hook = layers[INJECTION_LAYER].register_forward_hook(injection_hook_factory(inject_state))

    examples = []
    with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
        for line in f:
            ex = json.loads(line)
            act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
            if os.path.exists(act_path):
                examples.append(ex)
    random.shuffle(examples)
    eval_examples = examples[:args.num_eval]

    placeholder_id = tokenizer.encode(" ?", add_special_tokens=False)[-1]

    # Build shuffled activation pool - use activations from DIFFERENT examples
    shuffled_pool = []
    for ex in eval_examples:
        act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
        a = torch.load(act_path, weights_only=True)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        shuffled_pool.append(a)

    results = []
    print(f"\n=== Loss evaluation: {len(eval_examples)} examples x 3 conditions ===")
    start = time.time()

    for i, ex in enumerate(eval_examples):
        t0 = time.time()
        try:
            # Condition 1: Correct activations
            correct_loss = compute_loss(model, tokenizer, ex, inject_state, dev, placeholder_id,
                                         condition="correct")

            # Condition 2: Shuffled (use activations from a DIFFERENT example)
            shuffle_idx = (i + 1) % len(shuffled_pool)
            shuffled_loss = compute_loss(model, tokenizer, ex, inject_state, dev, placeholder_id,
                                          condition="shuffled",
                                          shuffled_acts=shuffled_pool[shuffle_idx])

            # Condition 3: Zero activations
            zero_loss = compute_loss(model, tokenizer, ex, inject_state, dev, placeholder_id,
                                      condition="zero")

            dt = time.time() - t0
            r = {
                "idx": i, "layer": ex["layer"],
                "correct_loss": round(correct_loss, 4),
                "shuffled_loss": round(shuffled_loss, 4),
                "zero_loss": round(zero_loss, 4),
                "time_sec": round(dt, 1),
            }
            results.append(r)
            print(f"Ex {i} L{ex['layer']} ({dt:.1f}s): correct={correct_loss:.4f} "
                  f"shuffled={shuffled_loss:.4f} zero={zero_loss:.4f}")

            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        except Exception as e:
            print(f"Error ex {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    hook.remove()
    elapsed = time.time() - start

    # Aggregate stats
    if results:
        correct_avg = sum(r["correct_loss"] for r in results) / len(results)
        shuffled_avg = sum(r["shuffled_loss"] for r in results) / len(results)
        zero_avg = sum(r["zero_loss"] for r in results) / len(results)

        print(f"\n=== Complete: {len(results)} examples in {elapsed/60:.1f}min ===")
        print(f"Average loss (correct activations):  {correct_avg:.4f}")
        print(f"Average loss (shuffled activations): {shuffled_avg:.4f}")
        print(f"Average loss (zero activations):     {zero_avg:.4f}")
        print()
        print(f"Gap correct vs shuffled: {shuffled_avg - correct_avg:.4f}")
        print(f"Gap correct vs zero:     {zero_avg - correct_avg:.4f}")
        print()
        if correct_avg < shuffled_avg and correct_avg < zero_avg:
            print("✓ Oracle uses activations: correct activations give LOWEST loss")
        else:
            print("✗ No clear signal from activations")


if __name__ == "__main__":
    main()
