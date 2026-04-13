#!/usr/bin/env python3
"""
Evaluate the trained activation oracle on TPU.
Loads the LoRA weights, injects activations from held-out examples,
and generates descriptions.
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

        # During autoregressive generation, the hidden state may only have seq_len=1.
        # Only inject when we have the full prompt (seq_len > max placeholder position).
        batch, seq, hid = hidden.shape
        max_pos = max(max(p) for p in positions if p)
        if seq <= max_pos:
            # Not the prompt pass - skip injection
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--output", default="eval_results/results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # SPMD setup
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

    # Apply LoRA
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", lora_dropout=0.0,
    )
    model = get_peft_model(model, lora)

    # Load LoRA weights
    print(f"Loading LoRA from {args.checkpoint}...")
    lora_state = torch.load(args.checkpoint, weights_only=True)
    # Load into model
    current_state = dict(model.named_parameters())
    loaded = 0
    for name, w in lora_state.items():
        if name in current_state:
            current_state[name].data.copy_(w.to(dev))
            loaded += 1
    print(f"Loaded {loaded}/{len(lora_state)} LoRA weights")

    model.eval()

    # Injection hook
    base = model.base_model.model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    layers = base.layers
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
    eval_examples = examples[:args.num_eval]

    placeholder_id = tokenizer.encode(" ?", add_special_tokens=False)[-1]

    results = []
    print(f"\n=== Evaluating {len(eval_examples)} examples ===")
    start = time.time()

    for i, ex in enumerate(eval_examples):
        try:
            act_path = os.path.join(DATA_DIR, "activations", ex["activation_file"])
            activations = torch.load(act_path, weights_only=True)
            if activations.dim() == 1:
                activations = activations.unsqueeze(0)
            num_acts = activations.shape[0]

            placeholders = " ?" * num_acts
            oracle_text = f"Layer {ex['layer']}:{placeholders} Describe the semantic content of this text."

            messages = [{"role": "user", "content": oracle_text}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            enc = tokenizer(formatted, return_tensors="pt", max_length=128, truncation=True)
            input_ids = enc["input_ids"].to(dev)
            attention_mask = enc["attention_mask"].to(dev)

            ph_positions = (enc["input_ids"][0] == placeholder_id).nonzero(as_tuple=True)[0].tolist()
            if not ph_positions:
                continue
            num_ph = min(len(ph_positions), num_acts)
            ph_positions = ph_positions[:num_ph]
            acts_to_inject = activations[:num_ph].unsqueeze(0).to(dev)

            inject_state["active"] = True
            inject_state["positions"] = [ph_positions]
            inject_state["vectors"] = acts_to_inject

            with torch.no_grad():
                out = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                xm.mark_step()

            inject_state["active"] = False

            new_ids = out[0][input_ids.shape[1]:].cpu()
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            result = {
                "idx": i,
                "text": ex["text"][:200],
                "ground_truth": ex["description"][:300],
                "generated": generated[:300],
                "layer": ex["layer"],
            }
            results.append(result)

            if i < 5:
                print(f"\n--- Example {i} (layer {ex['layer']}) ---")
                print(f"Text: {ex['text'][:100]}...")
                print(f"GT:  {ex['description'][:120]}...")
                print(f"GEN: {generated[:120]}...")
            elif i % 5 == 0:
                print(f"[{i}/{len(eval_examples)}]")

        except Exception as e:
            inject_state["active"] = False
            print(f"Error example {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    hook.remove()

    # Save results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\n=== Evaluation complete: {len(results)} examples in {elapsed/60:.1f}min ===")
    print(f"Results: {args.output}")

    if not results:
        print("No results to analyze")
        return

    # Simple stats
    total_len_gen = sum(len(r["generated"]) for r in results)
    total_len_gt = sum(len(r["ground_truth"]) for r in results)
    print(f"Avg generated length: {total_len_gen/len(results):.0f} chars")
    print(f"Avg ground truth length: {total_len_gt/len(results):.0f} chars")

    # Keyword overlap
    overlaps = []
    for r in results:
        gt_words = set(r["ground_truth"].lower().split())
        gen_words = set(r["generated"].lower().split())
        stop = {"the","a","an","is","are","was","were","in","on","at","to","for","of","with","and","or","but","this","that"}
        gt = gt_words - stop
        gen = gen_words - stop
        if gt:
            overlaps.append(len(gt & gen) / len(gt))
    if overlaps:
        print(f"Avg keyword overlap: {sum(overlaps)/len(overlaps):.3f}")


if __name__ == "__main__":
    main()
