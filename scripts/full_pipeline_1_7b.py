#!/usr/bin/env python3
"""
Full pipeline for Qwen3-1.7B: generate data + train on TPU.
Runs end-to-end without manual intervention.
"""

import json
import os
import sys
import time
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "data_1.7b"
CHECKPOINT_DIR = "checkpoints_1.7b"
SOURCE_LAYERS = [7, 14, 21]  # 25%, 50%, 75% of 28 layers

# Same 50 texts from simple_generate.py
TEXTS = [
    "The quantum computer achieved error correction using a new topological approach that stabilizes qubits at room temperature, a breakthrough that could revolutionize computing.",
    "Le chat est assis sur le tapis, regardant par la fenetre avec curiosite. Il attend patiemment le retour de son maitre qui est parti ce matin.",
    "In 2024, global renewable energy capacity exceeded fossil fuel capacity for the first time in recorded history, marking a pivotal moment in the energy transition.",
    "The patient presented with acute respiratory distress and bilateral infiltrates on chest X-ray. Initial oxygen saturation was 88% on room air.",
    "Mix flour, sugar, and butter until crumbly. Add eggs one at a time, beating well after each addition. Fold in chocolate chips gently.",
    "Der Baum im Garten tragt dieses Jahr besonders viele Apfel. Die Ernte wird gut sein, und wir werden genug Apfelkuchen backen konnen.",
    "Once upon a time, in a kingdom far away, there lived a wise old owl who counseled the king on matters of state and diplomacy.",
    "The GDP growth rate of 3.2% exceeded analysts' expectations of 2.8%, driving equity markets to new highs in the fourth quarter.",
    "She walked through the empty streets, the rain falling steadily on her umbrella as she headed home from the late shift at the hospital.",
    "According to Einstein's theory of general relativity, massive objects warp the fabric of spacetime, causing what we perceive as gravitational attraction.",
    "The board of directors approved the $5.2 billion merger with a 7-2 vote after months of intense negotiations between the two companies.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy captured by chlorophyll molecules in plant cells.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)  # Classic recursive implementation with exponential time complexity",
    "The Tokyo Olympics were postponed to 2021 due to the global pandemic that started in early 2020, affecting athletes worldwide.",
    "Machine learning models require large amounts of labeled training data and significant computational resources to achieve state-of-the-art performance.",
    "La biblioteca municipal abrira sus puertas el proximo lunes con una nueva coleccion de libros en espanol y una sala de lectura renovada.",
    "The ancient ruins of Machu Picchu sit high in the Andes mountains, a testament to the engineering prowess of the Inca civilization.",
    "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name HAVING COUNT(orders.id) > 5;",
    "Climate scientists warn that global temperatures could rise by 2 degrees Celsius above pre-industrial levels by 2050 without significant intervention.",
    "The neural network architecture consists of multiple transformer layers with self-attention mechanisms that allow the model to capture long-range dependencies.",
    "Two roads diverged in a yellow wood, and sorry I could not travel both and be one traveler, long I stood and looked down one as far as I could.",
    "The protein structure was determined using cryo-electron microscopy at 3.2 angstrom resolution, revealing a novel binding domain.",
    "Il ristorante italiano in centro citta offre una vasta selezione di piatti tradizionali, dalla pasta fatta in casa alle pizze cotte nel forno a legna.",
    "The central bank raised interest rates by 25 basis points, citing persistent inflationary pressures in the housing and services sectors.",
    "Hydrogen bonds between water molecules give water its unique properties, including high surface tension and its ability to act as a universal solvent.",
    "In the depths of the ocean, bioluminescent creatures create their own light, a phenomenon that scientists are still working to fully understand.",
    "The Supreme Court ruled 6-3 that the executive order exceeded the president's constitutional authority, marking a significant check on executive power.",
    "Mozart composed his first symphony at the age of eight, demonstrating a prodigious musical talent that would define the Classical era.",
    "The CRISPR-Cas9 gene editing technology has opened new possibilities for treating genetic diseases by precisely modifying DNA sequences.",
    "The microprocessor contains over 10 billion transistors on a chip smaller than a fingernail, each switching billions of times per second.",
    "Researchers found that regular meditation practice can reduce cortisol levels by up to 25%, contributing to improved mental health outcomes.",
    "The Higgs boson was experimentally confirmed at CERN in 2012, validating a key prediction of the Standard Model of particle physics.",
    "Fresh basil, ripe tomatoes, and mozzarella cheese combine to create the classic Italian Caprese salad, dressed simply with olive oil and balsamic vinegar.",
    "The spacecraft entered Mars orbit after a seven-month journey, preparing to deploy its rover to search for signs of ancient microbial life.",
    "Economic inequality has widened significantly over the past four decades, with the top 1% now holding more wealth than the bottom 50% combined.",
    "The coral reef ecosystem supports approximately 25% of all marine species despite covering less than 1% of the ocean floor.",
    "In computer science, a hash table provides O(1) average-case time complexity for insertions, deletions, and lookups, making it one of the most efficient data structures.",
    "The Renaissance period in Europe saw a revival of interest in classical Greek and Roman art, philosophy, and literature, fundamentally transforming Western culture.",
    "A balanced diet should include adequate protein, complex carbohydrates, healthy fats, vitamins, and minerals to support optimal body function.",
    "The electric vehicle market has grown exponentially, with global sales increasing by 35% year-over-year as battery costs continue to decline.",
    "Die Berliner Mauer fiel am 9. November 1989, ein Ereignis das das Ende des Kalten Krieges symbolisierte und Deutschland wiedervereinigte.",
    "The algorithm uses dynamic programming to solve the problem in O(n^2) time, storing intermediate results to avoid redundant computations.",
    "Deep beneath the Antarctic ice sheet, scientists discovered a subglacial lake teeming with microbial life, challenging assumptions about habitability.",
    "The violin concerto in D major by Beethoven is considered one of the greatest works in the violin repertoire, demanding both technical mastery and emotional depth.",
    "Blockchain technology enables decentralized, transparent record-keeping without the need for a central authority, with applications beyond cryptocurrency.",
    "The tropical rainforest canopy is so dense that less than 2% of sunlight reaches the forest floor, creating a unique and diverse ecosystem.",
    "L'intelligence artificielle transforme rapidement de nombreux secteurs, de la sante a l'education, en passant par les transports et la finance.",
    "The archaeological excavation uncovered pottery fragments dating back to 3000 BCE, providing evidence of early Bronze Age settlements in the region.",
    "Abstract expressionism emerged in New York in the 1940s, with artists like Pollock and de Kooning pushing the boundaries of traditional painting.",
]

DESCRIPTION_PROMPTS = [
    "Describe this text's semantic content (language, topic, grammar, style, meaning):\n\"{text}\"\n\nAnalysis:",
    "Analyze: what language, subject, writing style, and likely continuations?\n\"{text}\"\n\nAnalysis:",
    "Characterize this passage: language, domain, entities, structure, intent.\n\"{text}\"\n\nAnalysis:",
    "What information does this text convey? Cover language, topic, and style.\n\"{text}\"\n\nAnalysis:",
]


def generate_data(model, tokenizer, device, output_dir):
    """Generate descriptions + activations with Qwen3-1.7B."""
    from model import ActivationCollector

    os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

    collector = ActivationCollector(model, SOURCE_LAYERS)
    collector.register_hooks()

    dataset = []
    total = 0

    print(f"\n=== Generating data: {len(TEXTS)} texts, layers {SOURCE_LAYERS} ===")
    start = time.time()

    for i, text in enumerate(TEXTS):
        t0 = time.time()
        try:
            # Generate description
            template = DESCRIPTION_PROMPTS[i % len(DESCRIPTION_PROMPTS)]
            prompt = template.format(text=text[:300])
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

            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=96, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                try:
                    import torch_xla; torch_xla.sync()
                except (ImportError, NameError): pass

            new_ids = outputs[0][input_ids.shape[1]:]
            description = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            if "<think>" in description:
                import re
                description = re.sub(r'<think>.*?</think>', '', description, flags=re.DOTALL).strip()

            if len(description) < 15:
                description = f"This text discusses: {text[:200]}"

            # Collect activations
            text_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            text_ids = text_inputs["input_ids"].to(device)
            seq_len = text_ids.shape[1]

            candidates = []
            for pos in range(1, seq_len - 1):
                tok_str = tokenizer.decode([text_ids[0, pos].item()])
                if len(tok_str.strip()) > 2:
                    candidates.append(pos)
            if not candidates:
                candidates = list(range(1, seq_len - 1))
            positions = sorted(random.sample(candidates, min(3, len(candidates))))

            collector.activations = {}
            with torch.no_grad():
                model(input_ids=text_ids, attention_mask=text_inputs["attention_mask"].to(device))
                try:
                    import torch_xla; torch_xla.sync()
                except (ImportError, NameError): pass

            for layer_idx in SOURCE_LAYERS:
                if layer_idx not in collector.activations:
                    continue
                acts = collector.activations[layer_idx]
                pos_acts = acts[0, positions, :].cpu()
                act_file = f"act_{total}_L{layer_idx}.pt"
                torch.save(pos_acts, os.path.join(output_dir, "activations", act_file))
                dataset.append({
                    "idx": total, "text": text[:500], "description": description,
                    "layer": layer_idx, "positions": positions,
                    "num_activations": len(positions), "activation_file": act_file,
                    "seq_len": seq_len,
                })
                total += 1

            dt = time.time() - t0
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - start
                eta = (len(TEXTS) - i - 1) * elapsed / (i + 1) / 60
                print(f"  [{i+1}/{len(TEXTS)}] {dt:.1f}s | examples: {total} | ETA: {eta:.0f}min")

        except Exception as e:
            print(f"  ERROR text {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    collector.clear()

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\nData generation complete: {total} examples in {elapsed/60:.1f}min")
    return dataset_path


def train(model, tokenizer, dataset_path, output_dir, device):
    """Train the activation oracle with LoRA on TPU."""
    from peft import LoraConfig, get_peft_model, TaskType
    from data import AODataset

    data_dir = os.path.dirname(dataset_path)
    act_dir = os.path.join(data_dir, "activations")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = AODataset(
        activation_dir=act_dir, descriptions_path=dataset_path,
        tokenizer=tokenizer, source_layers=SOURCE_LAYERS,
    )

    # Import training function
    from train_tpu import train_ao
    model = train_ao(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        output_dir=output_dir, num_epochs=3, batch_size=2,
        learning_rate=2e-4, gradient_accumulation_steps=4,
        save_every=50, log_every=5, device=device,
        use_tpu=True,
    )
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "both"], default="both")
    parser.add_argument("--device", default="cpu", choices=["cpu", "xla"])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "xla":
        import torch_xla
        device = torch_xla.device()
        print(f"TPU device: {device}")
    else:
        device = torch.device("cpu")
        print(f"CPU device")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    if args.mode in ("generate", "both"):
        dataset_path = generate_data(model, tokenizer, device, OUTPUT_DIR)
        os.system(f"cd /home/smitop2/ao-self-distill && git add {OUTPUT_DIR}/dataset.jsonl && "
                  f"git commit -m 'Generated 1.7B training data' && git push origin master")
    else:
        dataset_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")

    if args.mode in ("train", "both"):
        print("\n\n=== TRAINING ===")
        model.train()
        train(model, tokenizer, dataset_path, CHECKPOINT_DIR, device)
        os.system(f"cd /home/smitop2/ao-self-distill && git add {CHECKPOINT_DIR}/training_log.jsonl && "
                  f"git commit -m 'Training complete (1.7B)' && git push origin master")

    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
