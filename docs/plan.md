# Self-Supervised Activation Oracle via Self-Distillation

## Overview

This project extends the Activation Oracle (AO) framework from Karvonen et al. (2025) with two key innovations:

1. **Self-supervised semantic description training**: Instead of relying on labeled classification tasks or curated QA datasets, we train the AO using self-generated semantic descriptions. The model generates rich natural language descriptions of text passages, then learns to reproduce those descriptions from activations alone.

2. **Latent reasoning interpretation**: We extend the AO to interpret latent reasoning tokens — continuous hidden states produced by Coconut-style latent reasoning — enabling the AO to describe what a model is "thinking" during non-verbal reasoning.

## Background

### Activation Oracles (Karvonen et al. 2025)

Activation Oracles are LLMs fine-tuned to accept other LLMs' internal activations as input and answer questions about them in natural language. Key design elements:

- **Injection mechanism**: After the 2nd transformer layer, modify residual stream activations at placeholder token positions using norm-matched additive steering: `h'_i = h_i + ||h_i|| * (v_i / ||v_i||)`
- **Placeholder tokens**: Use `" ?"` as placeholder tokens where activations are injected
- **Training data**: Original paper uses SPQA (system prompt QA), binary classification, and self-supervised context prediction (next/previous token prediction)
- **LoRA fine-tuning**: Computationally cheap — 10 H100 GPU hours for Qwen3-8B

### Coconut / Latent Reasoning

Coconut (Chain of Continuous Thought) trains models to reason in continuous latent space. Instead of generating discrete chain-of-thought tokens, the model produces hidden states that serve as "latent thoughts" and are fed back as input for the next reasoning step. This enables:
- More efficient reasoning (no tokenization bottleneck)
- Potentially richer internal representations
- Non-verbal reasoning that can't be expressed in natural language

## Experiment Design

### Phase 1: Self-Supervised AO via Semantic Description

#### Data Generation Pipeline

For each text passage from a diverse corpus:

1. **Collect activations**: Run Qwen3-8B on the passage, extract residual stream activations at layers 16 (25% depth), 32 (50% depth), and 48 (75% depth). Store activations at specific token positions (varying from 1 to ~10 tokens per example).

2. **Generate semantic descriptions**: Prompt Qwen3-8B with a template like:
   ```
   Describe the semantic content of the following text: '{passage}'
   Focus on: what language is being used, what the topic is, what the grammatical
   structure looks like, what continuations are likely, what the model might
   "believe" about the context.
   ```
   The model produces a rich free-form description covering:
   - Language identification
   - Topic/domain classification
   - Grammatical analysis
   - Sentiment/tone
   - Likely continuations
   - Cultural/contextual implications

3. **Create training examples**: Each example consists of:
   - **Input**: Oracle prompt with placeholder tokens + layer number + question
   - **Target**: The semantic description
   - **Injection data**: The collected activation vectors

#### Training Configuration

- **Model**: Qwen3-8B (64 layers, hidden dim 4096)
- **Fine-tuning**: LoRA (rank 16, alpha 32, applied to q_proj, k_proj, v_proj, o_proj)
- **Injection layer**: Layer 2 (after second transformer block)
- **Activation source layers**: 16, 32, 48 (25%, 50%, 75% depth)
- **Placeholder token**: `" ?"` (token ID varies by tokenizer)
- **Optimizer**: AdamW, lr=2e-4, cosine schedule with warmup
- **Batch size**: 8 per device, gradient accumulation as needed
- **Training**: ~50K-100K examples initially, scale up if promising

#### Text Corpus

Mix of diverse sources:
- **English Wikipedia** (general knowledge)
- **FineWeb** sample (web text diversity)
- **Multilingual text** (test cross-lingual transfer)
- **Code** (test code understanding)
- **Conversational data** (dialogue patterns)

#### Oracle Prompt Format

```
Layer {L}: {placeholder_tokens} Describe the semantic content of this text.
```

Where `{placeholder_tokens}` are K copies of ` ?` and L is the source layer number.

### Phase 2: Evaluation

#### Held-out Description Recovery
- Test on held-out text passages
- Measure ROUGE/BERTScore between predicted and target descriptions
- Qualitative analysis of what information is recoverable

#### Zero-shot Generalization Tests
1. **Language identification**: Can the AO identify the language from activations?
2. **Topic classification**: Can it identify topics never seen during training?
3. **Sentiment analysis**: Can it detect sentiment from activations?
4. **Code understanding**: Can it describe code snippets from activations?

#### Comparison with Context Prediction
- Compare our semantic description approach against the original paper's next/previous token prediction
- Hypothesis: Richer descriptions lead to better generalization because they force the model to extract higher-level semantic information

### Phase 3: Latent Reasoning Extension

#### Simple Coconut-like Implementation
1. Run Qwen3-8B on a reasoning problem (e.g., multi-step math)
2. At specific positions, instead of outputting tokens, take the hidden state as a "latent thought"
3. Feed this hidden state back as input for the next step (inject at the embedding layer)
4. Collect these latent thought vectors

#### AO Training for Latent Reasoning
- Generate reasoning descriptions: What step of reasoning is being performed?
- Train the AO to interpret latent thought vectors
- Evaluate: Can the AO describe intermediate reasoning steps that were never verbalized?

## Technical Architecture

```
Text Passage ──────► Qwen3-8B (forward pass) ──────► Residual Stream Activations
     │                                                        │
     │                                                        ▼
     └──► Qwen3-8B (description generation) ──► Semantic Description
                                                              │
                                                              ▼
                                              Training Pair: (activations, description)
                                                              │
                                                              ▼
                                              Fine-tune Qwen3-8B (LoRA) with
                                              activation injection mechanism
```

## Infrastructure

- **Hardware**: TPU v5litepod-64 (64 TPU v5e chips, 16 workers × 4 chips)
- **Framework**: PyTorch + torch_xla for TPU support
- **Training**: Single-worker initially (4 chips, 64GB HBM), scale to multi-worker if needed
- **Model**: ~16GB in bf16, fits comfortably on single worker

## Key Hypotheses

1. Self-generated semantic descriptions provide richer training signal than next/previous token prediction, leading to better AO generalization.
2. The semantic description approach is fully unsupervised — no human labels needed, making it infinitely scalable.
3. Latent reasoning vectors contain interpretable information about intermediate reasoning steps that the AO can learn to verbalize.
4. Training on diverse text (multilingual, code, conversation) improves AO generality, similar to the paper's finding that training data diversity helps.

## Risk Mitigation

- **Preemptible cluster**: Commit/push frequently, checkpoint every N steps
- **Incremental approach**: Start small (1K examples), validate the pipeline works, then scale
- **Fallback**: If semantic descriptions don't work, fall back to the paper's proven context prediction approach as a baseline
