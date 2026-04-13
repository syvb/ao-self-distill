# Self-Distillation Activation Oracles: Experiment Plan

## Overview

This project extends the Activation Oracle (AO) framework from Karvonen et al. (2025) with a
fully **unsupervised self-distillation** approach. Instead of using manually curated classification
labels or system prompt QA datasets, we train the model to describe its own activations using
descriptions it generates itself.

**Core idea**: The same model (Qwen3-8B) serves three roles:
1. **Target model**: Processes text and produces activations
2. **Description generator**: Prompted to describe the semantic content of that text
3. **Activation oracle**: Fine-tuned to reproduce those descriptions given only activations

This is a form of **model self-distillation** — the model's text-understanding capabilities are
distilled into its ability to read its own internal representations.

## Model Details

- **Model**: Qwen/Qwen3-8B (instruct variant)
- **Architecture**: 36 transformer layers, hidden dim 4096
- **Activation extraction layers**: 9 (25%), 18 (50%), 27 (75%) — primarily layer 18
- **Platform**: v5litepod-64 TPU cluster (4 chips per node, 8 nodes total)

## Experiment Design

### Phase 1: Data Collection

#### 1a. Text Corpus Assembly
- Source diverse text from multiple domains:
  - English Wikipedia passages (varied topics)
  - Multilingual text (French, German, Spanish, Chinese, etc.)
  - Code snippets
  - Poetry and literature
  - Scientific abstracts
  - Conversational data
- Target: ~10K diverse passages, 50-200 tokens each

#### 1b. Activation Extraction
For each text passage:
1. Tokenize and run through Qwen3-8B
2. Extract residual stream activations at the **50% depth layer** (layer 18)
3. Store activations for 1-5 selected token positions per passage
4. Token positions selected to be semantically interesting (content words, not function words)

#### 1c. Self-Distillation Label Generation
For each text passage and selected token position, prompt Qwen3-8B:

```
Describe the semantic content of the following text, focusing on what is happening
at and around the word '{token}' (position {pos}/{total}).

Text: "{passage}"

Provide a detailed description covering:
- The language being used
- The topic and subject matter
- The grammatical role of the focused word
- What the text is communicating
- What continuations or context might follow
- Any notable stylistic or structural features
```

The model generates a free-form description (50-200 tokens) that becomes the training label.

### Phase 2: Training

#### Architecture
Following the AO paper:
- **Placeholder token**: ` ?` (space + question mark), as in the paper
- **Injection mechanism**: Norm-matched addition after transformer layer 2
  - `h'_i = h_i + ||h_i|| * v_i / ||v_i||`
- **Training**: LoRA (rank 64, alpha 128, all linear layers)
- **Format**: `Layer {L}: ? ? ? ... Describe what this text is about.` -> description

#### Training Hyperparameters
- Learning rate: 1e-5
- Batch size: 16 (per device)
- LoRA rank: 64, alpha: 128, dropout: 0.05
- Optimizer: AdamW
- Schedule: Linear warmup (10%) + linear decay
- Epochs: 2-3

### Phase 3: Evaluation

1. **Held-out description quality**: Compare activation-based descriptions to text-based ones
   using the model as a judge
2. **Information preservation**: What aspects of the text are best/worst recovered?
   - Language identification
   - Topic detection
   - Grammatical structure
   - Named entities
   - Sentiment
3. **Layer comparison**: How do descriptions differ across layers 9, 18, 27?
4. **Token count scaling**: How does description quality change with 1 vs 5 vs 20 activation tokens?

### Phase 4: Extensions (if time permits)

1. **Iterative self-distillation**: Train round 1, use it to generate richer descriptions
   (by injecting activations AND asking for more detail), then retrain
2. **Diverse question types**: Instead of just "describe", mix in:
   - "What language is this?"
   - "What is the topic?"
   - "What word comes next?"
   - "What is the grammatical structure?"
3. **Multi-layer injection**: Combine activations from different layers
4. **Activation difference descriptions**: Describe what changes between layers
5. **Cross-passage comparison**: "Are these two activations from similar text?"

## Key Differences from the Original AO Paper

| Aspect | Original AO | Our Approach |
|--------|------------|--------------|
| Training labels | Manual (SPQA, classification) | Self-generated descriptions |
| Label diversity | Narrow task-specific | Rich free-form descriptions |
| Scalability | Limited by manual datasets | Unlimited (any text corpus) |
| Supervision | Supervised | Unsupervised (self-distillation) |
| Output format | Short answers / yes-no | Detailed paragraphs |
| Training cost | ~10 H100-hours | Similar (LoRA) |

## Why This Could Work

1. **The model already knows**: When prompted with text, Qwen3-8B can describe semantics
   fluently. The activations encode this information — we're just training the model to access
   it via a different input modality.
2. **Self-consistency**: Because the same model generates both activations and descriptions,
   there's a natural alignment between what's encoded and what's described.
3. **Rich signal**: Free-form descriptions contain much more training signal per example than
   binary labels, potentially requiring fewer examples to learn.

## Risks and Mitigations

1. **Text inversion**: The model might just learn to invert activations to text, then describe that.
   - Mitigation: Test with activations from fine-tuned models (unseen during training)
   - Mitigation: Check if descriptions contain info not in the text (e.g., from model knowledge)
2. **Description quality**: Self-generated descriptions might be shallow or formulaic
   - Mitigation: Use diverse prompts, temperature sampling, multiple description styles
3. **TPU compatibility**: Qwen3-8B + custom hooks might be tricky on TPU
   - Mitigation: Fall back to CPU for data generation if needed, TPU for training only

## Timeline

1. Environment setup + data pipeline: ~2-3 hours
2. Description generation: ~3-4 hours (can parallelize across nodes)
3. Training: ~2-4 hours
4. Evaluation: ~1-2 hours
5. Extensions: remaining time
