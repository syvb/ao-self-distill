# Experiment Log

## Session 1: Initial Setup (2026-04-13)

### Environment
- **Cluster**: v5litepod-64 TPU (16 workers, 4 chips each, 64 total)
- **Worker 0**: 188GB RAM, AMD EPYC 7B13 (240 cores), 97GB disk
- **Software**: Python 3.10, PyTorch 2.9.0, torch_xla 2.9.0, transformers 5.5.4
- **Model**: Qwen/Qwen3-8B (36 layers, 4096 hidden dim)

### TPU Configuration
Single-host mode on worker 0:
```
TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
TPU_HOST_BOUNDS=1,1,1
TPU_VISIBLE_CHIPS=0,1,2,3
PJRT_DEVICE=TPU
```

### Key Findings
1. **TPU generation is impractical**: XLA recompilation on each autoregressive step
   makes `model.generate()` very slow on TPU. Forward passes work fine.
2. **CPU generation speed**: ~1 text description per minute (128 tokens, greedy)
   with Qwen3-8B bf16 on 240 AMD cores
3. **Qwen3 thinking mode**: Must disable with `enable_thinking=False` in
   `apply_chat_template()` to get clean descriptions without `<think>` tags
4. **Multi-host SSH**: Not available - SSH keys not set up between workers

### Architecture Decisions
- **Activation layers**: 9 (25%), 18 (50%), 27 (75%) depth
- **Injection**: After layer 2 (following the original AO paper)
- **Injection method**: Norm-matched addition: h' = h + ||h|| * (v / ||v||)
- **LoRA config**: rank=64, alpha=128, all-linear layers, dropout=0.05
- **Placeholder token**: " ?" (space + question mark)

### Pipeline
1. **Data generation**: Separated into two steps:
   - Description generation (slow, ~1/min on CPU)
   - Activation collection (fast, forward pass only)
2. **Training**: LoRA fine-tuning with custom injection hooks
3. **Evaluation**: Generate from activations, compare with ground truth

### Bugs Fixed
- `ActivationCollector.clear()` was clearing stored activations before they
  were read in `collect_activations_for_text()` - fixed by not clearing
  activations dict in `clear()`
- `AODataset` expected `activations_{i}.pt` dict format but generation saves
  per-layer files `act_{i}_L{layer}.pt` - fixed dataset class to match

### Data Generation Status
- Target: 200 texts → ~600 training examples (3 layers × ~1 position each)
- Descriptions: greedy decoding, 128 max tokens, varied prompt templates
- ETA: ~3.5 hours from start

### Next Steps
1. Wait for enough data (~100+ examples) to start training
2. Train on TPU with LoRA (should be fast once data is ready)
3. Evaluate on held-out examples
4. Try multi-layer activation injection
5. Explore iterative self-distillation (train → generate better descriptions → retrain)
