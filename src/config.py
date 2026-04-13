"""Configuration for the self-distillation activation oracle experiment."""

from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-8B"
    num_layers: int = 36  # Qwen3-8B has 36 transformer layers
    hidden_size: int = 4096
    # Layers to extract activations from (25%, 50%, 75% depth)
    extraction_layers: list = field(default_factory=lambda: [9, 18, 27])
    # Primary layer for evaluation
    primary_layer: int = 18  # 50% depth
    # Layer to inject activations at in the oracle (after this layer)
    injection_layer: int = 1  # After layer 1 (0-indexed), following the paper
    # Placeholder token for activation injection
    placeholder_token: str = " ?"
    dtype: str = "bfloat16"

@dataclass
class DataConfig:
    # Number of text passages to collect
    num_passages: int = 10000
    # Tokens per passage
    min_passage_tokens: int = 50
    max_passage_tokens: int = 200
    # Number of token positions to sample per passage
    min_positions_per_passage: int = 1
    max_positions_per_passage: int = 5
    # Description generation
    max_description_tokens: int = 256
    description_temperature: float = 0.7
    # Data sources
    text_sources: list = field(default_factory=lambda: [
        "fineweb",       # English web text
        "wikipedia",     # Multilingual articles
    ])
    # Output paths
    data_dir: str = "data"
    activations_dir: str = "data/activations"
    descriptions_dir: str = "data/descriptions"
    dataset_path: str = "data/training_dataset.jsonl"

@dataclass
class TrainingConfig:
    # LoRA parameters (matching the paper)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all"  # All linear layers
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4  # Per device, conservative for 8B model
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512
    # Optimizer
    optimizer: str = "adamw"
    # Output
    output_dir: str = "checkpoints"
    save_steps: int = 500
    logging_steps: int = 50

@dataclass
class EvalConfig:
    num_eval_samples: int = 200
    eval_batch_size: int = 4
    eval_output_dir: str = "eval_results"


def get_tpu_env():
    """Return environment variables needed for single-host TPU operation."""
    return {
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_VISIBLE_CHIPS": "0,1,2,3",
        "PJRT_DEVICE": "TPU",
    }
