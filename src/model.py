"""
Model utilities for the Self-Supervised Activation Oracle.

Handles:
- Loading Qwen3-8B with appropriate device placement
- Collecting residual stream activations at configurable layers
- Implementing the activation injection mechanism (norm-matched steering at layer 2)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from contextlib import contextmanager


MODEL_NAME = "Qwen/Qwen3-8B"
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
INJECTION_LAYER = 2  # Inject after 2nd transformer block (0-indexed)
DEFAULT_SOURCE_LAYERS = [9, 18, 27]  # 25%, 50%, 75% depth
PLACEHOLDER_TOKEN = " ?"


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device if device != "xla" else None,
    )

    if device == "xla":
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        model = model.to(dev)

    model.eval()
    return model, tokenizer


def get_placeholder_token_id(tokenizer) -> int:
    """Get the token ID for the placeholder token ' ?'."""
    ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    # Use the last token if the placeholder encodes to multiple tokens
    return ids[-1]


class ActivationCollector:
    """Collects residual stream activations at specified layers during a forward pass."""

    def __init__(self, model, source_layers: list[int] = None):
        self.model = model
        self.source_layers = source_layers or DEFAULT_SOURCE_LAYERS
        self.activations = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # For Qwen3, the decoder layer output is a tuple: (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on the target layers."""
        self.clear()
        for layer_idx in self.source_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def clear(self):
        """Remove all hooks (but keep stored activations for later access)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @contextmanager
    def collect(self):
        """Context manager that registers hooks, yields, then cleans up."""
        self.register_hooks()
        try:
            self.activations = {}
            yield self
        finally:
            self.clear()

    def get_activations(self, layer_idx: int, token_positions: list[int] = None):
        """Get activations for a specific layer, optionally at specific token positions.

        Returns: tensor of shape (num_positions, hidden_size)
        """
        acts = self.activations[layer_idx]  # (batch, seq_len, hidden_size)
        if token_positions is not None:
            acts = acts[:, token_positions, :]  # (batch, num_positions, hidden_size)
        return acts.squeeze(0)  # Remove batch dim if batch=1


class ActivationInjector:
    """Injects activation vectors at placeholder positions using norm-matched additive steering.

    After the injection layer (layer 2), for each placeholder token position i:
        h'_i = h_i + ||h_i|| * (v_i / ||v_i||)

    This adds a norm-matched version of the source activation to the placeholder's
    representation, preserving the original activation magnitude.
    """

    def __init__(self, model, injection_layer: int = INJECTION_LAYER):
        self.model = model
        self.injection_layer = injection_layer
        self._hook = None
        self._injection_data = None  # (placeholder_positions, activation_vectors)

    def set_injection_data(self, placeholder_positions: list[int], activation_vectors: torch.Tensor):
        """Set the data to inject.

        Args:
            placeholder_positions: Token positions where activations should be injected
            activation_vectors: Tensor of shape (num_positions, hidden_size)
        """
        self._injection_data = (placeholder_positions, activation_vectors)

    def _injection_hook(self, module, input, output):
        if self._injection_data is None:
            return output

        positions, vectors = self._injection_data

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Ensure vectors are on the same device and dtype as hidden states
        vectors = vectors.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Norm-matched additive steering
        for i, pos in enumerate(positions):
            h_i = hidden_states[:, pos, :]  # (batch, hidden_size)
            v_i = vectors[i].unsqueeze(0)  # (1, hidden_size)

            h_norm = torch.norm(h_i, dim=-1, keepdim=True)  # (batch, 1)
            v_norm = torch.norm(v_i, dim=-1, keepdim=True)  # (1, 1)

            # h'_i = h_i + ||h_i|| * (v_i / ||v_i||)
            v_normalized = v_i / (v_norm + 1e-8)
            hidden_states[:, pos, :] = h_i + h_norm * v_normalized

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    @contextmanager
    def inject(self, placeholder_positions: list[int], activation_vectors: torch.Tensor):
        """Context manager that sets up injection, yields, then cleans up."""
        self.set_injection_data(placeholder_positions, activation_vectors)
        layer = self.model.model.layers[self.injection_layer]
        self._hook = layer.register_forward_hook(self._injection_hook)
        try:
            yield self
        finally:
            if self._hook is not None:
                self._hook.remove()
                self._hook = None
            self._injection_data = None


def collect_activations_for_text(
    model,
    tokenizer,
    text: str,
    source_layers: list[int] = None,
    token_positions: list[int] = None,
    max_length: int = 512,
):
    """Collect residual stream activations for a text passage.

    Args:
        model: The model to collect activations from
        tokenizer: The tokenizer
        text: Input text passage
        source_layers: Which layers to collect from (default: 25%, 50%, 75%)
        token_positions: Which token positions to collect (default: all)
        max_length: Maximum sequence length

    Returns:
        dict mapping layer_idx -> tensor of shape (num_positions, hidden_size)
        list of token positions used
        input_ids tensor
    """
    source_layers = source_layers or DEFAULT_SOURCE_LAYERS

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    seq_len = input_ids.shape[1]
    if token_positions is None:
        # Default: collect from all positions
        token_positions = list(range(seq_len))

    collector = ActivationCollector(model, source_layers)
    with collector.collect():
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

    result = {}
    for layer_idx in source_layers:
        result[layer_idx] = collector.get_activations(layer_idx, token_positions)

    return result, token_positions, input_ids


def build_oracle_prompt(
    tokenizer,
    num_activations: int,
    source_layer: int,
    question: str = "Describe the semantic content of this text.",
) -> tuple[str, list[int]]:
    """Build an oracle prompt with placeholder tokens.

    Returns the prompt string and the positions of placeholder tokens in the tokenized sequence.
    """
    # Build prompt: "Layer {L}: ? ? ? ... ? {question}"
    placeholders = PLACEHOLDER_TOKEN * num_activations
    prompt = f"Layer {source_layer}:{placeholders} {question}"

    # Tokenize and find placeholder positions
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"][0]

    placeholder_id = get_placeholder_token_id(tokenizer)
    placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()

    return prompt, placeholder_positions, tokens
