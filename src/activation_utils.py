"""Utilities for extracting and injecting activations, following the AO paper."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager


class ActivationExtractor:
    """Extract residual stream activations from specific layers and positions."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.activations: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # For Qwen3, each layer outputs a tuple; residual stream is output[0]
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach()
            else:
                self.activations[layer_idx] = output.detach()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on the specified layers."""
        self.clear_hooks()
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def extract(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                positions: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and extract activations.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            positions: Token positions to extract. If None, extract all.

        Returns:
            Dict mapping layer_idx -> activations [batch, num_positions, hidden_size]
        """
        self.activations = {}

        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        result = {}
        for layer_idx, acts in self.activations.items():
            if positions is not None:
                # Extract specific positions
                result[layer_idx] = acts[:, positions, :]
            else:
                result[layer_idx] = acts

        return result


class ActivationInjector:
    """Inject activations at placeholder token positions using norm-matched addition.

    Following the paper (Equation 1):
        h'_i = h_i + ||h_i|| * v_i / ||v_i||

    This is applied after the injection layer (default: layer 1).
    """

    def __init__(self, model, injection_layer: int = 1):
        self.model = model
        self.injection_layer = injection_layer
        self.hook = None
        self.injection_vectors: Optional[torch.Tensor] = None
        self.injection_positions: Optional[List[int]] = None

    def _injection_hook(self, module, input, output):
        """Hook that performs norm-matched addition at placeholder positions."""
        if self.injection_vectors is None or self.injection_positions is None:
            return output

        # Get the residual stream from the output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()

        # Apply norm-matched addition at each placeholder position
        for i, pos in enumerate(self.injection_positions):
            if i >= self.injection_vectors.shape[0]:
                break

            h_i = hidden_states[:, pos, :]  # [batch, hidden]
            v_i = self.injection_vectors[i].to(h_i.device)  # [hidden] or [batch, hidden]

            if v_i.dim() == 1:
                v_i = v_i.unsqueeze(0).expand_as(h_i)

            # Norm-matched addition: h'_i = h_i + ||h_i|| * v_i / ||v_i||
            h_norm = torch.norm(h_i, dim=-1, keepdim=True)
            v_norm = torch.norm(v_i, dim=-1, keepdim=True).clamp(min=1e-8)

            hidden_states[:, pos, :] = h_i + h_norm * (v_i / v_norm)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def register_hook(self):
        """Register the injection hook."""
        self.clear_hook()
        layer = self.model.model.layers[self.injection_layer]
        self.hook = layer.register_forward_hook(self._injection_hook)

    def clear_hook(self):
        """Remove the injection hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def set_injection(self, vectors: torch.Tensor, positions: List[int]):
        """
        Set the vectors to inject and their positions.

        Args:
            vectors: Activation vectors to inject [num_positions, hidden_size]
            positions: Token positions where placeholders are
        """
        self.injection_vectors = vectors
        self.injection_positions = positions

    def clear_injection(self):
        """Clear current injection state."""
        self.injection_vectors = None
        self.injection_positions = None


@contextmanager
def extract_activations(model, layers: List[int]):
    """Context manager for activation extraction."""
    extractor = ActivationExtractor(model, layers)
    extractor.register_hooks()
    try:
        yield extractor
    finally:
        extractor.clear_hooks()


@contextmanager
def inject_activations(model, injection_layer: int = 1):
    """Context manager for activation injection."""
    injector = ActivationInjector(model, injection_layer)
    injector.register_hook()
    try:
        yield injector
    finally:
        injector.clear_hook()


def find_placeholder_positions(input_ids: torch.Tensor, tokenizer) -> List[int]:
    """Find positions of placeholder tokens (` ?`) in the input."""
    # The placeholder token " ?" - get its token ID
    placeholder_ids = tokenizer.encode(" ?", add_special_tokens=False)
    if len(placeholder_ids) == 1:
        placeholder_id = placeholder_ids[0]
    else:
        # If " ?" tokenizes to multiple tokens, use the last one
        placeholder_id = placeholder_ids[-1]

    positions = []
    for i in range(input_ids.shape[1]):
        if input_ids[0, i].item() == placeholder_id:
            positions.append(i)

    return positions
