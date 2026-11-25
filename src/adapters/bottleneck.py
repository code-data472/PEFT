"""
Bottleneck adapter modules for transformer blocks.
Ref: Houlsby et al., ICML 2019.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class BottleneckAdapter(nn.Module):
    """
    Feedforward bottleneck adapter: down-proj -> activation -> up-proj + residual.
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int, activation: str = "relu"):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = ACTIVATIONS[activation]()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return self.layer_norm(residual + h)


class AdapterWrappedLayer(nn.Module):
    """Wraps a transformer sublayer output with an adapter."""

    def __init__(self, original_layer: nn.Module, adapter: BottleneckAdapter):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.original_layer(*args, **kwargs)
        if isinstance(out, tuple):
            hidden = out[0]
            hidden = self.adapter(hidden)
            return (hidden,) + out[1:]
        return self.adapter(out)


def inject_bottleneck_adapters(
    model: nn.Module,
    bottleneck_dim: int = 64,
    activation: str = "relu",
    model_family: str = "xlsr",
):
    if model_family in ("xlsr", "mms"):
        encoder_layers = model.wav2vec2.encoder.layers
        hidden_size = model.config.hidden_size
    elif model_family == "whisper":
        encoder_layers = model.model.encoder.layers
        hidden_size = model.config.d_model
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    count = 0
    for layer in encoder_layers:
        # Post-attention adapter
        adapter_attn = BottleneckAdapter(hidden_size, bottleneck_dim, activation)
        for p in adapter_attn.parameters():
            p.requires_grad = True

        # Post-FFN adapter
        adapter_ffn = BottleneckAdapter(hidden_size, bottleneck_dim, activation)
        for p in adapter_ffn.parameters():
            p.requires_grad = True

        if model_family in ("xlsr", "mms"):
            layer.attention = AdapterWrappedLayer(layer.attention, adapter_attn)
            layer.feed_forward = AdapterWrappedLayer(layer.feed_forward, adapter_ffn)
        elif model_family == "whisper":
            layer.self_attn = AdapterWrappedLayer(layer.self_attn, adapter_attn)
            layer.fc2 = AdapterWrappedLayer(layer.fc2, adapter_ffn)

        count += 2

    logger.info(f"Injected {count} bottleneck adapters (m={bottleneck_dim})")
    return count
