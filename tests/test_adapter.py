import torch
import torch.nn as nn
import pytest
from src.adapters.bottleneck import BottleneckAdapter


def test_adapter_output_shape():
    adapter = BottleneckAdapter(hidden_size=256, bottleneck_dim=32)
    x = torch.randn(2, 10, 256)
    y = adapter(x)
    assert y.shape == x.shape


def test_adapter_initial_residual():
    """Adapter should be near-identity at init (up_proj weights = 0)."""
    adapter = BottleneckAdapter(hidden_size=128, bottleneck_dim=16)
    x = torch.randn(1, 5, 128)
    with torch.no_grad():
        y = adapter(x)
    # After LayerNorm, not exactly equal but close
    assert torch.allclose(y, adapter.layer_norm(x), atol=1e-5)
