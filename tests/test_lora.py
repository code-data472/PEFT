import torch
import torch.nn as nn
import pytest
from src.adapters.lora import LoRALinear


def test_lora_output_shape():
    orig = nn.Linear(64, 32)
    lora = LoRALinear(orig, rank=4, alpha=8)
    x = torch.randn(2, 10, 64)
    y = lora(x)
    assert y.shape == (2, 10, 32)


def test_lora_initial_identity():
    """LoRA should produce identical output to original at init (B=0)."""
    orig = nn.Linear(64, 32)
    lora = LoRALinear(orig, rank=4)
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        y_orig = orig(x)
        y_lora = lora(x)
    assert torch.allclose(y_orig, y_lora, atol=1e-6)


def test_lora_trainable_params():
    orig = nn.Linear(64, 32)
    lora = LoRALinear(orig, rank=4)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    expected = 64 * 4 + 4 * 32  # A + B
    assert trainable == expected
