"""
Low-Rank Adaptation (LoRA) for transformer attention projections.
Ref: Hu et al., ICLR 2022.
"""

import math
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer: W_eff = W_frozen + (alpha/r) * A @ B
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen original weight
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # Low-rank factors
        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero => initial output = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.rank}, s={self.scaling:.2f}"


def inject_lora_modules(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: List[str] = None,
    model_family: str = "xlsr",
    dropout: float = 0.0,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                orig = getattr(module, target)
                if isinstance(orig, nn.Linear):
                    lora_layer = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, target, lora_layer)
                    replaced += 1

    logger.info(f"Injected LoRA (r={rank}) into {replaced} linear layers")
    return replaced
