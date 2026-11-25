"""
Adaptation module injection and parameter freezing logic.
Supports: full_ft, lora, adapter, lastk.
"""

import logging
import re
from typing import Tuple

import torch
import torch.nn as nn

from src.adapters.lora import LoRALinear, inject_lora_modules
from src.adapters.bottleneck import BottleneckAdapter, inject_bottleneck_adapters

logger = logging.getLogger(__name__)


def inject_adaptation_modules(model: nn.Module, args) -> Tuple[int, int]:
    """
    Modify model in-place: freeze appropriate parameters and inject
    PEFT modules according to args.adaptation.

    Returns (trainable_params, total_params).
    """
    total_params = sum(p.numel() for p in model.parameters())

    if args.adaptation == "full_ft":
        # Only freeze feature encoder (already done in factory for CTC)
        # For Whisper: train everything
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    elif args.adaptation == "lora":
        freeze_parameters(model, freeze_all=True)
        _unfreeze_head(model, args.model_family)
        inject_lora_modules(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            model_family=args.model_family,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    elif args.adaptation == "adapter":
        freeze_parameters(model, freeze_all=True)
        _unfreeze_head(model, args.model_family)
        inject_bottleneck_adapters(
            model,
            bottleneck_dim=args.adapter_bottleneck,
            activation=args.adapter_activation,
            model_family=args.model_family,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    elif args.adaptation == "lastk":
        freeze_parameters(model, freeze_all=True)
        _unfreeze_head(model, args.model_family)
        _unfreeze_last_k_layers(model, args.lastk_layers, args.model_family)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    elif args.adaptation == "frozen":
        freeze_parameters(model, freeze_all=True)
        _unfreeze_head(model, args.model_family)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise ValueError(f"Unknown adaptation: {args.adaptation}")

    return trainable_params, total_params


def freeze_parameters(model: nn.Module, freeze_all: bool = True):
    for p in model.parameters():
        p.requires_grad = not freeze_all


def _unfreeze_head(model: nn.Module, model_family: str):
    if model_family in ("xlsr", "mms"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif model_family == "whisper":
        # Unfreeze decoder LM head projection
        if hasattr(model, "proj_out"):
            for p in model.proj_out.parameters():
                p.requires_grad = True


def _unfreeze_last_k_layers(model: nn.Module, k: int, model_family: str):
    if model_family in ("xlsr", "mms"):
        encoder_layers = model.wav2vec2.encoder.layers
    elif model_family == "whisper":
        encoder_layers = model.model.encoder.layers
    else:
        raise ValueError

    n = len(encoder_layers)
    for i in range(n - k, n):
        for p in encoder_layers[i].parameters():
            p.requires_grad = True

    logger.info(f"Unfroze last {k}/{n} encoder layers")
