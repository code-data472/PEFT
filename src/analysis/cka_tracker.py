"""
Layer-wise CKA computation and drift tracking.
Computes linear CKA between frozen and adapted encoder representations.
"""

import copy
import logging
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def _centering_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def linear_hsic(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    H = _centering_matrix(n)
    return np.trace(X.T @ H @ Y @ Y.T @ H @ X) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    hsic_xy = linear_hsic(X, Y)
    hsic_xx = linear_hsic(X, X)
    hsic_yy = linear_hsic(Y, Y)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


class CKATracker:
    def __init__(
        self,
        model_family: str,
        num_utterances: int = 300,
        language: str = "cy",
        device: torch.device = None,
    ):
        self.model_family = model_family
        self.num_utterances = num_utterances
        self.language = language
        self.device = device or torch.device("cpu")
        self._frozen_representations: Optional[Dict[int, np.ndarray]] = None
        self._probe_inputs = None

    def _get_encoder_layers(self, model: nn.Module):
        if self.model_family in ("xlsr", "mms"):
            return model.wav2vec2.encoder.layers
        elif self.model_family == "whisper":
            return model.model.encoder.layers
        raise ValueError

    def _load_probe_data(self):
        from src.data.sampler import load_probe_utterances
        return load_probe_utterances(self.language, self.num_utterances)

    @torch.no_grad()
    def _extract_representations(
        self, model: nn.Module, inputs: torch.Tensor
    ) -> Dict[int, np.ndarray]:
        model.eval()
        representations = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Flatten time: (B, T, D) -> (B*T, D)
                representations[layer_idx] = h.detach().cpu().reshape(-1, h.shape[-1]).numpy()
            return hook_fn

        layers = self._get_encoder_layers(model)
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        # Forward in chunks to avoid OOM
        chunk_size = 8
        for start in range(0, inputs.shape[0], chunk_size):
            chunk = inputs[start:start + chunk_size].to(self.device)
            if self.model_family in ("xlsr", "mms"):
                model(chunk)
            elif self.model_family == "whisper":
                model.model.encoder(chunk)

        for h in hooks:
            h.remove()

        # Concatenate chunks (already accumulated via dict)
        return representations

    def cache_frozen_representations(self, model: nn.Module):
        if self._probe_inputs is None:
            self._probe_inputs = self._load_probe_data()
        self._frozen_representations = self._extract_representations(model, self._probe_inputs)
        logger.info(f"Cached frozen representations for {len(self._frozen_representations)} layers")

    def compute_layerwise_cka(self, model: nn.Module) -> List[float]:
        if self._frozen_representations is None:
            raise RuntimeError("Must call cache_frozen_representations first")

        adapted_reps = self._extract_representations(model, self._probe_inputs)
        cka_scores = []

        for layer_idx in sorted(self._frozen_representations.keys()):
            X_frozen = self._frozen_representations[layer_idx]
            X_adapted = adapted_reps[layer_idx]
            # Subsample frames for efficiency
            n = min(X_frozen.shape[0], X_adapted.shape[0], 10000)
            idx = np.random.choice(X_frozen.shape[0], n, replace=False)
            score = linear_cka(X_frozen[idx], X_adapted[idx])
            cka_scores.append(score)

        return cka_scores

    @staticmethod
    def compute_aggregate_drift(cka_scores: List[float], top_k: int = 6) -> float:
        top_scores = cka_scores[-top_k:]
        return float(np.mean([1 - s for s in top_scores]))

    @staticmethod
    def compute_drift_profile(cka_scores: List[float]) -> List[float]:
        return [1.0 - s for s in cka_scores]
