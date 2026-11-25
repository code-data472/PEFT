"""
Speaker-stratified nested subset sampler for controlled label budgets.
Guarantees 10min ⊂ 30min ⊂ 1h ⊂ 3h ⊂ full.
"""

import os
import logging
from typing import Optional, List

import numpy as np
import torch
from datasets import load_dataset, Audio

logger = logging.getLogger(__name__)

CV_VERSION = "17.0"
BUDGET_MAP = {
    "10min": 600,
    "30min": 1800,
    "1h": 3600,
    "3h": 10800,
}

LANG_CODES = {
    "cy": "cy",
    "eu": "eu",
    "mt": "mt",
    "br": "br",
    "ha": "ha",
}


class NestedSubsetSampler:
    def __init__(self, language: str, budget_seconds: int, seed: int = 42):
        self.language = language
        self.budget_seconds = budget_seconds
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._dataset = None

    def _load_cv(self, split: str):
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            LANG_CODES[self.language],
            split=split,
            trust_remote_code=True,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    def _speaker_stratified_sample(self, dataset, budget_s: float):
        durations = []
        speakers = []
        for ex in dataset:
            dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            durations.append(dur)
            speakers.append(ex.get("client_id", "unknown"))

        durations = np.array(durations)
        speakers = np.array(speakers)
        unique_speakers = np.unique(speakers)
        self._rng.shuffle(unique_speakers)

        selected_indices = []
        total_dur = 0.0

        for spk in unique_speakers:
            spk_mask = speakers == spk
            spk_indices = np.where(spk_mask)[0]
            self._rng.shuffle(spk_indices)
            for idx in spk_indices:
                if total_dur + durations[idx] > budget_s:
                    continue
                selected_indices.append(int(idx))
                total_dur += durations[idx]
                if total_dur >= budget_s:
                    break
            if total_dur >= budget_s:
                break

        logger.info(
            f"Sampled {len(selected_indices)} utterances, "
            f"{total_dur:.1f}s / {budget_s}s budget"
        )
        return sorted(selected_indices)

    def get_train_dataset(self, processor, max_len_s: float = 30.0):
        ds = self._load_cv("train")
        indices = self._speaker_stratified_sample(ds, self.budget_seconds)
        ds = ds.select(indices)
        ds = ds.filter(
            lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= max_len_s
        )
        return self._preprocess(ds, processor)

    def get_val_dataset(self, processor, max_len_s: float = 30.0):
        ds = self._load_cv("validation")
        ds = ds.filter(
            lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= max_len_s
        )
        return self._preprocess(ds, processor)

    def get_test_dataset(self, processor, max_len_s: float = 30.0):
        ds = self._load_cv("test")
        ds = ds.filter(
            lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= max_len_s
        )
        return self._preprocess(ds, processor)

    def _preprocess(self, ds, processor):
        def _map_fn(batch):
            audio = batch["audio"]
            inputs = processor(
                audio["array"],
                sampling_rate=16000,
                return_tensors="np",
                padding=False,
            )
            with processor.as_target_processor():
                labels = processor(batch["sentence"]).input_ids
            return {
                "input_values": inputs.input_values[0] if hasattr(inputs, "input_values") else None,
                "input_features": inputs.input_features[0] if hasattr(inputs, "input_features") else None,
                "labels": labels,
            }

        ds = ds.map(_map_fn, remove_columns=ds.column_names)
        return ds


def load_probe_utterances(language: str, num_utterances: int = 300) -> torch.Tensor:
    """Load a fixed set of utterances for CKA probing (from validation split)."""
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        LANG_CODES[language],
        split="validation",
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.select(range(min(num_utterances, len(ds))))

    # Stack with zero-padding
    arrays = [torch.tensor(ex["audio"]["array"], dtype=torch.float32) for ex in ds]
    max_len = max(a.shape[0] for a in arrays)
    padded = torch.zeros(len(arrays), max_len)
    for i, a in enumerate(arrays):
        padded[i, :a.shape[0]] = a

    return padded
