"""
WER/CER computation with text normalisation pipeline.
"""

import re
from typing import List

import jiwer


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(references: List[str], predictions: List[str]) -> float:
    refs = [_normalise(r) for r in references]
    preds = [_normalise(p) for p in predictions]
    refs = [r if r else "<empty>" for r in refs]
    preds = [p if p else "<empty>" for p in preds]
    return jiwer.wer(refs, preds) * 100


def compute_cer(references: List[str], predictions: List[str]) -> float:
    refs = [_normalise(r) for r in references]
    preds = [_normalise(p) for p in predictions]
    refs = [r if r else "<empty>" for r in refs]
    preds = [p if p else "<empty>" for p in preds]
    return jiwer.cer(refs, preds) * 100
