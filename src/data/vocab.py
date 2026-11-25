"""
Character-level vocabulary construction for CTC-based models.
Builds vocab from Common Voice training transcripts.
"""

import re
import json
import logging
from collections import Counter
from typing import Dict

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "|": 4}


def build_char_vocab(language: str, min_count: int = 2) -> Dict[str, int]:
    from datasets import load_dataset
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        language, split="train", trust_remote_code=True,
    )
    counter = Counter()
    for ex in ds:
        text = _normalise(ex["sentence"])
        counter.update(text)

    vocab = dict(SPECIAL_TOKENS)
    idx = len(vocab)
    for ch, cnt in sorted(counter.items()):
        if cnt >= min_count and ch not in vocab:
            vocab[ch] = idx
            idx += 1

    logger.info(f"Built vocab for {language}: {len(vocab)} tokens")
    return vocab


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
