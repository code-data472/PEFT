"""
Model and processor instantiation for the three pretrained families.
Handles CTC head construction for XLS-R / MMS and seq2seq setup for Whisper.
"""

import logging
from typing import Tuple, Any

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCTC,
    AutoProcessor,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "xlsr": {
        "checkpoint": "facebook/wav2vec2-xls-r-300m",
        "type": "ctc",
        "num_layers": 24,
        "hidden_size": 1024,
    },
    "whisper": {
        "checkpoint": "openai/whisper-small",
        "type": "seq2seq",
        "num_encoder_layers": 12,
        "hidden_size": 768,
    },
    "mms": {
        "checkpoint": "facebook/mms-300m",
        "type": "ctc",
        "num_layers": 24,
        "hidden_size": 1024,
    },
}

_VOCAB_CACHE = {}


def _load_vocab(language: str):
    """Load or build character-level vocabulary for CTC models."""
    if language in _VOCAB_CACHE:
        return _VOCAB_CACHE[language]
    from src.data.vocab import build_char_vocab
    vocab = build_char_vocab(language)
    _VOCAB_CACHE[language] = vocab
    return vocab


def build_model(model_family: str, language: str) -> Tuple[Any, Any]:
    spec = MODEL_REGISTRY[model_family]
    ckpt = spec["checkpoint"]

    if spec["type"] == "ctc":
        vocab = _load_vocab(language)
        processor = Wav2Vec2Processor.from_pretrained(
            ckpt,
            vocab=vocab,
            do_lower_case=True,
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            ckpt,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
        )
        # Freeze feature encoder (conv layers) per Baevski et al. 2020
        model.freeze_feature_encoder()
        logger.info(f"Loaded CTC model: {ckpt} | vocab_size={len(processor.tokenizer)}")

    elif spec["type"] == "seq2seq":
        processor = WhisperProcessor.from_pretrained(ckpt)
        model = WhisperForConditionalGeneration.from_pretrained(ckpt)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=_WHISPER_LANG_MAP.get(language, language),
            task="transcribe",
        )
        model.config.suppress_tokens = []
        logger.info(f"Loaded seq2seq model: {ckpt}")
    else:
        raise ValueError(f"Unknown model type: {spec['type']}")

    return model, processor


_WHISPER_LANG_MAP = {
    "cy": "welsh",
    "eu": "basque",
    "mt": "maltese",
    "br": "breton",
    "ha": "hausa",
}
