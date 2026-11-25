#!/usr/bin/env python3
"""
Unified adaptation training loop for CTC and seq2seq pretrained speech models.
Dispatches to method-specific parameter freezing and module injection logic
based on the adaptation config (full_ft | lora | adapter | lastk).

Usage:
    python -m src.train --config configs/sweeps/xlsr_cy_10min_lora.yaml
    torchrun --nproc_per_node=1 -m src.train --config ... --distributed
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.cuda.amp import GradScaler, autocast

from src.models.factory import build_model
from src.adapters.inject import inject_adaptation_modules, freeze_parameters
from src.data.sampler import BudgetSampler, NestedSubsetSampler
from src.data.collator import DataCollatorCTCWithPadding, DataCollatorSeq2SeqWithPadding
from src.eval.metrics import compute_wer, compute_cer
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_utils import setup_logging, TensorBoardLogger
from src.analysis.cka_tracker import CKATracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    config_path: str = ""
    output_dir: str = "outputs/"
    model_family: str = "xlsr"       # xlsr | whisper | mms
    adaptation: str = "lora"         # full_ft | lora | adapter | lastk
    language: str = "cy"
    budget_seconds: int = 600        # 600=10min, 3600=1h, 10800=3h
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    gradient_checkpointing: bool = True
    early_stopping_patience: int = 5
    eval_steps: int = 50
    save_total_limit: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    adapter_bottleneck: int = 64
    adapter_activation: str = "relu"
    lastk_layers: int = 2
    cka_every_n_epochs: int = 5
    cka_num_utterances: int = 300
    max_audio_length_s: float = 30.0
    dataloader_num_workers: int = 4
    log_level: str = "INFO"
    resume_from: Optional[str] = None


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    cli = parser.parse_args()

    import yaml
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    for ov in cli.override:
        k, v = ov.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v

    args = TrainingArgs(**{k: v for k, v in cfg.items() if k in TrainingArgs.__dataclass_fields__})
    args.config_path = cli.config
    return args


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train(args: TrainingArgs):
    setup_logging(args.log_level)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Config: {json.dumps(asdict(args), indent=2, default=str)}")

    # Build model + processor
    model, processor = build_model(args.model_family, args.language)
    model = model.to(device)

    # Inject adaptation modules and freeze
    trainable_params, total_params = inject_adaptation_modules(model, args)
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Data
    sampler = NestedSubsetSampler(
        language=args.language,
        budget_seconds=args.budget_seconds,
        seed=args.seed,
    )
    train_dataset = sampler.get_train_dataset(processor, args.max_audio_length_s)
    val_dataset = sampler.get_val_dataset(processor, args.max_audio_length_s)
    test_dataset = sampler.get_test_dataset(processor, args.max_audio_length_s)

    is_seq2seq = args.model_family == "whisper"
    collator_cls = DataCollatorSeq2SeqWithPadding if is_seq2seq else DataCollatorCTCWithPadding
    collator = collator_cls(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=args.dataloader_num_workers,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.max_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=args.fp16)
    early_stop = EarlyStopping(args.early_stopping_patience)

    # CKA tracker
    cka_tracker = CKATracker(
        model_family=args.model_family,
        num_utterances=args.cka_num_utterances,
        language=args.language,
        device=device,
    )
    cka_tracker.cache_frozen_representations(model)

    tb_logger = TensorBoardLogger(os.path.join(args.output_dir, "tb_logs"))
    best_wer = float("inf")
    global_step = 0

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.eval_steps == 0:
                    val_wer = evaluate(model, val_loader, processor, is_seq2seq, device)
                    tb_logger.log_scalar("val/wer", val_wer, global_step)
                    logger.info(f"Step {global_step} | Val WER: {val_wer:.2f}%")

                    improved = early_stop.step(val_wer)
                    if improved:
                        best_wer = val_wer
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                        best_wer, args.output_dir)

                    if early_stop.should_stop:
                        logger.info(f"Early stopping at step {global_step}")
                        break

        if early_stop.should_stop:
            break

        avg_loss = epoch_loss / max(len(train_loader), 1)
        tb_logger.log_scalar("train/loss", avg_loss, epoch)

        # Periodic CKA
        if (epoch + 1) % args.cka_every_n_epochs == 0:
            cka_scores = cka_tracker.compute_layerwise_cka(model)
            for layer_idx, score in enumerate(cka_scores):
                tb_logger.log_scalar(f"cka/layer_{layer_idx}", score, epoch)
            drift = cka_tracker.compute_aggregate_drift(cka_scores, top_k=6)
            tb_logger.log_scalar("cka/aggregate_drift_top6", drift, epoch)

    # Final eval
    ckpt = load_checkpoint(args.output_dir, model)
    test_wer = evaluate(model, DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=args.dataloader_num_workers,
    ), processor, is_seq2seq, device)

    results = {
        "test_wer": test_wer,
        "best_val_wer": best_wer,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "epochs_trained": epoch + 1,
        "global_steps": global_step,
        "config": asdict(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Test WER: {test_wer:.2f}% | Best Val WER: {best_wer:.2f}%")
    return results


@torch.no_grad()
def evaluate(model, loader, processor, is_seq2seq, device):
    model.eval()
    all_preds, all_refs = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if is_seq2seq:
            generated = model.generate(
                batch["input_features"],
                max_new_tokens=225,
                language=batch.get("language", None),
            )
            preds = processor.batch_decode(generated, skip_special_tokens=True)
        else:
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(pred_ids)

        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(labels, group_tokens=False)
        all_preds.extend(preds)
        all_refs.extend(refs)

    model.train()
    return compute_wer(all_refs, all_preds)


if __name__ == "__main__":
    args = parse_args()
    train(args)
