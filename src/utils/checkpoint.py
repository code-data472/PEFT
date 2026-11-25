"""
Checkpoint save/load with automatic cleanup of old checkpoints.
"""

import os
import glob
import logging
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_wer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_best.pt")
    state = {
        "model_state_dict": {
            k: v for k, v in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_wer": best_wer,
    }
    torch.save(state, path)
    logger.debug(f"Saved checkpoint: {path} (WER={best_wer:.2f})")


def load_checkpoint(output_dir, model, optimizer=None, scheduler=None):
    path = os.path.join(output_dir, "checkpoint_best.pt")
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return None

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    logger.info(
        f"Loaded checkpoint: epoch={state['epoch']}, "
        f"step={state['global_step']}, WER={state['best_wer']:.2f}"
    )
    return state
