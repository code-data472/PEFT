"""Logging and tensorboard helpers."""

import os
import sys
import logging
from typing import Optional


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = None

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.log_dir, exist_ok=True)
                self._writer = SummaryWriter(self.log_dir)
            except ImportError:
                pass
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int):
        w = self._get_writer()
        if w:
            w.add_scalar(tag, value, step)

    def close(self):
        if self._writer:
            self._writer.close()
