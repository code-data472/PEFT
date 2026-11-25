#!/usr/bin/env python3
"""
Batch CKA computation across all experiment checkpoints.
Extracts layer-wise CKA between frozen and adapted encoder for each run.
"""

import os
import sys
import json
import glob
import argparse
import logging

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model
from src.analysis.cka_tracker import CKATracker
from src.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_utterances", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    result_files = glob.glob(os.path.join(args.experiments_dir, "**/results*.json"), recursive=True)

    for rf in sorted(result_files):
        with open(rf) as f:
            meta = json.load(f)

        model_family = meta.get("model_family", "xlsr")
        language = meta.get("language", "cy")
        method = meta.get("adaptation", "lora")
        budget = meta.get("budget", "10min")

        ckpt_dir = os.path.dirname(rf)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pt")

        if not os.path.exists(ckpt_path):
            logger.warning(f"No checkpoint for {rf}, skipping")
            continue

        model, processor = build_model(model_family, language)
        model = model.to(device)

        tracker = CKATracker(model_family, args.num_utterances, language, device)
        tracker.cache_frozen_representations(model)

        load_checkpoint(ckpt_dir, model)
        cka_scores = tracker.compute_layerwise_cka(model)
        drift = tracker.compute_drift_profile(cka_scores)

        output = {
            "model": model_family,
            "language": language,
            "budget": budget,
            "method": method,
            "seed": meta.get("seed", 42),
            "num_probe_utterances": args.num_utterances,
            "layerwise_cka": cka_scores,
            "layerwise_drift": drift,
            "aggregate_drift_top6": tracker.compute_aggregate_drift(cka_scores, top_k=6),
        }

        outname = f"cka_{model_family}_{language}_{budget}_{method}_s{meta.get('seed',42)}.json"
        with open(os.path.join(args.output_dir, outname), "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"CKA computed: {outname}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
