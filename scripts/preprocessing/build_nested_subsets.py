#!/usr/bin/env python3
"""
Build nested subset indices for controlled label budgets.
Outputs JSON index files: idx_10min.json ⊂ idx_30min.json ⊂ idx_1h.json ⊂ idx_3h.json
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset, Audio


BUDGETS = [
    ("10min", 600),
    ("30min", 1800),
    ("1h", 3600),
    ("3h", 10800),
]


def speaker_stratified_nested_sample(dataset, seed=42):
    rng = np.random.RandomState(seed)
    durations = []
    speakers = []
    for ex in dataset:
        dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
        durations.append(dur)
        speakers.append(ex.get("client_id", "unk"))

    durations = np.array(durations)
    speakers = np.array(speakers)
    unique_spk = np.unique(speakers)
    rng.shuffle(unique_spk)

    ordering = []
    for spk in unique_spk:
        spk_idx = np.where(speakers == spk)[0]
        rng.shuffle(spk_idx)
        ordering.extend(spk_idx.tolist())

    subsets = {}
    cumulative_dur = 0.0
    for i, idx in enumerate(ordering):
        cumulative_dur += durations[idx]
        for name, threshold in BUDGETS:
            if name not in subsets and cumulative_dur >= threshold:
                subsets[name] = ordering[:i+1]

    # Fill remaining
    for name, threshold in BUDGETS:
        if name not in subsets:
            subsets[name] = ordering

    return subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--output_dir", default="data/subsets/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        args.language, split="train", trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    subsets = speaker_stratified_nested_sample(ds, args.seed)

    outdir = os.path.join(args.output_dir, args.language)
    os.makedirs(outdir, exist_ok=True)

    for name, indices in subsets.items():
        total_dur = sum(
            len(ds[i]["audio"]["array"]) / 16000 for i in indices
        )
        meta = {
            "language": args.language,
            "budget": name,
            "num_utterances": len(indices),
            "total_seconds": round(total_dur, 1),
            "seed": args.seed,
            "indices": sorted(indices),
        }
        with open(os.path.join(outdir, f"idx_{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{args.language}/{name}: {len(indices)} utt, {total_dur:.0f}s")


if __name__ == "__main__":
    main()
