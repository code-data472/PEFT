#!/usr/bin/env python3
"""
Aggregate all experiment results into unified tables (LaTeX + CSV).
"""

import os
import json
import glob
import csv
import argparse
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--output_dir", default="analysis/")
    args = parser.parse_args()

    results = defaultdict(lambda: defaultdict(list))
    all_files = glob.glob(os.path.join(args.experiments_dir, "**/results*.json"), recursive=True)

    for f in sorted(all_files):
        with open(f) as fh:
            data = json.load(fh)
        key = (data.get("model_family",""), data.get("language",""),
               data.get("budget",""), data.get("adaptation",""))
        results[key]["wer"].append(data["test_wer"])
        results[key]["cer"].append(data.get("test_cer", 0))
        results[key]["gpu_hours"].append(data.get("gpu_hours", 0))

    rows = []
    for key, vals in sorted(results.items()):
        model, lang, budget, method = key
        wer_arr = np.array(vals["wer"])
        rows.append({
            "model": model, "language": lang, "budget": budget, "method": method,
            "wer_mean": round(np.mean(wer_arr), 2),
            "wer_std": round(np.std(wer_arr), 2) if len(wer_arr) > 1 else None,
            "n_seeds": len(wer_arr),
            "cer_mean": round(np.mean(vals["cer"]), 2),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "aggregated_results.csv")
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {outpath}")


if __name__ == "__main__":
    main()
