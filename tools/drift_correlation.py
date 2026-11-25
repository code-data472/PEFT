#!/usr/bin/env python3
"""
Compute Spearman/Pearson correlations between CKA drift and WER.
Generates scatter plot data and statistical test results.
"""

import os
import json
import glob
import argparse
import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cka_dir", required=True)
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budget_filter", default="10min")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cka_files = glob.glob(os.path.join(args.cka_dir, f"*_{args.budget_filter}_*.json"))
    pairs = []

    for cf in cka_files:
        with open(cf) as f:
            cka_data = json.load(f)

        drift = cka_data["aggregate_drift_top6"]
        model = cka_data["model"]
        lang = cka_data["language"]
        method = cka_data["method"]

        result_pattern = os.path.join(
            args.experiments_dir, model, lang, args.budget_filter, f"*{method}*/results*.json"
        )
        result_files = glob.glob(result_pattern)
        if not result_files:
            continue

        with open(result_files[0]) as f:
            result = json.load(f)
        wer = result["test_wer"]

        pairs.append({
            "model": model, "language": lang, "method": method,
            "drift_top6": drift, "wer": wer,
        })

    if len(pairs) < 5:
        print("Insufficient data for correlation analysis")
        return

    drifts = [p["drift_top6"] for p in pairs]
    wers = [p["wer"] for p in pairs]

    spearman_r, spearman_p = stats.spearmanr(drifts, wers)
    pearson_r, pearson_p = stats.pearsonr(drifts, wers)

    summary = {
        "budget": args.budget_filter,
        "n": len(pairs),
        "spearman_rho": round(spearman_r, 3),
        "spearman_p": round(spearman_p, 6),
        "pearson_r": round(pearson_r, 3),
        "pearson_p": round(pearson_p, 6),
        "pairs": pairs,
    }

    with open(os.path.join(args.output_dir, f"drift_wer_{args.budget_filter}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f}), n={len(pairs)}")
    print(f"Pearson  r={pearson_r:.3f} (p={pearson_p:.4f})")


if __name__ == "__main__":
    main()
