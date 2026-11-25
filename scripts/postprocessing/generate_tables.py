#!/usr/bin/env python3
"""
Generate LaTeX tables from aggregated experiment results.
Outputs Table 3 (main WER) and Table 5 (degradation indicators).
"""

import json
import csv
import argparse
import numpy as np
from collections import defaultdict


def format_wer(mean, std=None):
    if std is not None and std > 0:
        return f"{mean:.1f}$\\pm${std:.1f}"
    return f"{mean:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="analysis/aggregated_results.csv")
    parser.add_argument("--output_dir", default="tables/")
    args = parser.parse_args()

    with open(args.results_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by model/language/budget
    grouped = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["language"], r["budget"])
        grouped[key][r["method"]] = {
            "mean": float(r["wer_mean"]),
            "std": float(r["wer_std"]) if r["wer_std"] else None,
        }

    print("% Table 3: Main WER results")
    print("% Auto-generated from experiment data")
    for key in sorted(grouped.keys()):
        model, lang, budget = key
        methods = grouped[key]
        line = f"  {model} & {budget}"
        for m in ["frozen", "last2_ft", "adapter_m64", "lora_r8", "full_ft"]:
            if m in methods:
                line += f" & {format_wer(methods[m]['mean'], methods[m].get('std'))}"
            else:
                line += " & ---"
        line += " \\\\"
        print(line)


if __name__ == "__main__":
    main()
