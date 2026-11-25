#!/usr/bin/env python3
"""
Generate publication-quality figures from experiment data.
Requires matplotlib, seaborn.
"""

import os
import json
import glob
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_scaling_curves(experiments_dir, output_dir):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True)
    models = ["xlsr_300m", "whisper_small", "mms_300m"]
    languages = ["cy", "eu", "mt"]
    methods = ["full_ft", "lora_r8", "adapter_m64", "last2_ft"]
    budgets_min = [10, 60, 180]

    colors = {"full_ft": "#d62728", "lora_r8": "#1f77b4",
              "adapter_m64": "#2ca02c", "last2_ft": "#ff7f0e"}

    for i, lang in enumerate(languages):
        for j, model in enumerate(models):
            ax = axes[i][j]
            for method in methods:
                wers = []
                for budget in ["10min", "1h", "3h"]:
                    pattern = f"{experiments_dir}/{model}/{lang}/{budget}/{method}/results*.json"
                    files = glob.glob(pattern)
                    if files:
                        with open(files[0]) as f:
                            wers.append(json.load(f)["test_wer"])
                if wers:
                    ax.plot(budgets_min, wers, "o-", color=colors[method],
                            label=method.replace("_", " "), linewidth=1.5)

            ax.set_title(f"{model} / {lang}", fontsize=10)
            if i == 2: ax.set_xlabel("Budget (min)")
            if j == 0: ax.set_ylabel("WER (%)")
            ax.set_xscale("log")

    axes[0][2].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_scaling_curves.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_cka_heatmaps(cka_dir, output_dir):
    if not HAS_MPL:
        return
    # Implementation omitted for brevity; uses analysis/cka/*.json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--cka_dir", default="analysis/cka/")
    parser.add_argument("--output_dir", default="figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_scaling_curves(args.experiments_dir, args.output_dir)
    print("Figures generated")


if __name__ == "__main__":
    main()
