#!/usr/bin/env python3
"""
Statistical significance tests: Wilcoxon signed-rank + Bonferroni correction.
"""

import json
import glob
import argparse
import numpy as np
from scipy import stats
from collections import defaultdict


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--output", default="analysis/stat_tests.json")
    args = parser.parse_args()

    data = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f"{args.experiments_dir}/**/results*.json", recursive=True):
        with open(f) as fh:
            r = json.load(fh)
        key = (r.get("model_family",""), r.get("language",""), r.get("budget",""))
        method = r.get("adaptation","")
        data[key][method].append(r["test_wer"])

    results = []
    for key, methods in sorted(data.items()):
        model, lang, budget = key
        if "full_ft" not in methods:
            continue
        full_wers = methods["full_ft"]
        for m in ["lora", "adapter"]:
            if m not in methods:
                continue
            peft_wers = methods[m]
            if len(full_wers) >= 3 and len(peft_wers) >= 3:
                n = min(len(full_wers), len(peft_wers))
                stat, p = stats.wilcoxon(full_wers[:n], peft_wers[:n])
                d = cohens_d(full_wers[:n], peft_wers[:n])
                results.append({
                    "model": model, "language": lang, "budget": budget,
                    "comparison": f"full_ft vs {m}",
                    "wilcoxon_stat": round(stat, 4),
                    "p_value": round(p, 6),
                    "cohens_d": round(d, 3),
                    "n": n,
                })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} test results")


if __name__ == "__main__":
    main()
