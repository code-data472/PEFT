#!/usr/bin/env python3
"""
Prepare Common Voice 17.0 datasets: download, resample, normalise text,
verify speaker disjointness between splits.
"""

import os
import re
import sys
import json
import argparse
import unicodedata
from collections import defaultdict

from datasets import load_dataset, Audio


def normalise_text(text, language):
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def verify_speaker_disjointness(train_ds, val_ds, test_ds):
    train_spk = set(ex["client_id"] for ex in train_ds)
    val_spk = set(ex["client_id"] for ex in val_ds)
    test_spk = set(ex["client_id"] for ex in test_ds)

    tv = train_spk & val_spk
    tt = train_spk & test_spk
    vt = val_spk & test_spk

    if tv or tt or vt:
        print(f"WARNING: speaker overlap detected")
        print(f"  train∩val: {len(tv)}, train∩test: {len(tt)}, val∩test: {len(vt)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["cy", "eu", "mt"])
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--cv_version", default="17.0")
    args = parser.parse_args()

    for lang in args.languages:
        print(f"\n=== Processing {lang} ===")
        ds_name = f"mozilla-foundation/common_voice_{args.cv_version.replace('.','_')}"

        for split in ["train", "validation", "test"]:
            ds = load_dataset(ds_name, lang, split=split, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            print(f"  {split}: {len(ds)} utterances")

            total_dur = sum(len(ex["audio"]["array"]) / 16000 for ex in ds)
            print(f"  Duration: {total_dur/3600:.2f}h")

        print(f"  Speaker disjointness: ", end="")
        train = load_dataset(ds_name, lang, split="train", trust_remote_code=True)
        val = load_dataset(ds_name, lang, split="validation", trust_remote_code=True)
        test = load_dataset(ds_name, lang, split="test", trust_remote_code=True)
        ok = verify_speaker_disjointness(train, val, test)
        print("OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
