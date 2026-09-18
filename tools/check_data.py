#!/usr/bin/env python3
"""Sanity-check the dataset layout before anyone wastes a night on it.

    python tools/check_data.py --data data/train
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset import list_ids   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    a = p.parse_args()
    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    ok = True
    for d in (gt_dir, lr_dir):
        if not os.path.isdir(d):
            print(f"MISSING: {d}")
            ok = False
    if not ok:
        sys.exit(1)

    g, l = list_ids(gt_dir), list_ids(lr_dir)
    print(f"GT files      : {len(g)}")
    print(f"NoisyLR files : {len(l)}")
    missing = set(g) ^ set(l)
    print(f"unpaired      : {len(missing)}" + (f"  e.g. {sorted(missing)[:5]}" if missing else "  (good)"))

    bad = 0
    for i in g[:50]:
        gt = np.load(os.path.join(gt_dir, i + ".npy"))
        lr = np.load(os.path.join(lr_dir, i + ".npy"))
        if gt.shape != tuple(2 * s for s in lr.shape):
            print(f"  SHAPE MISMATCH {i}: GT{gt.shape} vs LR{lr.shape}")
            bad += 1
    gt = np.load(os.path.join(gt_dir, g[0] + ".npy"))
    lr = np.load(os.path.join(lr_dir, g[0] + ".npy"))
    print(f"example       : GT {gt.shape} {gt.dtype} [{gt.min():.3f},{gt.max():.3f}]")
    print(f"                LR {lr.shape} {lr.dtype} [{lr.min():.3f},{lr.max():.3f}]")
    print("\nOK" if not bad and not missing else "\nPROBLEMS FOUND -- fix before training")


if __name__ == "__main__":
    main()
