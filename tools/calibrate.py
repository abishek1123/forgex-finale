#!/usr/bin/env python3
"""Does our synthetic damage match KLA's real damage? Owner: Person A.

    python tools/calibrate.py --data data/train

Fits  var(residual) = sigma_add^2 + sigma_mul^2 * I^2  on (a) KLA's real pairs
and (b) pairs we generate ourselves, and prints both. If the two sets of numbers
agree, our degradation model is faithful and we are entitled to train on
synthetic data. This output is the evidence for the Data block on slide 8.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset import list_ids            # noqa: E402
from degrade import box_down, degrade, OBSERVED, TRAIN   # noqa: E402


def fit(clean, resid, nbins=18):
    """Least squares for var = a + b*I^2 -> (sigma_add, sigma_mul)."""
    c, r = clean.ravel(), resid.ravel()
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(c, edges) - 1
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 300:
            continue
        xs.append(c[m].mean())
        ys.append(r[m].var())
    if len(xs) < 5:
        return None
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs ** 2]).T
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(np.sqrt(max(sol[0], 0))), float(np.sqrt(max(sol[1], 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    p.add_argument("--n", type=int, default=60)
    a = p.parse_args()
    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    ids = list_ids(gt_dir)[: a.n]
    rng = np.random.default_rng(0)

    for label, cfg in [("KLA real pairs", None), ("ours (OBSERVED)", OBSERVED), ("ours (TRAIN)", TRAIN)]:
        adds, muls = [], []
        for i in ids:
            gt = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float32)
            clean = box_down(gt)
            lr = np.load(os.path.join(lr_dir, i + ".npy")).astype(np.float32) if cfg is None \
                else degrade(gt, rng, cfg)
            f = fit(clean, lr - clean)
            if f:
                adds.append(f[0])
                muls.append(f[1])
        print(f"{label:18} sigma_add {np.mean(adds):.4f} [{np.min(adds):.3f}-{np.max(adds):.3f}]   "
              f"sigma_mul {np.mean(muls):.4f} [{np.min(muls):.3f}-{np.max(muls):.3f}]   (n={len(adds)})")
    print("\nIf 'ours (OBSERVED)' brackets 'KLA real pairs', the degradation model is faithful.")
    print("'ours (TRAIN)' should be WIDER on purpose -- that is what buys out-of-distribution robustness.")


if __name__ == "__main__":
    main()
