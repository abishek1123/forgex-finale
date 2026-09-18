#!/usr/bin/env python3
"""Recover the dataset's hidden category structure from the images themselves.

    python tools/categories.py --data <train-root> [--out docs/per_image_stats.csv]

The round-2 brief states the data contains 10 categories of surface morphology.
It does NOT say where the boundaries are. This measures, for every pair:

    f90     radial frequency below which 90% of non-DC energy sits  (fineness)
    var     total residual variance of the noisy input vs the box-averaged GT
    mean    mean brightness
    smul    per-image speckle estimate, from the slope of var against m^2

then looks for step changes in f90 along the id axis. If the ids are grouped by
category -- which 4785/10 = 478.5 suggests -- the steps land on multiples of 478.

numpy only: no torch, no GPU. Safe to run while a training job holds the card.
"""
import argparse, os, sys, time
import numpy as np


_RCACHE = {}


def _radial(shape):
    """Sort order by radius, cached -- identical for every image of a given size,
    and the argsort is what dominates otherwise."""
    if shape not in _RCACHE:
        H, W = shape
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
        order = np.argsort(r, axis=None)
        _RCACHE[shape] = (order, r.flat[order].copy())
    return _RCACHE[shape]


def f90(img):
    """Radial frequency containing 90% of non-DC energy, normalised to Nyquist."""
    F = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
    order, rs = _radial(F.shape)
    c = np.cumsum(F.flat[order])
    if c[-1] <= 0:
        return 0.0
    return float(rs[np.searchsorted(c, 0.9 * c[-1])])


def box_down(x, f=2):
    H, W = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="docs/per_image_stats.csv")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    lr_dir = os.path.join(a.data, "NoisyLR")
    ids = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".npy"))
    ids = ids[a.start::a.stride]
    print(f"{len(ids)} pairs", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    new = not os.path.isfile(a.out)
    with open(a.out, "a") as fh:
        if new:
            fh.write("id,f90,mean,var_resid,smul\n")
        for k, i in enumerate(ids):
            gt = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float64)
            if gt.ndim == 3:
                gt = gt[..., 0]
            m = box_down(gt)
            rec = [i, f90(gt), gt.mean()]
            lr_path = os.path.join(lr_dir, i + ".npy")
            if os.path.isfile(lr_path):
                lr = np.load(lr_path).astype(np.float64)
                if lr.ndim == 3:
                    lr = lr[..., 0]
                d = lr - m
                rec.append(d.var())
                # slope of residual^2 against m^2 -- a crude per-image speckle proxy
                x = (m * m).ravel()
                y = (d * d).ravel()
                xm, ym = x.mean(), y.mean()
                den = ((x - xm) ** 2).mean()
                slope = (((x - xm) * (y - ym)).mean() / den) if den > 0 else 0.0
                rec.append(np.sqrt(max(slope, 0.0)))
            else:
                rec += [float("nan"), float("nan")]
            fh.write("%s,%.6f,%.6f,%.8f,%.6f\n" % tuple(rec))
            if k % 250 == 0:
                fh.flush()
                el = time.perf_counter() - t0
                print(f"  {k}/{len(ids)}  {el:.0f}s  eta {el/max(k,1)*(len(ids)-k):.0f}s", flush=True)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
