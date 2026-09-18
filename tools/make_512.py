#!/usr/bin/env python3
"""Turn NFFA-EUROPE natives into 512->256 pairs in KLA's exact format.

    python tools/make_512.py --src ../nffa --out ../data512 --check
    python tools/make_512.py --src ../nffa --out ../data512 --per-image 2

The round-2 training set contains no 512x512 pairs at all, so the 256->512 path
named in the problem statement is untrained. The NFFA natives are 1024x768,
which is big enough to cut REAL 512 crops -- not upscaled fakes.

Two things make this non-trivial:

  1. Every image carries an instrument DATA BAR at the bottom (a near-black
     separator row, then grey rows of text, magnification and scale bar). Its
     position moves from image to image. Cropping blind would put typography in
     the training data. We detect the separator per image and cut above it.

  2. KLA's format is specific: grayscale, per-image min-max to exactly [0,1],
     float32 .npy. Anything else and the pairs are not comparable to theirs.

Writes GT/ (512x512) and NoisyLR/ (256x256) using the calibrated generator.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from degrade import degrade_at, OBSERVED, GAMMA_K            # noqa: E402

SAFE_ROWS = 620          # fallback content height if no separator is found
MARGIN    = 14           # rows discarded above the separator, for the bright lip


def content_height(a):
    """Rows of real SEM image, above the instrument data bar.

    The bar is introduced by a near-black full-width separator low in the frame.
    We take the HIGHEST such row (banners never sit above the midpoint) and back
    off a little, because a bright strip usually sits just above the line.
    """
    H = a.shape[0]
    lo = int(H * 0.55)
    rows = a[lo:]
    dark = np.where((rows.mean(axis=1) < 25) & (rows.var(axis=1) < 400))[0]
    if len(dark):
        return max(lo + int(dark[0]) - MARGIN, 0)
    return min(SAFE_ROWS, H)


def to_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def norm01(x):
    """KLA's normalisation: per-image min-max to exactly [0, 1]."""
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x, dtype=np.float32) if hi <= lo else ((x - lo) / (hi - lo)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../nffa")
    ap.add_argument("--out", default="../data512")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="report detected crops and exit, writing nothing")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.src, "*", "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    if not files:
        print(f"no images under {a.src}"); return
    print(f"{len(files)} source images in {len(set(os.path.dirname(f) for f in files))} categories")

    if a.check:
        print(f"\n{'file':<40}{'size':>12}{'content rows':>14}{'512 fits?':>11}")
        print("-" * 78)
        heights = []
        for f in files[::max(1, len(files) // 25)][:25]:
            g = to_gray(f); h = content_height(g); heights.append(h)
            print(f"{os.path.basename(f)[:38]:<40}{f'{g.shape[1]}x{g.shape[0]}':>12}"
                  f"{h:>14}{'yes' if h >= a.size else 'NO':>11}")
        h = np.array(heights)
        print(f"\ncontent rows: min {h.min()}  median {int(np.median(h))}  max {h.max()}")
        bad = (h < a.size).sum()
        print(f"{bad}/{len(h)} sampled images too short for a {a.size}px crop"
              f"{'  <-- lower --size or accept fewer crops' if bad else ''}")
        return

    gt_dir, lr_dir = os.path.join(a.out, "GT"), os.path.join(a.out, "NoisyLR")
    os.makedirs(gt_dir, exist_ok=True); os.makedirs(lr_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    S, n, skipped = a.size, 0, 0

    for f in files:
        g = to_gray(f)
        H = content_height(g)
        W = g.shape[1]
        if H < S or W < S:
            skipped += 1; continue
        for _ in range(a.per_image):
            y = int(rng.integers(0, H - S + 1))
            x = int(rng.integers(0, W - S + 1))
            hr = norm01(g[y:y + S, x:x + S])
            # noise drawn from the OBSERVED ranges -- this is a TEST set, so it
            # must look like KLA's data, not like our wider training ranges
            lr = degrade_at(hr, rng,
                            float(rng.uniform(*OBSERVED["sigma_add"])),
                            float(rng.uniform(*OBSERVED["sigma_mul"])),
                            float(rng.uniform(*OBSERVED["c"])), GAMMA_K)
            np.save(os.path.join(gt_dir, f"{n:06d}.npy"), hr.astype(np.float32))
            np.save(os.path.join(lr_dir, f"{n:06d}.npy"), lr.astype(np.float32))
            n += 1

    print(f"\nwrote {n} pairs  GT {S}x{S} -> NoisyLR {S//2}x{S//2}   ({skipped} images too short)")
    print(f"  {gt_dir}\n  {lr_dir}")
    if n:
        q = np.load(os.path.join(gt_dir, "000000.npy"))
        r = np.load(os.path.join(lr_dir, "000000.npy"))
        print(f"\nsanity: GT {q.shape} {q.dtype} range [{q.min():.3f}, {q.max():.3f}]")
        print(f"        LR {r.shape} {r.dtype} range [{r.min():.3f}, {r.max():.3f}]")


if __name__ == "__main__":
    main()
