#!/usr/bin/env python3
"""Score the model on TEN LABELLED MORPHOLOGY CATEGORIES.

    python tools/per_category.py --src ../nffa --ckpt models/model.pt

"Models should generalize across multiple image categories and distributions."
We tried to recover KLA's category structure from spectral content and failed --
one number (f90) cannot separate ten morphologies, because several classes share
a fineness signature.

But NFFA-EUROPE ships the labels, and its images are a DIFFERENT SOURCE from
KLA's. So each folder is simultaneously a named category and an out-of-source
test. Crop above the instrument data bar, normalise the way KLA does, degrade
with the calibrated generator, and score.

Every category is scored as GAIN OVER BICUBIC on identical inputs, so an
intrinsically hard morphology does not read as a bad model.
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import Restorer                                   # noqa: E402
from degrade import degrade_at, OBSERVED, GAMMA_K            # noqa: E402
from metrics import psnr as _psnr, ssim as _ssim, lpips as _lpips   # noqa: E402

SAFE, MARGIN = 620, 14


def content_height(a):
    H = a.shape[0]; lo = int(H * 0.55); r = a[lo:]
    dark = np.where((r.mean(axis=1) < 25) & (r.var(axis=1) < 400))[0]
    return max(lo + int(dark[0]) - MARGIN, 0) if len(dark) else min(SAFE, H)


def norm01(x):
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x, np.float32) if hi <= lo else ((x - lo) / (hi - lo)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../nffa")
    ap.add_argument("--ckpt", default="models/model.pt")
    ap.add_argument("--n", type=int, default=24, help="crops per category")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()

    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = Restorer(**(d.get("config", {}) or {})).eval().to(dev)
    m.load_state_dict(d.get("state_dict", d))

    cats = sorted(x for x in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, x)))
    print(f"{len(cats)} labelled categories · {a.n} crops each · {a.size}x{a.size} · device={dev}\n")
    print(f"{'category':<30}{'bicubic':>9}{'model':>9}{'gain':>8}{'SSIM':>8}{'LPIPS':>8}")
    print("-" * 72)

    rows = []
    S = a.size
    for cat in cats:
        rng = np.random.default_rng(0)
        fs = sorted(glob.glob(os.path.join(a.src, cat, "*")))
        bp, mp, ss, lp = [], [], [], []
        for f in fs:
            if len(mp) >= a.n:
                break
            g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
            H, W = content_height(g), g.shape[1]
            if H < S or W < S:
                continue
            y = int(rng.integers(0, H - S + 1)); x = int(rng.integers(0, W - S + 1))
            hr = norm01(g[y:y + S, x:x + S])
            lr = degrade_at(hr, rng, float(rng.uniform(*OBSERVED["sigma_add"])),
                            float(rng.uniform(*OBSERVED["sigma_mul"])),
                            float(rng.uniform(*OBSERVED["c"])), GAMMA_K)
            t = torch.from_numpy(lr)[None, None].to(dev)
            gt = torch.from_numpy(hr)[None, None].to(dev)
            with torch.inference_mode():
                out = m(t).float().clamp(0, 1)
            bic = F.interpolate(t, size=(S, S), mode="bicubic", align_corners=False).clamp(0, 1)
            bp.append(_psnr(bic, gt)); mp.append(_psnr(out, gt))
            ss.append(_ssim(out, gt)); lp.append(_lpips(out, gt, device=dev))
        if not mp:
            continue
        b, p2 = float(np.mean(bp)), float(np.mean(mp))
        rows.append((cat, b, p2, p2 - b, float(np.mean(ss)), float(np.nanmean(lp))))
        print(f"{cat[:29]:<30}{b:>9.2f}{p2:>9.2f}{p2-b:>+8.2f}{np.mean(ss):>8.4f}{np.nanmean(lp):>8.4f}",
              flush=True)

    if not rows:
        print("no categories scored"); return
    g = np.array([r[3] for r in rows])
    print("-" * 72)
    print(f"{'MEAN across categories':<30}{'':>9}{'':>9}{g.mean():>+8.2f}")
    print(f"{'WORST category':<30}{'':>9}{'':>9}{g.min():>+8.2f}   {rows[int(g.argmin())][0]}")
    print(f"{'BEST category':<30}{'':>9}{'':>9}{g.max():>+8.2f}   {rows[int(g.argmax())][0]}")
    print(f"{'SPREAD':<30}{'':>9}{'':>9}{g.max()-g.min():>8.2f} dB")
    print("\nEvery category is a DIFFERENT SOURCE from KLA's training images.")
    print("A positive gain in all ten is generalisation across categories AND across sources.")
    with open(os.path.join(ROOT, "docs_tmp", "per_category.csv"), "w") as fh:
        fh.write("category,bicubic_psnr,model_psnr,gain,ssim,lpips\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.5f},{r[5]:.5f}\n")
    print("written to docs_tmp/per_category.csv")


if __name__ == "__main__":
    main()
