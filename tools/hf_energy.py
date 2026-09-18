#!/usr/bin/env python3
"""Does the model INVENT detail? Measured, not asserted.

    python tools/hf_energy.py --data <train-root> --ckpts runs/*/last.pt

We refuse GAN and perceptual losses on the grounds that invented texture on an
inspection image is a fabricated defect. That has been an argument. This makes
it a number.

A 2x downsample destroys every spatial frequency above the low-resolution
Nyquist limit. Those frequencies are GONE -- any energy the model puts back
there is inferred, not recovered. So compare, in exactly that band:

    ratio = HF energy of the OUTPUT / HF energy of the GROUND TRUTH

    ratio ~ 1.0   the model reconstructs about as much fine detail as truly
                  exists. Ideal.
    ratio < 1.0   under-sharpened: blurry, but everything it draws is real.
                  The safe failure for a fab.
    ratio > 1.0   it is generating MORE fine structure than the ground truth
                  contains. That is hallucination, and on an inspection image
                  it is a defect that was never there.

Bicubic is included as the floor: it invents nothing, so its ratio shows how
much of the band is genuinely unrecoverable.

This is what makes an LPIPS term safe to TEST rather than refuse outright -- add
the term, watch this number, and reject any weight that pushes it past 1.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import Restorer                                    # noqa: E402
from dataset import list_ids, make_split                      # noqa: E402

_R = {}


def hf_fraction(x, cut=0.5):
    """Energy above `cut` x Nyquist, as a fraction of total non-DC energy.

    cut=0.5 is exactly the low-resolution Nyquist limit for a 2x downsample --
    the band the degradation destroyed and the model must infer.
    """
    x = np.asarray(x, dtype=np.float64)
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    H, W = F2.shape
    if (H, W) not in _R:
        y, xx = np.ogrid[:H, :W]
        _R[(H, W)] = np.sqrt(((y - H // 2) / (H // 2)) ** 2 + ((xx - W // 2) / (W // 2)) ** 2)
    r = _R[(H, W)]
    tot = F2.sum()
    return float(F2[r > cut].sum() / tot) if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cut", type=float, default=0.5)
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    lr_dir = os.path.join(a.data, "NoisyLR")
    _, val_ids = make_split(gt_dir, n_val=200, seed=0)
    val_ids = val_ids[:a.n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(val_ids)} held-out images, band above {a.cut:.2f}x Nyquist, device={dev}\n")

    gts, lrs = [], []
    for i in val_ids:
        g = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float32)
        l = np.load(os.path.join(lr_dir, i + ".npy")).astype(np.float32)
        gts.append(g[..., 0] if g.ndim == 3 else g)
        lrs.append(l[..., 0] if l.ndim == 3 else l)

    gt_hf = np.array([hf_fraction(g, a.cut) for g in gts])

    rows = []
    # bicubic floor -- invents nothing by construction
    bic = []
    for l, g in zip(lrs, gts):
        t = torch.from_numpy(l)[None, None]
        up = F.interpolate(t, size=g.shape, mode="bicubic", align_corners=False)
        bic.append(hf_fraction(up[0, 0].numpy().clip(0, 1), a.cut))
    rows.append(("bicubic (floor)", np.array(bic)))

    for ck in a.ckpts:
        tag = os.path.basename(os.path.dirname(ck)) or os.path.basename(ck)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        m = Restorer(**(d.get("config", {}) or {})).eval().to(dev)
        m.load_state_dict(d.get("state_dict", d))
        out = []
        with torch.inference_mode():
            for l, g in zip(lrs, gts):
                t = torch.from_numpy(l)[None, None].to(dev)
                y = m(t).float().clamp(0, 1)[0, 0].cpu().numpy()
                out.append(hf_fraction(y, a.cut))
        rows.append((tag, np.array(out)))

    # AGGREGATE ratio (total energy / total energy), not the mean of per-image
    # ratios. Per-image ratios divide by HF(gt), which on a very smooth image is
    # near zero -- so a trace of interpolation ringing produces a huge quotient
    # and a p95 that says "hallucinating" about bicubic, which is a fixed linear
    # filter and cannot invent anything. The aggregate has no such blow-up, and
    # the median per-image ratio is reported beside it as a robust check.
    print(f"{'model':<20}{'HF(out)':>10}{'HF(gt)':>10}{'aggregate':>11}{'median':>9}"
          f"{'>1':>6}   verdict")
    print("-" * 86)
    for tag, hf in rows:
        agg = float(hf.sum() / max(gt_hf.sum(), 1e-12))
        per = hf / np.maximum(gt_hf, 1e-12)
        med = float(np.median(per))
        over = int((per > 1.0).sum())
        if agg > 1.05:
            v = "HALLUCINATING -- invents structure"
        elif agg > 1.00:
            v = "at the line -- watch it"
        elif agg > 0.60:
            v = "safe -- close to truth, draws no more"
        else:
            v = "safe -- conservative, under-draws"
        print(f"{tag:<20}{hf.mean():>10.4f}{gt_hf.mean():>10.4f}{agg:>11.3f}{med:>9.3f}"
              f"{over:>6}   {v}")
    print("\naggregate > 1 means the model puts MORE energy above the LR Nyquist than")
    print("the ground truth contains -- detail it made up. Under 1 means it draws less")
    print("fine structure than truly exists, which is the safe direction for inspection.")
    print("'>1' counts individual images over the line; a handful on very smooth images")
    print("is ringing, not fabrication.")


if __name__ == "__main__":
    main()
