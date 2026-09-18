#!/usr/bin/env python3
"""Before/after figure for the slide deck.

    python tools/preview.py --data data/train --ckpt runs/v1/best.pt --n 4
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset import make_split          # noqa: E402
from metrics import psnr                # noqa: E402
from model import Restorer              # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", default="preview.png")
    a = p.parse_args()

    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    _, val_ids = make_split(gt_dir, seed=0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    model = Restorer(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])

    fig, ax = plt.subplots(a.n, 4, figsize=(13, 3.3 * a.n))
    ax = np.atleast_2d(ax)
    for r, name in enumerate(val_ids[: a.n]):
        lr = np.load(os.path.join(lr_dir, name + ".npy")).astype(np.float32)
        gt = np.load(os.path.join(gt_dir, name + ".npy")).astype(np.float32)
        t = torch.from_numpy(lr)[None, None].to(dev)
        with torch.no_grad():
            out = model(t).clamp(0, 1)
        bic = torch.nn.functional.interpolate(t, scale_factor=2, mode="bicubic",
                                              align_corners=False).clamp(0, 1)
        g = torch.from_numpy(gt)[None, None]
        panels = [(lr, "degraded input"),
                  (bic[0, 0].cpu().numpy(), f"bicubic  {psnr(bic.cpu(), g):.2f} dB"),
                  (out[0, 0].cpu().numpy(), f"ours  {psnr(out.cpu(), g):.2f} dB"),
                  (gt, "ground truth")]
        for c, (img, title) in enumerate(panels):
            ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(title, fontsize=11)
            else:
                ax[r, c].set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
