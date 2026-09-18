#!/usr/bin/env python3
"""Score a checkpoint on the held-out split. Owner: Person C.

    python src/validate.py --data data/train --ckpt runs/v1/best.pt

Prints PSNR / SSIM / LPIPS and milliseconds per image, and appends a row to
results.csv. Every row carries the git commit and the sha1 of the weights, so a
number can always be traced back to the code and the checkpoint that produced it.
That csv becomes slide 6 -- run it after every training run, including the bad
ones.
"""
import argparse
import csv
import hashlib
import os
import subprocess
import sys
import time

import numpy as np
import torch


def _nm(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float('nan')


def git_sha():
    """Short commit of the working tree this run came from, '+dirty' if edited.

    A row without this is a row you cannot reproduce. Never blank it out.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "no-git"
        sha = sha.stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return sha + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "no-git"


def weight_sha1(path):
    """sha1 of the checkpoint file. Identical size != identical model --
    four files on this project are 5,493,743 bytes and three are wrong.
    """
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "unreadable"
    return h.hexdigest()


HEADER = ["tag", "ckpt", "epoch", "psnr", "ssim", "lpips", "ms_per_img",
          "params_M", "git_sha", "weight_sha1"]


def _migrate_csv(path):
    """One-time widen of a pre-provenance results.csv, so appends stay square.

    Old rows get empty git_sha / weight_sha1 -- which is the honest answer:
    nobody recorded where they came from.
    """
    if not os.path.isfile(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == HEADER or rows[0][:1] != ["tag"]:
        return
    pad = len(HEADER) - len(rows[0])
    if pad <= 0:
        return
    rows[0] = HEADER
    rows[1:] = [r + [""] * (len(HEADER) - len(r)) for r in rows[1:]]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"widened {path}: added {', '.join(HEADER[-pad:])}")
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ValDataset, make_split, make_block_split      # noqa: E402
from metrics import lpips, psnr, ssim           # noqa: E402
from model import Restorer                      # noqa: E402

# --------------------------------------------------------------------------
# Architecture-aware model construction. Every scoring tool must build the model
# the way the SHIPPED entry point builds it -- run.build_model() reads the
# checkpoint's own config (and, failing that, sniffs the state_dict) and returns
# the right class. Constructing `Restorer(**config)` directly is how a tool ends
# up silently scoring a DIFFERENT architecture than the one being shipped, or
# crashing on an unexpected kwarg like kind='gate'.
# --------------------------------------------------------------------------
def _build_arch(ck):
    """-> (eval()-mode model on CPU, arch name). Strict load: no silent drift."""
    import os as _os, sys as _sys
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import run as _run
    _st = ck.get("state_dict", ck)
    _m, _arch = _run.build_model(ck.get("config", {}) or {}, _st)
    _m.load_state_dict(_st, strict=True)
    return _m.eval(), _arch



def _tta(model, t):
    """8-fold dihedral self-ensemble. Mirrors inference.py exactly."""
    vs = [t, t.flip(-1), t.flip(-2), t.flip(-1, -2), t.transpose(-1, -2),
          t.transpose(-1, -2).flip(-1), t.transpose(-1, -2).flip(-2),
          t.transpose(-1, -2).flip(-1, -2)]
    inv = [lambda o: o, lambda o: o.flip(-1), lambda o: o.flip(-2), lambda o: o.flip(-1, -2),
           lambda o: o.transpose(-1, -2), lambda o: o.flip(-1).transpose(-1, -2),
           lambda o: o.flip(-2).transpose(-1, -2), lambda o: o.flip(-1, -2).transpose(-1, -2)]
    return torch.stack([inv[k](model(v)) for k, v in enumerate(vs)]).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-range", default="",
                   help="score on a contiguous id range instead of the random split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--tag", default="")
    p.add_argument("--baseline", action="store_true", help="also score plain bicubic")
    p.add_argument("--tta", action="store_true",
                   help="8x dihedral self-ensemble, same transform set as inference.py --tta")
    a = p.parse_args()

    device = torch.device(a.device)
    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    if a.holdout_range:
        lo, hi = (int(v) for v in a.holdout_range.split("-"))
        _, val_ids = make_block_split(gt_dir, lo, hi)
        print(f"scoring on held-out id block {lo}-{hi} ({len(val_ids)} images)")
    else:
        _, val_ids = make_split(gt_dir, n_val=a.n_val, seed=a.seed)
    dl = DataLoader(ValDataset(gt_dir, lr_dir, val_ids), batch_size=1, num_workers=0)

    ck = torch.load(a.ckpt, map_location=device)
    model, _arch = _build_arch(ck)
    model = model.to(device)
    print(f"  arch={_arch}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    rows = {"model": [[], [], [], []]}
    if a.baseline:
        rows["bicubic"] = [[], [], [], []]

    with torch.no_grad():
        for lr, gt, _ in dl:
            lr, gt = lr.to(device), gt.to(device)
            t0 = time.perf_counter()
            out = (_tta(model, lr) if a.tta else model(lr)).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            for arr, pred in [("model", out)] + ([("bicubic", torch.nn.functional.interpolate(
                    lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1))] if a.baseline else []):
                r = rows[arr]
                r[0].append(psnr(pred, gt))
                r[1].append(ssim(pred, gt))
                r[2].append(lpips(pred, gt, device=device))
                r[3].append(dt if arr == "model" else 0.0)

    print(f"\n{'what':10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'ms/img':>8}")
    for k, r in rows.items():
        print(f"{k:10} {np.mean(r[0]):8.3f} {np.mean(r[1]):8.4f} {_nm(r[2]):8.4f} {np.mean(r[3]):8.2f}")

    gsha, wsha = git_sha(), weight_sha1(a.ckpt)
    print(f"\ngit {gsha}   weights {wsha}")

    _migrate_csv(a.csv)
    new = not os.path.isfile(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        r = rows["model"]
        w.writerow([(a.tag or os.path.basename(os.path.dirname(a.ckpt))) + ("+tta" if a.tta else ""),
                    a.ckpt, ck.get("epoch", -1),
                    round(float(np.mean(r[0])), 4), round(float(np.mean(r[1])), 5),
                    round(float(_nm(r[2])), 5), round(float(np.mean(r[3])), 3),
                    round(model.n_params() / 1e6, 3),
                    gsha, wsha])
    print(f"\nappended to {a.csv}")


if __name__ == "__main__":
    main()
