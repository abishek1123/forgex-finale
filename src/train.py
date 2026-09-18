#!/usr/bin/env python3
"""Training. Owner: Person B (Model).

Typical use:
    python src/train.py --data /path/to/train --out runs/v1
    python src/train.py --data /path/to/train --out runs/v1 --resume runs/v1/last.pt
    python src/train.py --data /path/to/train --smoke        # 30-second CPU check

--smoke runs a handful of iterations on a tiny model. Everyone on the team
should be able to run it on a laptop with no GPU; it catches every bug except
"does it converge", which is Person B's problem.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import RestoreDataset, ValDataset, make_split, make_block_split          # noqa: E402
from degrade import TRAIN as DEGRADE_TRAIN                          # noqa: E402
from losses import RestorationLoss                                  # noqa: E402
from metrics import psnr as psnr_fn, ssim as ssim_fn                # noqa: E402
from model import Restorer                                          # noqa: E402


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train", help="folder containing GT/ and NoisyLR/")
    p.add_argument("--out", default="runs/v1")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--iters", type=int, default=500, help="optimiser steps per epoch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--crop", type=int, default=64, help="LR crop size")
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--nb", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p-real", type=float, default=0.3, help="fraction of KLA's own noisy files")
    p.add_argument("--extra-gt", default="",
                   help="directory of EXTRA clean GT images from another source "
                        "(e.g. NFFA crops). No real noisy pairs exist for these, "
                        "so they are always degraded synthetically. Widens CONTENT.")
    p.add_argument("--p-extra", type=float, default=0.0,
                   help="fraction of training draws taken from --extra-gt")
    p.add_argument("--hard-w", type=float, default=0.0,
                   help="content-balanced sampling: tilt draws toward FINE structure "
                        "(high f90), where we are weakest. 0 = uniform, 1 = "
                        "proportional to f90, 2 = proportional to f90 squared. "
                        "Requires --stats-csv.")
    p.add_argument("--stats-csv", default="docs/per_image_stats.csv",
                   help="per-image f90 table from tools/categories.py")
    p.add_argument("--wide-p", type=float, default=0.0,
                   help="of the SYNTHETIC draws, the fraction from the wide OOD "
                        "degradation family (blur, soft kernels, wider ranges). "
                        "0 = calibrated generator only, as before.")
    p.add_argument("--loss", choices=["combo", "charbonnier"], default="combo")
    p.add_argument("--deg-c", type=float, default=None,
                   help="override the detail term c in the degradation model. "
                        "0.15 recentres effective speckle toward the measured real mean.")
    p.add_argument("--w-ssim", type=float, default=0.15, help="weight on the (1 - SSIM) term")
    p.add_argument("--w-lpips", type=float, default=0.0,
                   help="weight on a differentiable LPIPS term. 0 = off (round-1 behaviour).")
    p.add_argument("--w-laplacian", type=float, default=0.0,
                   help="weight on multi-scale Laplacian-pyramid Charbonnier. "
                        "0 = OFF and behaviour is bit-identical to before. The "
                        "gradient term is broadband; this one weights each octave "
                        "separately, which is what the measured HF deficit needs.")
    p.add_argument("--laplacian-levels", type=int, default=3,
                   help="Laplacian pyramid levels, finest first (weights 1.0/0.5/0.25)")
    p.add_argument("--w-grad", type=float, default=0.05,
                   help="weight on the edge/gradient term. RAISE THIS if outputs look "
                        "over-smoothed: it is the only term fighting blur.")
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-range", default="",
                   help="hold out a CONTIGUOUS id range as validation, e.g. 2868-3345. "
                        "The ids are ordered by content, so this is an out-of-distribution "
                        "test rather than the interpolation test a random split gives.")
    p.add_argument("--split-seed", type=int, default=0,
                   help="controls the train/val split ONLY. Keep this fixed while varying "
                        "--seed, so two runs are scored on the SAME held-out images.")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.iters, a.batch, a.ch, a.nb = 1, 5, 2, 16, 2
        a.n_val, a.workers, a.device, a.amp = 2, 0, "cpu", False
        a.out = a.out.rstrip("/") + "_smoke"
    return a


@torch.no_grad()
def validate(model, loader, device, limit=None):
    model.eval()
    ps, ss = [], []
    for n, (lr, gt, _) in enumerate(loader):
        if limit and n >= limit:
            break
        out = model(lr.to(device)).clamp(0, 1).cpu()
        ps.append(psnr_fn(out, gt))
        ss.append(ssim_fn(out, gt))
    model.train()
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = pick_device(a.device)

    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    for d in (gt_dir, lr_dir):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: {d} not found. --data must point at the folder holding GT/ and NoisyLR/")

    if a.holdout_range:
        lo, hi = (int(v) for v in a.holdout_range.split("-"))
        train_ids, val_ids = make_block_split(gt_dir, lo, hi)
        print(f"OUT-OF-DISTRIBUTION split: holding out ids {lo}-{hi} entirely")
    else:
        train_ids, val_ids = make_split(gt_dir, n_val=a.n_val, seed=a.split_seed)
    if a.smoke:
        train_ids, val_ids = train_ids[:8], val_ids[:2]
    print(f"device={device}  train={len(train_ids)}  val={len(val_ids)}")

    deg_cfg = dict(DEGRADE_TRAIN)
    if a.deg_c is not None:
        deg_cfg["c"] = (a.deg_c, a.deg_c)
        print(f"degradation override: c={a.deg_c}")
    train_ds = RestoreDataset(gt_dir, lr_dir, train_ids, crop=a.crop, p_real=a.p_real,
                              cfg=deg_cfg, seed=a.seed, length=a.iters * a.batch, wide_p=a.wide_p,
                              extra_gt=(a.extra_gt or None), p_extra=a.p_extra,
                              hard_w=a.hard_w, stats_csv=a.stats_csv)
    val_ds = ValDataset(gt_dir, lr_dir, val_ids)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          pin_memory=(device.type == "cuda"), drop_last=True,
                          persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = Restorer(ch=a.ch, nb=a.nb).to(device)
    print(f"parameters: {model.n_params()/1e6:.2f} M")
    crit = RestorationLoss(mode=a.loss, w_ssim=a.w_ssim, w_grad=a.w_grad,
                           w_lpips=a.w_lpips, w_laplacian=a.w_laplacian,
                           laplacian_levels=a.laplacian_levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * a.iters, eta_min=a.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    start_epoch, best = 0, -1.0
    if a.resume and os.path.isfile(a.resume):
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from {a.resume} at epoch {start_epoch} (best PSNR {best:.3f})")

    log_path = os.path.join(a.out, "log.csv")
    new_log = not os.path.isfile(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "lr", "secs"])

    for epoch in range(start_epoch, a.epochs):
        t0, running, skipped = time.time(), 0.0, 0
        for i, (lr_b, hr_b) in enumerate(train_dl):
            lr_b, hr_b = lr_b.to(device, non_blocking=True), hr_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # Model forward in mixed precision; LOSS ALWAYS IN FP32. Computing
            # the loss inside the autocast block silently runs SSIM in fp16,
            # where its denominator underflows to Inf and then NaN -- and NaN is
            # scale-invariant, so GradScaler skips every step and nothing trains.
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(lr_b)
            loss, parts = crit(pred.float(), hr_b.float())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            # Exact skip detection: the gradient norm is non-finite for exactly
            # the steps GradScaler refuses. (Do not infer this from the scale --
            # after enough halvings it underflows to 0 and stops changing, which
            # makes a permanently-failing run look healthy.)
            if not torch.isfinite(gnorm):
                skipped += 1
            else:
                sched.step()
            running += float(loss.detach())
            if i % 50 == 0:
                bits = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                print(f"  e{epoch} {i}/{len(train_dl)} loss={float(loss.detach()):.4f} {bits}", flush=True)

        vp, vs = validate(model, val_dl, device)
        secs = time.time() - t0
        if skipped:
            frac = skipped / max(len(train_dl), 1)
            print(f"  WARNING: {skipped}/{len(train_dl)} optimiser steps skipped ({frac:.0%}) "
                  f"- non-finite gradients. A few percent is normal AMP warm-up; "
                  f"anything above ~20% means the model is not training.")
            if scaler.is_enabled() and scaler.get_scale() < 1.0:
                print(f"  WARNING: loss scale has collapsed to {scaler.get_scale():.3g} "
                      f"- persistent NaN gradients, not recoverable overflow.")
            if frac > 0.9:
                sys.exit("ABORTING: essentially every step was skipped. Re-run with --no-amp "
                         "to confirm, and check the loss for fp16 overflow/underflow.")
        print(f"epoch {epoch}: loss={running/max(len(train_dl),1):.4f}  val PSNR={vp:.3f} dB  SSIM={vs:.4f}  ({secs:.0f}s)")
        logw.writerow([epoch, running / max(len(train_dl), 1), vp, vs, sched.get_last_lr()[0], round(secs, 1)])
        logf.flush()

        ck = dict(state_dict=model.state_dict(), config=model.config, epoch=epoch, best=max(best, vp),
                  opt=opt.state_dict(), sched=sched.state_dict(), args=vars(a))
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if vp > best:
            best = vp
            torch.save(dict(state_dict=model.state_dict(), config=model.config,
                            epoch=epoch, best=best, args=vars(a)), os.path.join(a.out, "best.pt"))
            print(f"  new best: {best:.3f} dB -> {a.out}/best.pt")

    logf.close()
    print(f"done. best val PSNR = {best:.3f} dB")


if __name__ == "__main__":
    main()
