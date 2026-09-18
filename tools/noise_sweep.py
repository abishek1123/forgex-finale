#!/usr/bin/env python3
"""How does a checkpoint hold up as the noise level moves away from what it was trained on?

    python tools/noise_sweep.py --data <train-root> --ckpt runs/<run>/last.pt

The organisers confirmed the hidden test set varies BOTH content and noise level.
A single number at the nominal noise level therefore says almost nothing about the
score. This sweeps sigma_mul across a grid, re-degrading real ground truth with our
calibrated generator, and reports PSNR/SSIM/LPIPS at each level against a bicubic
baseline -- so we can see where a model degrades gracefully and where it falls off.

Writes noise_sweep_<tag>.csv. Owner: Person A (Data).
"""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from src.model import Restorer

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

from src.degrade import box_down, degrade_at, OBSERVED
from metrics import ssim as m_ssim, lpips as m_lpips


# degrade_at now comes from src/degrade.py -- the rebuilt, data-fitted model.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing GT/")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=60, help="ground-truth images per noise level")
    p.add_argument("--s-add", type=float, default=0.0449, help="additive sigma (round-2 measured)")
    p.add_argument("--c", type=float, default=0.1606, help="detail term (round-2 measured mean)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grid", default="0.05,0.10,0.15,0.19,0.25,0.30,0.35,0.40,0.45")
    a = p.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu")
    m, arch = _build_arch(ck); m = m.to(dev)
    print(f"  arch={arch}  params={sum(p.numel() for p in m.parameters()):,}")

    gts = [np.load(f).astype(np.float32)
           for f in sorted(glob.glob(os.path.join(a.data, "GT", "*.npy")))[: a.n]]
    grid = [float(v) for v in a.grid.split(",")]
    tag = a.tag or os.path.basename(os.path.dirname(a.ckpt))

    print(f"{len(gts)} images · sigma_add={a.s_add} · checkpoint {a.ckpt}\n")
    print(f"{'sigma_mul':>10}{'bicubic':>10}{'model':>9}{'gain':>8}{'SSIM':>9}{'LPIPS':>9}   in training range?")
    rows = []
    for s in grid:
        pm, pb, sm, lp = [], [], [], []
        for i, gt in enumerate(gts):
            lr = degrade_at(gt, np.random.default_rng(1000 + i), a.s_add, s, a.c)
            t = torch.from_numpy(lr)[None, None].to(dev)
            with torch.inference_mode():
                o = m(t).clamp(0, 1)
                b = F.interpolate(t.float(), scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
            g = torch.from_numpy(gt)[None, None].to(dev)
            mse = lambda x: float(torch.mean((x - g) ** 2))
            pm.append(10 * np.log10(1 / max(mse(o), 1e-12)))
            pb.append(10 * np.log10(1 / max(mse(b), 1e-12)))
            sm.append(float(m_ssim(o, g)))
            if i < 20: lp.append(m_lpips(o, g, device=dev))
        lpv = float(np.nanmean(lp)) if lp else float("nan")
        inr = "yes" if 0.080 <= s <= 0.260 else "OUT OF RANGE"
        real = "   <- KLA's real level" if abs(s - 0.16) < 0.011 else ""
        print(f"{s:>10.2f}{np.mean(pb):>10.2f}{np.mean(pm):>9.2f}{np.mean(pm)-np.mean(pb):>+8.2f}"
              f"{np.mean(sm):>9.4f}{lpv:>9.4f}   {inr}{real}")
        rows.append((s, np.mean(pb), np.mean(pm), np.mean(sm), lpv))

    out = os.path.join(HERE, "..", f"noise_sweep_{tag}.csv")
    with open(out, "w", newline="") as f:
        f.write("sigma_mul,bicubic_psnr,model_psnr,model_ssim,model_lpips\n")
        for r in rows: f.write(",".join(f"{v:.5f}" for v in r) + "\n")
    print(f"\nwritten to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
