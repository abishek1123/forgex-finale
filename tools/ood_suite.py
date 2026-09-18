#!/usr/bin/env python3
"""Score a model on FOUR out-of-distribution axes, not just one.

    python tools/ood_suite.py --data <train-root> --ckpts runs/*/last.pt

Our OOD evidence has been a sigma sweep -- one axis. But the test set is "from
different sources", and a different instrument differs in more than noise level.
We even built blur and a soft downsample kernel into the WIDE training family
and then never tested a model on blurred input. This closes that.

  noise    sigma above and below the observed 0.155-0.258 band
  blur     optics differ: Gaussian blur applied to the clean image before
           sampling, sigma 0.4 / 0.8 / 1.2
  kernel   a Gaussian-weighted 2x2 readout instead of KLA's exact box average
  content  the coarsest and finest content regions, measured -- ids 1850-2134
           (f90 0.26) against 0-1534 (f90 0.74)

Every axis is scored as GAIN OVER BICUBIC on the same inputs, so a hard axis
does not look like a bad model. Bicubic is recomputed per axis.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import Restorer                                           # noqa: E402

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

from degrade import degrade_at, block_stats, _unit_skewed, _blur, _soft_down, GAMMA_K  # noqa: E402

S_ADD, C_DET = 0.0449, 0.1606          # measured round-2 means
BAND = 0.19                             # KLA's real speckle level


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(1.0 / max(m, 1e-20))


def bicubic(lr, hw):
    t = torch.from_numpy(lr)[None, None].float()
    return F.interpolate(t, size=hw, mode="bicubic", align_corners=False)[0, 0].clamp(0, 1).numpy()


def make(gt, rng, kind, level):
    """Degrade `gt` under one OOD condition. Returns the low-res input."""
    x = np.asarray(gt, dtype=np.float64)
    if kind == "blur":
        x = _blur(x, level)
        s_mul = BAND
    elif kind == "kernel":
        m = _soft_down(x)
        _, v = block_stats(x)
        var = S_ADD ** 2 + BAND ** 2 * (m * m) + C_DET * v
        return np.ascontiguousarray(m + np.sqrt(np.maximum(var, 0)) *
                                    _unit_skewed(rng, m.shape, GAMMA_K), dtype=np.float32)
    else:                                # noise, or content at the real level
        s_mul = level
    return degrade_at(x, rng, S_ADD, s_mul, C_DET, GAMMA_K)


AXES = [
    ("noise 0.05",  "noise",  0.05),
    ("noise 0.19",  "noise",  0.19),
    ("noise 0.40",  "noise",  0.40),
    ("blur 0.4",    "blur",   0.4),
    ("blur 0.8",    "blur",   0.8),
    ("blur 1.2",    "blur",   1.2),
    ("soft kernel", "kernel", 0.0),
]
BLOCKS = {"content coarse": (1850, 2134), "content fine": (0, 1534)}


def load_model(ck, dev):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m, arch = _build_arch(d)
    print(f"  arch={arch}")
    return m.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40, help="images per axis")
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    ids = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".npy"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def pick(lo=None, hi=None):
        pool = [i for i in ids if lo is None or lo <= int(i) <= hi]
        return pool[:: max(1, len(pool) // a.n)][:a.n]

    jobs = [(name, kind, lvl, pick()) for name, kind, lvl in AXES]
    # Content axes are defined by TRAINING-set id ranges. On any other dataset
    # (e.g. the organisers' 297-image test set, ids 0-296) those ranges are
    # empty, which used to produce a NaN row and a NaN MEAN. Skip them, and say
    # so, rather than reporting a number that is not a number.
    skipped = []
    for name, (lo, hi) in BLOCKS.items():
        sel = pick(lo, hi)
        if len(sel) < max(4, a.n // 4):
            skipped.append(f"{name} (ids {lo}-{hi}: only {len(sel)} images here)")
            continue
        jobs.append((name, "noise", BAND, sel))
    if skipped:
        print("skipping content axes -- id range not present in this dataset:")
        for x in skipped:
            print("   " + x)
        print()

    print(f"{len(jobs)} axes x {a.n} images, device={dev}\n")
    models = {os.path.basename(os.path.dirname(c)) or c: load_model(c, dev) for c in a.ckpts}

    table = {}
    for name, kind, lvl, sel in jobs:
        rng = np.random.default_rng(0)              # same inputs for every model
        pairs = []
        for i in sel:
            g = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float32)
            if g.ndim == 3: g = g[..., 0]
            pairs.append((g, make(g, rng, kind, lvl)))
        bic = float(np.mean([psnr(bicubic(l, g.shape), g) for g, l in pairs]))
        row = {"bicubic": bic}
        for tag, m in models.items():
            vals = []
            with torch.inference_mode():
                for g, l in pairs:
                    y = m(torch.from_numpy(l)[None, None].to(dev)).float().clamp(0, 1)
                    vals.append(psnr(y[0, 0].cpu().numpy(), g))
            row[tag] = float(np.mean(vals)) - bic          # GAIN over bicubic
        table[name] = row
        print(f"  {name:<15} bicubic {bic:6.2f}   " +
              "  ".join(f"{t} {row[t]:+.2f}" for t in models), flush=True)

    names = list(models)
    print("\n" + "=" * (20 + 12 * len(names)))
    print(f"{'axis':<16}{'bicubic':>9}" + "".join(f"{t:>12}" for t in names))
    print("-" * (20 + 12 * len(names)))
    for name, row in table.items():
        print(f"{name:<16}{row['bicubic']:>9.2f}" + "".join(f"{row[t]:>+12.2f}" for t in names))
    print("-" * (20 + 12 * len(names)))
    means = {t: np.mean([r[t] for r in table.values()]) for t in names}
    worst = {t: min(r[t] for r in table.values()) for t in names}
    print(f"{'MEAN gain':<16}{'':>9}" + "".join(f"{means[t]:>+12.2f}" for t in names))
    print(f"{'WORST axis':<16}{'':>9}" + "".join(f"{worst[t]:>+12.2f}" for t in names))
    print("=" * (20 + 12 * len(names)))
    best = max(names, key=lambda t: means[t])
    rob  = max(names, key=lambda t: worst[t])
    print(f"\nbest mean gain across axes: {best}")
    print(f"best worst-case axis:       {rob}"
          f"{'  (same model -- unambiguous)' if best == rob else '  -- differs from best mean; the worst case is what the OOD half punishes'}")


if __name__ == "__main__":
    main()
