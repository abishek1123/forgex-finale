#!/usr/bin/env python3
"""Is the noise Poisson (shot) or speckle (multiplicative)?

    python tools/noise_physics.py --data <train-root> [--n 400]

We fitted   var = a + b*m^2 + c*v   and never tested the alternative. SEM
detector noise is dominated by SHOT noise, which is Poisson: variance grows
LINEARLY with signal, not quadratically. If the data prefers a*m, our whole
"speckle" framing is wrong -- and if it prefers m^2, we can say the noise is
multiplicative rather than photon-limited, which is a physical claim rather
than a curve fit.

Compares four models on identical bins, by adjusted R^2 and BIC:
    M1  a + b*m^2 + c*v          (what we ship)
    M2  a + p*m   + c*v          (Poisson / shot noise)
    M3  a + p*m   + b*m^2 + c*v  (both)
    M4  a + c*v                  (no signal dependence -- the null)

numpy only. No GPU.
"""
import argparse, os
import numpy as np


def box_stats(gt, f=2):
    H, W = (gt.shape[0] // f) * f, (gt.shape[1] // f) * f
    b = gt[:H, :W].reshape(H // f, f, W // f, f)
    m = b.mean(axis=(1, 3))
    v = np.maximum((b ** 2).mean(axis=(1, 3)) - m * m, 0.0)
    return m, v


def fit(X, y, w):
    """Weighted least squares -> (coefs, weighted SSE)."""
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X * W, y * np.sqrt(w), rcond=None)
    r = y - X @ coef
    return coef, float((w * r * r).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--mbins", type=int, default=24)
    ap.add_argument("--vbins", type=int, default=12)
    a = ap.parse_args()

    gt_dir, lr_dir = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    ids = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".npy"))
    step = max(1, len(ids) // a.n)
    ids = ids[::step][:a.n]

    M, V, D = [], [], []
    for i in ids:
        gt = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float64)
        lr = np.load(os.path.join(lr_dir, i + ".npy")).astype(np.float64)
        if gt.ndim == 3: gt = gt[..., 0]
        if lr.ndim == 3: lr = lr[..., 0]
        m, v = box_stats(gt)
        M.append(m.ravel()); V.append(v.ravel()); D.append((lr - m).ravel())
    m = np.concatenate(M); v = np.concatenate(V); d = np.concatenate(D)
    print(f"{len(ids)} images, {len(m):,} low-res pixels")

    # bin in (m, v) and take the residual VARIANCE in each bin
    mq = np.quantile(m, np.linspace(0, 1, a.mbins + 1))
    vq = np.quantile(v, np.linspace(0, 1, a.vbins + 1))
    mi = np.clip(np.searchsorted(mq, m, "right") - 1, 0, a.mbins - 1)
    vi = np.clip(np.searchsorted(vq, v, "right") - 1, 0, a.vbins - 1)
    key = mi * a.vbins + vi
    nb = a.mbins * a.vbins
    cnt = np.bincount(key, minlength=nb).astype(float)
    sm  = np.bincount(key, weights=m, minlength=nb)
    sv  = np.bincount(key, weights=v, minlength=nb)
    sd2 = np.bincount(key, weights=d * d, minlength=nb)
    sd  = np.bincount(key, weights=d, minlength=nb)
    ok = cnt >= 200
    mb, vb = sm[ok] / cnt[ok], sv[ok] / cnt[ok]
    yb = sd2[ok] / cnt[ok] - (sd[ok] / cnt[ok]) ** 2      # variance per bin
    w = cnt[ok]
    n = len(yb)
    print(f"{n} usable bins (>=200 px each), {int(w.sum()):,} pixels used\n")

    one = np.ones(n)
    models = {
        "M1  a + b*m^2 + c*v          (shipped: speckle)": np.c_[one, mb ** 2, vb],
        "M2  a + p*m   + c*v          (Poisson / shot)  ": np.c_[one, mb, vb],
        "M3  a + p*m + b*m^2 + c*v    (both)            ": np.c_[one, mb, mb ** 2, vb],
        "M4  a + c*v                  (null: no signal) ": np.c_[one, vb],
    }
    tot = float((w * (yb - np.average(yb, weights=w)) ** 2).sum())
    print(f"{'model':<50}{'R2':>9}{'adj R2':>9}{'BIC':>12}")
    print("-" * 80)
    out = {}
    for name, X in models.items():
        c, sse = fit(X, yb, w)
        k = X.shape[1]
        r2 = 1 - sse / tot
        adj = 1 - (1 - r2) * (n - 1) / (n - k)
        bic = n * np.log(max(sse / n, 1e-300)) + k * np.log(n)
        out[name] = (c, r2, adj, bic)
        print(f"{name:<50}{r2:>9.4f}{adj:>9.4f}{bic:>12.1f}")

    best = min(out, key=lambda k_: out[k_][3])
    print(f"\nlowest BIC (best model): {best.strip()}")
    print("\ncoefficients:")
    for name, (c, r2, adj, bic) in out.items():
        print(f"  {name.strip():<46} {np.array2string(c, precision=5, suppress_small=True)}")
    c1, c2 = out[list(models)[0]], out[list(models)[1]]
    print(f"\nspeckle (m^2) vs Poisson (m):  adj R2 {c1[2]:.4f} vs {c2[2]:.4f}   "
          f"BIC {c1[3]:.1f} vs {c2[3]:.1f}")
    print("A BIC difference above 10 is decisive; below 2 is 'no preference'.")


if __name__ == "__main__":
    main()
