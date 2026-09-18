#!/usr/bin/env python3
"""The output contract, enforced identically for every optimisation config.

Seven requirements, from the optimisation brief:
  1 compare outputs to reference   2 shape   3 float32   4 finite
  5 range [0,1]                    6 PSNR/SSIM/LPIPS      7 stress suite (separate)
"""
import numpy as np

# The SHIPPED model, e1f_gate-120 (models/model.pt sha1 f95377a21e86...), on the
# organisers' 297-image test set. Per-image means -- see kla2/results/final.csv.
REF = dict(psnr=23.8296, ssim=0.62104, lpips=0.18126)
# Round 2, superseded: dict(psnr=23.632, ssim=0.60792, lpips=0.19287)


def contract(out, lr_shape_list=None):
    """-> (ok: bool, list[str] failures). `out` is (N,2H,2W) float32."""
    f = []
    if out.dtype != np.float32:
        f.append(f"dtype {out.dtype} != float32")
    if out.ndim != 3:
        f.append(f"ndim {out.ndim} != 3")
    if not np.isfinite(out).all():
        n = int((~np.isfinite(out)).sum()); f.append(f"{n} non-finite values")
    lo, hi = float(out.min()), float(out.max())
    if lo < 0.0 or hi > 1.0:
        f.append(f"range [{lo:.6f},{hi:.6f}] outside [0,1]")
    if lr_shape_list is not None:
        for i, (h, w) in enumerate(lr_shape_list):
            if out[i].shape != (2 * h, 2 * w):
                f.append(f"img {i}: {out[i].shape} != {(2*h, 2*w)}"); break
    return (len(f) == 0), f


def vs_reference(out, ref):
    if ref is None:
        return None, None
    d = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    mse = float((d ** 2).mean())
    return float(d.max()), (float("inf") if mse <= 0 else 10 * np.log10(1.0 / mse))


def report(tag, out, ref=None, quality=None):
    ok, fails = contract(out)
    md, ps = vs_reference(out, ref)
    line = f"  [{'PASS' if ok else 'FAIL'}] {tag:<26}"
    if md is not None:
        line += f" max|diff| {md:.3e}  PSNR-vs-ref {ps:8.2f} dB"
    if quality:
        P, S, L = quality
        line += (f"  |  {P:.4f}/{S:.5f}/{L:.5f}"
                 f"  dPSNR {P-REF['psnr']:+.4f} dSSIM {S-REF['ssim']:+.5f} "
                 f"dLPIPS {L-REF['lpips']:+.5f}")
    print(line)
    for x in fails:
        print(f"         !! {x}")
    return ok, md
