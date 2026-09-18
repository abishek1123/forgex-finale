"""Metrics. Owner: Person C (Loss & evaluation).

PSNR and SSIM are dependency-free. LPIPS is optional: if the `lpips` package is
not installed it returns nan rather than crashing, so training never depends on
it. inference.py imports NOTHING from this file.
"""
import math
import torch
import torch.nn.functional as F

from losses import ssim as _ssim

_LPIPS = None


def psnr(pred, gt, data_range=1.0):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return 10.0 * math.log10((data_range ** 2) / max(mse, 1e-12))


def ssim(pred, gt):
    return float(_ssim(pred.float().clamp(0, 1), gt.float()))


def lpips(pred, gt, net="alex", device=None):
    """Lower is better. Returns nan if the optional package is missing."""
    global _LPIPS
    try:
        import lpips as _lp
    except Exception:
        return float("nan")
    if _LPIPS is None:
        _LPIPS = _lp.LPIPS(net=net, verbose=False)
        if device is not None:
            _LPIPS = _LPIPS.to(device)
    p = pred.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    g = gt.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        return float(_LPIPS(p, g).mean())
