"""Loss functions. Owner: Person C (Loss).

Default = Charbonnier + SSIM + gradient. Rationale:
  * Charbonnier (smooth L1) beats MSE on PSNR in practice.
  * SSIM is one of the three metrics KLA scores, so we optimise it directly.
  * The gradient term sharpens edges WHERE THE GROUND TRUTH HAS EDGES. Unlike a
    perceptual/GAN loss it has no mechanism to invent texture -- which matters,
    because a hallucinated texture on an inspection image is a fabricated defect.

Reference: Zhao et al., "Loss Functions for Image Restoration with Neural
Networks", IEEE Trans. Computational Imaging, 2017.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x.float() - y.float()) ** 2 + eps ** 2).mean()


def _gauss_window(ws, sigma, device, dtype):
    g = torch.arange(ws, dtype=torch.float32, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].to(dtype)


def ssim(x, y, ws=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """Mean SSIM over the batch. Inputs (B,1,H,W), data range 1.0.

    ALWAYS computed in float32. In fp16 the local variances go slightly
    negative and the denominator product underflows below 6e-5 (fp16's
    smallest normal), which yields Inf and then NaN gradients -- and NaN is
    scale-invariant, so GradScaler skips every step forever and the model
    never trains. Do not remove the .float() calls.
    """
    # autocast MUST be disabled explicitly: F.conv2d inside an autocast region
    # re-casts its inputs to fp16 no matter what .float() we call beforehand.
    # cuDNN TF32 must ALSO be disabled: on Ampere/Ada GPUs "fp32" convolution
    # runs in TF32 (10-bit mantissa) by default, and the variance term
    # E[x^2]-E[x]^2 cancels below TF32 precision in flat regions -- measured
    # to underestimate SSIM by ~0.017 vs CPU on this dataset.
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        return _ssim_fp32(x, y, ws, sigma, C1, C2)
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32


def _ssim_fp32(x, y, ws, sigma, C1, C2):
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        w = _gauss_window(ws, sigma, x.device, x.dtype)
        pad = ws // 2
        mx = F.conv2d(x, w, padding=pad)
        my = F.conv2d(y, w, padding=pad)
        vx = (F.conv2d(x * x, w, padding=pad) - mx * mx).clamp_min(0)
        vy = (F.conv2d(y * y, w, padding=pad) - my * my).clamp_min(0)
        vxy = F.conv2d(x * y, w, padding=pad) - mx * my
        s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
        return s.mean()


def gradient_loss(x, y, eps=1e-3):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return charbonnier(dx(x), dx(y), eps) + charbonnier(dy(x), dy(y), eps)


# ---------------------------------------------------------------------------
# Multi-scale Laplacian-pyramid Charbonnier.
#
# WHY THIS AND NOT A BIGGER GRADIENT WEIGHT. The measured failure is
# high-frequency UNDER-reconstruction: on 150 held-out images the shipped family
# recovers only ~25% of the ground truth's HF energy, and 0 images exceed it.
# Raising --w-grad 0.05 -> 0.20 (run e1g_grad20) made HF energy WORSE
# (0.2490 -> 0.2411): a single first-difference is one broadband term and cannot
# target a band. A Laplacian pyramid decomposes the error per octave, so the
# deficient bands get their own weight.
#
# Burt-Adelson construction: a fixed 5x5 binomial kernel, decimate by 2, expand
# by zero-insert + the same kernel x4, band = current - expanded.
# ---------------------------------------------------------------------------
_GAUSS5 = {}


def _gauss5(device, dtype):
    key = (device, dtype)
    if key not in _GAUSS5:
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
        k = k[:, None] * k[None, :]
        _GAUSS5[key] = (k / k.sum())[None, None]
    return _GAUSS5[key]


def _lap_down(x, w):
    return F.conv2d(x, w, padding=2)[:, :, ::2, ::2]


def _lap_up(x, w, size):
    up = torch.zeros(x.shape[0], x.shape[1], size[0], size[1],
                     device=x.device, dtype=x.dtype)
    up[:, :, ::2, ::2] = x
    return F.conv2d(up * 4.0, w, padding=2)


def laplacian_pyramid(x, levels, w):
    """-> list of `levels` band-pass images, finest first."""
    bands, cur = [], x
    for _ in range(levels):
        d = _lap_down(cur, w)
        bands.append(cur - _lap_up(d, w, cur.shape[-2:]))
        cur = d
    return bands


LAP_WEIGHTS = (1.0, 0.5, 0.25)          # finest -> coarsest


def laplacian_loss(pred, gt, levels=3, eps=1e-3):
    """Charbonnier per Laplacian band, weighted finest-first.

    FP32 with autocast explicitly disabled, and cuDNN TF32 disabled, for the same
    reason as ssim(): a band is a DIFFERENCE of two nearly equal blurred images,
    so it cancels catastrophically at reduced mantissa. F.conv2d inside an
    autocast region re-casts its inputs regardless of any .float() beforehand.
    """
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            p, g = pred.float(), gt.float()
            w = _gauss5(p.device, p.dtype)
            bp = laplacian_pyramid(p, levels, w)
            bg = laplacian_pyramid(g, levels, w)
            total = p.new_zeros(())
            for k in range(levels):
                wk = LAP_WEIGHTS[k] if k < len(LAP_WEIGHTS) else LAP_WEIGHTS[-1] * 0.5 ** (
                    k - len(LAP_WEIGHTS) + 1)
                total = total + wk * charbonnier(bp[k], bg[k], eps)
            return total
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32


_LPIPS_NET = None


def lpips_loss(pred, gt, net="alex"):
    """Differentiable perceptual distance. Lower is better.

    Matches metrics.lpips EXACTLY -- same network, same 1->3 channel
    replication, same [0,1] -> [-1,1] mapping -- so we optimise the quantity we
    are actually scored on rather than a lookalike.

    Computed in fp32 with autocast explicitly disabled, for the same reason as
    ssim(): F.conv2d inside an autocast region re-casts its inputs no matter
    what .float() was called beforehand, and one NaN here makes GradScaler skip
    every step forever with no error raised. That cost us a night in round 1.
    """
    global _LPIPS_NET
    import lpips as _lp
    if _LPIPS_NET is None:
        _LPIPS_NET = _lp.LPIPS(net=net, verbose=False).to(pred.device).eval()
        for q in _LPIPS_NET.parameters():
            q.requires_grad_(False)
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        p = pred.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        g = gt.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        return _LPIPS_NET(p, g).mean()


class RestorationLoss(nn.Module):
    """mode='charbonnier' is the control run; mode='combo' is the default."""

    def __init__(self, mode="combo", w_ssim=0.15, w_grad=0.05, w_lpips=0.0,
                 w_laplacian=0.0, laplacian_levels=3):
        super().__init__()
        self.mode, self.w_ssim, self.w_grad = mode, w_ssim, w_grad
        self.w_lpips = w_lpips
        self.w_laplacian, self.laplacian_levels = w_laplacian, laplacian_levels

    def forward(self, pred, gt):
        parts = {"charb": charbonnier(pred, gt)}
        total = parts["charb"]
        if self.mode == "combo":
            parts["ssim"] = 1.0 - ssim(pred.clamp(0, 1), gt)
            parts["grad"] = gradient_loss(pred, gt)
            total = total + self.w_ssim * parts["ssim"] + self.w_grad * parts["grad"]
        if self.w_lpips > 0:
            parts["lpips"] = lpips_loss(pred, gt)
            total = total + self.w_lpips * parts["lpips"]
        # w_laplacian == 0.0 -> this branch never runs and behaviour is unchanged.
        if self.w_laplacian > 0:
            parts["lap"] = laplacian_loss(pred, gt, self.laplacian_levels)
            total = total + self.w_laplacian * parts["lap"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
