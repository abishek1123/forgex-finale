"""The restoration network. Owner: Person B (Model).

Design notes (these are the slide-4 talking points):

 1. Everything runs at LOW resolution; the only upsample is a PixelShuffle in
    the last block. A network that upsamples first does 4x the work for no
    accuracy gain, and KLA benchmarks inference time.

 2. Global residual: we add a bicubic upsample of the input, so the network only
    has to predict the CORRECTION to a cheap baseline. It starts at ~23.4 dB
    instead of at zero and converges far faster.

 3. Variance-stabilising stem: speckle noise has variance proportional to I^2,
    so it is loud in bright regions and quiet in dark ones. Feeding raw / sqrt /
    log views lets the first conv choose the representation in which the noise
    is closest to uniform. Costs ~0 parameters.

 4. No BatchNorm -- it is well established to hurt super-resolution.

The forward pass does NOT clamp to [0,1]: clamping during training zeroes the
gradient for out-of-range predictions. inference.py clamps at the very end.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceStabilisingStem(nn.Module):
    def forward(self, x):
        p = x.clamp_min(0.0)
        return torch.cat([x, torch.sqrt(p + 1e-6), torch.log1p(p)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.c2(F.relu(self.c1(x)))


class Restorer(nn.Module):
    def __init__(self, ch=64, nb=16, scale=2, res_scale=0.1):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch, res_scale) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1), nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        residual = self.tail(self.up(f))
        base = F.interpolate(x.float(), scale_factor=self.scale, mode="bicubic", align_corners=False)
        # fp32 on purpose: in fp16 the spacing near 0.5 is ~5e-4, so an early
        # (small) residual would be rounded straight out of the sum.
        return base + residual.float()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(cfg=None):
    cfg = cfg or {}
    return Restorer(**{k: cfg[k] for k in ("ch", "nb", "scale", "res_scale") if k in cfg})
