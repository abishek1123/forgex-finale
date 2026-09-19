"""
Architecture registry for the ForgeX comparison study.

Every architecture here obeys the SAME contract as src/model.py's Restorer:

    Model(ch=<width>, nb=<depth>)        constructor signature
    .config                              dict, so run.py can rebuild it
    .n_params()                          int
    forward(x: B,1,H,W) -> B,1,2H,2W     float32, NOT clamped

That contract is what lets tools/train_arch.py swap the architecture without
touching a single line of src/train.py -- so data, degradation engine, split,
loss, optimiser, schedule, seed and epoch count are byte-identical across every
row of the comparison table. The architecture is the only variable.

Same-resolution backbones (UNet, NAFNet, Restormer) get a PixelShuffle x2 tail
rather than being fed a bicubic-upsampled input. That keeps compute comparable
to ours and isolates the backbone; feeding them 256x256 would make them 4x more
expensive and the table would measure that instead of the architecture.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- shared
class VarianceStabilisingStem(nn.Module):
    """x, sqrt(x), log1p(x) side by side. Parameter-free."""
    def forward(self, x):
        p = x.clamp_min(0.0)
        return torch.cat([x, torch.sqrt(p + 1e-6), torch.log1p(p)], dim=1)


class PixelShuffleTail(nn.Module):
    """ch features at LR -> 1 channel at 2x. Zero-init so training starts at the skip."""
    def __init__(self, ch, scale=2):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1),
                                nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, f):
        return self.tail(self.up(f))


def bicubic_up(x, scale=2):
    # fp32 on purpose: in fp16 the spacing near 0.5 is ~5e-4, so a small early
    # residual would be rounded straight out of the sum.
    return F.interpolate(x.float(), scale_factor=scale, mode="bicubic", align_corners=False)


class _Base(nn.Module):
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------- 1. ForgeX (and ablations)
class ResBlock(nn.Module):
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.c2(F.relu(self.c1(x)))


class ForgeX(_Base):
    """Our architecture, with the two contributions individually switchable.

    stem=False, skip=False  ->  a plain EDSR baseline. That row is the ablation
    that says what the variance-stabilising stem and the global bicubic skip
    actually bought.
    """
    def __init__(self, ch=64, nb=16, scale=2, res_scale=0.1, stem=True, skip=True):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale,
                           stem=stem, skip=skip)
        self.scale, self.use_stem, self.use_skip = scale, stem, skip
        self.stem = VarianceStabilisingStem() if stem else None
        self.head = nn.Conv2d(3 if stem else 1, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch, res_scale) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)
        if not skip:
            # with no bicubic base to correct, a zero-init tail cannot learn
            nn.init.kaiming_normal_(self.out.tail.weight, nonlinearity="linear")
            nn.init.zeros_(self.out.tail.bias)

    def forward(self, x):
        f = self.head(self.stem(x) if self.use_stem else x)
        f = f + self.body_tail(self.body(f))
        y = self.out(f).float()
        return bicubic_up(x, self.scale) + y if self.use_skip else y


# --------------------------------------------------------------- 2. U-Net
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class UNet(_Base):
    """Plain encoder-decoder with skips. The sanity floor: if this ties the
    others, the table is telling you architecture is not the lever."""
    def __init__(self, ch=32, nb=3, scale=2):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale)
        self.scale, self.depth = scale, nb
        self.stem = VarianceStabilisingStem()
        chs = [ch * (2 ** i) for i in range(nb + 1)]
        self.enc = nn.ModuleList()
        cin = 3
        for c in chs[:-1]:
            self.enc.append(ConvBlock(cin, c)); cin = c
        self.mid = ConvBlock(cin, chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(nb - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2))
            self.dec.append(ConvBlock(chs[i] * 2, chs[i]))
        self.out = PixelShuffleTail(chs[0], scale)

    def forward(self, x):
        f = self.stem(x)
        skips = []
        for e in self.enc:
            f = e(f); skips.append(f); f = F.max_pool2d(f, 2)
        f = self.mid(f)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            f = u(f)
            if f.shape[-2:] != s.shape[-2:]:
                f = F.interpolate(f, size=s.shape[-2:], mode="nearest")
            f = d(torch.cat([f, s], dim=1))
        return bicubic_up(x, self.scale) + self.out(f).float()


# -------------------------------------------------------------- 3. NAFNet
class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.g + self.b


class NAFBlock(nn.Module):
    """Chen et al., 'Simple Baselines for Image Restoration', ECCV 2022.
    Spot-check against the official repo before quoting numbers."""
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        d = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        f = c * ffn_expand
        self.conv4 = nn.Conv2d(c, f, 1)
        self.conv5 = nn.Conv2d(f // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.conv2(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(_Base):
    """NAFNet backbone at LR resolution + PixelShuffle x2 tail.
    Already validated on SEM by Park, Oh & Jang, Sci. Rep. 2025."""
    def __init__(self, ch=32, nb=16, scale=2):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(ch) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        return bicubic_up(x, self.scale) + self.out(f).float()


# --------------------------------------------------------------- registry
ARCHS = {
    "forgex":       lambda ch, nb: ForgeX(ch, nb, stem=True,  skip=True),
    "edsr_base":    lambda ch, nb: ForgeX(ch, nb, stem=False, skip=False),
    "abl_nostem":   lambda ch, nb: ForgeX(ch, nb, stem=False, skip=True),
    "abl_noskip":   lambda ch, nb: ForgeX(ch, nb, stem=True,  skip=False),
    "unet":         lambda ch, nb: UNet(ch, nb),
    "nafnet":       lambda ch, nb: NAFNet(ch, nb),
}

# depth is fixed per architecture; the width is what the matcher solves for
DEFAULT_NB = {"forgex": 16, "edsr_base": 16, "abl_nostem": 16, "abl_noskip": 16,
              "unet": 3, "nafnet": 16}

TARGET_PARAMS = 1_368_705          # the shipped ForgeX model


def build(name, ch=None, nb=None):
    nb = DEFAULT_NB[name] if nb is None else nb
    if ch is None:
        ch = match_width(name, nb)[0]
    return ARCHS[name](ch, nb)


def match_width(name, nb=None, target=TARGET_PARAMS, lo=4, hi=512):
    """Smallest width whose parameter count is closest to `target`.

    Returns (ch, n_params, relative_error). Widths are searched on a coarse
    grid then refined, because parameter count is monotone in width but not
    smooth (grouped convs, channel halving in SimpleGate)."""
    nb = DEFAULT_NB[name] if nb is None else nb
    best = None
    for ch in range(lo, hi + 1):
        try:
            n = ARCHS[name](ch, nb).n_params()
        except Exception:
            continue
        err = abs(n - target) / target
        if best is None or err < best[2]:
            best = (ch, n, err)
        if n > target * 1.6:            # monotone enough to stop early
            break
    return best


# ---- fold in the official SwinIR / NAFNet / Restormer when their source
# ---- files have been downloaded into this folder. Absent -> silently skipped,
# ---- so the six in-repo architectures always work on their own.
_EXT = {}
try:
    from .external import EXTERNAL as _EXT, EXTERNAL_NB as _EXT_NB, match_width as _ext_match
    ARCHS.update(_EXT)
    DEFAULT_NB.update(_EXT_NB)
except Exception:                                    # not downloaded yet
    pass

_own_match = match_width


def match_width(name, nb=None, target=TARGET_PARAMS, lo=4, hi=512):        # noqa: F811
    """Transformers blow up long before ch=512, so the external architectures
    use their own bounded search."""
    if name in _EXT:
        return _ext_match(name, target=target)
    return _own_match(name, nb=nb, target=target, lo=lo, hi=hi)


if __name__ == "__main__":
    print(f"target {TARGET_PARAMS:,} parameters\n")
    print(f"{'arch':<14}{'ch':>5}{'nb':>4}{'params':>12}{'err':>8}   forward")
    print("-" * 64)
    x = torch.randn(2, 1, 128, 128)
    for name in ARCHS:
        ch, n, err = match_width(name)
        m = ARCHS[name](ch, DEFAULT_NB[name]).eval()
        with torch.no_grad():
            y = m(x)
        ok = (y.shape == (2, 1, 256, 256) and y.dtype == torch.float32)
        print(f"{name:<14}{ch:>5}{DEFAULT_NB[name]:>4}{n:>12,}{err:>7.1%}   "
              f"{tuple(y.shape)} {'OK' if ok else 'BAD'}")


# ---- ForgeX capacity study: gated + scaled
# Appended by tools/register_scaled.py. Each import is guarded on its own, so a
# missing file removes its rows and breaks nothing else.
try:
    from .attn import E1 as _E1, E1_NB as _E1_NB, match_width as _e1_match
    ARCHS.update(_E1)
    DEFAULT_NB.update(_E1_NB)
except Exception:
    _E1 = {}

try:
    from .gated import GATED as _GT, GATED_NB as _GT_NB
    ARCHS.update(_GT)
    DEFAULT_NB.update(_GT_NB)
except Exception:
    _GT = {}

try:
    from .scaled import (SCALED as _SC, SCALED_NB as _SC_NB,
                         match_width as _sc_match)
    ARCHS.update(_SC)
    DEFAULT_NB.update(_SC_NB)
    _chain_match = match_width

    def match_width(name, nb=None, target=TARGET_PARAMS, lo=4, hi=512):   # noqa: F811
        """Scaled architectures solve their own width; everything else falls
        through to the chain that was already here."""
        if name in _E1:
            return _e1_match(name, nb=nb, target=target)
        if name in _SC:
            return _sc_match(name, nb=nb, target=target)
        return _chain_match(name, nb=nb, target=target, lo=lo, hi=hi)
except Exception:
    _SC = {}
