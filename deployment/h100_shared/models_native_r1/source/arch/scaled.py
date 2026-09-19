"""Architectures that only become themselves at capacity, plus RCAN.

WHY THIS FILE EXISTS

The eight-architecture study was run at ONE budget: 1,368,705 parameters, the size
of the shipped model. The whole field landed inside 0.12 dB. That result is often
read as "architecture does not matter". It does not say that. It says architecture
does not matter AT 1.37 M, which is a different claim, and a much weaker one.

At 1.37 M every architecture in that table was matched by shrinking its WIDTH while
its DEPTH stayed at whatever the study fixed. For our own residual net that is fine
-- EDSR-family nets are width-scaled by design. For the others it is not:

  NAFNet   was built with ENC/MID/DEC = [1,1,1] / 2 / [1,1,1]  -- five blocks.
           The paper's SIDD NAFNet is [2,2,4,8] / 12 / [2,2,2,2] -- thirty-two.
           We benchmarked a shallow U-Net and called it NAFNet.
  SwinIR   was built with 4 RSTBs. SwinIR-classical is 6 RSTBs at embed 180.
  Restormer was searched over block layouts, so it alone was scaled honestly.

So the earlier table is a fair comparison of these architectures' BLOCKS and an
unfair comparison of their SHAPES. This file restores the shapes, so that the
question the H100 lets us ask -- "does the ranking change with capacity?" -- is
asked of the real architectures rather than of miniatures.

WHAT IS ADDED

  rcan / rcan_ours       RCAB groups with channel attention. Absent from the first
                         study and it should not have been: it is the strongest
                         pure-convolution SR backbone of the pre-transformer era,
                         it scales by DEPTH rather than width, and being pure conv
                         it is the one large model that stays fast on an H100.
                         If a 15 M model is going to win on both quality and
                         throughput, this is the most likely one.
  nafnet_deep            NAFNet at [2,2,4] / 8 / [2,2,2].
  nafnet_xdeep           NAFNet at [2,2,4] / 12 / [2,2,2].
  swinir_deep            SwinIR at 6 RSTBs x depth 6 -- the classical-SR shape.
  restormer_deep         Restormer at the paper layout [4,6,6,8], 4 refinement.

Every one keeps the study contract: (ch, nb) constructor, .config, .n_params(),
B,1,H,W -> B,1,2H,2W float32, and the global bicubic skip so that training starts
at bicubic for all of them and the table measures architecture, not initialisation.
"""
import math

import torch
import torch.nn as nn

from .registry import (_Base, VarianceStabilisingStem, PixelShuffleTail,
                       bicubic_up, TARGET_PARAMS)


# ============================================================== RCAN
class CALayer(nn.Module):
    """Squeeze-and-excite over channels. One scalar per feature map, from the
    global average. Costs 2/r of a 1x1 conv and is where RCAN's gain lives."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        r = max(4, ch // reduction)
        self.f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class RCAB(nn.Module):
    def __init__(self, ch, res_scale=1.0, reduction=16):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1), CALayer(ch, reduction))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.b(x)


class ResidualGroup(nn.Module):
    """RCAB x n, then a conv, then a group-level skip. The 'residual in residual'
    that lets RCAN go deep without the signal dying on the way through."""

    def __init__(self, ch, n, res_scale=1.0, reduction=16):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(ch, res_scale, reduction) for _ in range(n)],
                                  nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class RCAN(_Base):
    """nb is the TOTAL number of RCABs; they are split into groups of `per_group`.

    nb=32 -> 4 groups of 8.  nb=64 -> 8 groups of 8.  Depth is the scaling axis,
    which is the point: our residual net scales by width and saturates, RCAN
    scales by depth and the question is whether that saturates at the same place.
    """

    def __init__(self, ch=64, nb=32, scale=2, stem=True, skip=True,
                 per_group=8, res_scale=1.0, reduction=16):
        super().__init__()
        nb = int(nb)
        ng = max(1, int(round(nb / float(per_group))))
        per = max(1, int(round(nb / float(ng))))
        self.config = dict(ch=ch, nb=nb, scale=scale, stem=stem, skip=skip,
                           per_group=per, groups=ng, kind="rcan")
        self.scale, self.use_stem, self.use_skip = scale, stem, skip
        self.stem = VarianceStabilisingStem() if stem else None
        self.head = nn.Conv2d(3 if stem else 1, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualGroup(ch, per, res_scale, reduction)
                                    for _ in range(ng)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)
        if not skip:
            nn.init.kaiming_normal_(self.out.tail.weight, nonlinearity="linear")
            nn.init.zeros_(self.out.tail.bias)

    def forward(self, x):
        f = self.head(self.stem(x) if self.use_stem else x)
        f = f + self.body_tail(self.body(f))
        y = self.out(f).float()
        return bicubic_up(x, self.scale) + y if self.use_skip else y


# ================================================ deeper external shapes
_EXT_OK = True
try:
    from .external import NAFNetSR, SwinIRSR, RestormerSR
except Exception:                                    # sources not downloaded
    _EXT_OK = False


SCALED = {
    "rcan":      lambda ch, nb: RCAN(ch, nb, stem=False, skip=True),
    "rcan_ours": lambda ch, nb: RCAN(ch, nb, stem=True,  skip=True),
}
SCALED_NB = {"rcan": 32, "rcan_ours": 32}


if _EXT_OK:
    def _naf(enc, mid, dec, stem):
        class _N(NAFNetSR):
            ENC, MID, DEC = list(enc), mid, list(dec)
        return lambda ch, nb=None: _N(ch, nb, stem=stem)

    SCALED.update({
        "nafnet_deep":     _naf([2, 2, 4], 8, [2, 2, 2], False),
        "nafnet_deep_ours": _naf([2, 2, 4], 8, [2, 2, 2], True),
        "nafnet_xdeep":    _naf([2, 2, 4], 12, [2, 2, 2], False),
        "swinir_deep":     lambda ch, nb: SwinIRSR(ch, 6, stem=False, depth=6),
        "restormer_deep":  lambda ch, nb: RestormerSR(ch, None, stem=False,
                                                      blocks=[4, 6, 6, 8], refine=4),
    })
    SCALED_NB.update({"nafnet_deep": None, "nafnet_deep_ours": None,
                      "nafnet_xdeep": None, "swinir_deep": 6,
                      "restormer_deep": None})


# ================================================ the width solver
# Parameter count is very close to quadratic in width for every architecture
# here, so we probe once, jump straight to sqrt(target/n0)*probe, and refine on
# a small bracket. A linear scan would build a hundred 30 M models to find one.
STEP = {"restormer_deep": 8, "swinir_deep": 6}


def _n(make, ch):
    try:
        return make(ch).n_params()
    except Exception:
        return None


def _solve(make, target, lo=8, hi=512, step=1, probe=32):
    n0 = None
    for c in range(probe, hi + 1, max(step, 1)):
        n0 = _n(make, c)
        if n0:
            probe = c
            break
    if not n0:
        return None
    ch0 = int(round(probe * math.sqrt(float(target) / n0)))
    a = max(lo, int(ch0 * 0.55))
    b = min(hi, int(ch0 * 1.60) + step)
    a = a + (-a) % step if step > 1 else a
    best = None
    for ch in range(max(a, step), b + 1, step):
        n = _n(make, ch)
        if n is None:
            continue
        e = abs(n - target) / float(target)
        if best is None or e < best[2]:
            best = (ch, n, e)
    return best


def match_width(name, nb=None, target=TARGET_PARAMS, lo=8, hi=512):
    """(ch, n_params, relative_error) for a scaled architecture at `target`."""
    if nb is None:
        nb = SCALED_NB[name]
    step = STEP.get(name, 1)
    return _solve(lambda ch: SCALED[name](ch, nb), target, lo=lo, hi=hi, step=step,
                  probe=max(step, 24))


if __name__ == "__main__":
    import sys
    tgt = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else TARGET_PARAMS
    x = torch.randn(2, 1, 128, 128)
    print(f"target {tgt:,.0f} parameters\n")
    print(f"{'arch':<20}{'ch':>5}{'nb':>6}{'params':>13}{'err':>8}   forward")
    print("-" * 74)
    for name in SCALED:
        try:
            ch, n, e = match_width(name, target=tgt)[:3]
            m = SCALED[name](ch, SCALED_NB[name]).eval()
            with torch.no_grad():
                y = m(x)
            ok = tuple(y.shape) == (2, 1, 256, 256) and y.dtype == torch.float32
            print(f"{name:<20}{ch:>5}{str(SCALED_NB[name]):>6}{n:>13,}{e:>7.1%}   "
                  f"{tuple(y.shape)} {'OK' if ok else 'BAD'}")
        except Exception as ex:
            print(f"{name:<20}    -     -            -       -   FAILED: "
                  f"{type(ex).__name__}: {ex}")
