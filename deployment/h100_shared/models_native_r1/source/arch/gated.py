"""ForgeX with NAFNet's nonlinearity. Two extra rows for the comparison table.

The eight-architecture study said NAFNet beats ForgeX on both metrics at 9% LOWER
cost, while SwinIR and Restormer win only by paying 2-4x. NAFNet is therefore the
one worth learning from, and the question is WHICH part of it does the work.

The hypothesis these two rows test:

    Our noise is MULTIPLICATIVE. We established that on 400 images and 6.5M
    pixels: var = sigma_add^2 + sigma_mul^2 * m^2 + c*v, with speckle beating
    Poisson by dBIC 1118.3. EDSR's ReLU is a piecewise-LINEAR nonlinearity.
    NAFNet's SimpleGate MULTIPLIES two halves of the feature map together. A
    multiplicative nonlinearity is a structurally better match to multiplicative
    noise than an additive one.

Everything else is ForgeX, unchanged: the variance-stabilising stem, the global
bicubic skip, the zero-init PixelShuffle tail, residual scaling. Only the block's
interior changes, so a win is attributable.

    forgex_gate     SimpleGate instead of ReLU          isolates the gating
    forgex_gated    SimpleGate + LayerNorm per block    both, as NAFNet has them

Run forgex_gated first. If it wins, forgex_gate attributes the gain.
"""
import torch.nn as nn

from .registry import (_Base, VarianceStabilisingStem, PixelShuffleTail,
                       LayerNorm2d, SimpleGate, bicubic_up)


class GatedResBlock(nn.Module):
    """x + res_scale * c2(SimpleGate(c1(x))), optionally LayerNorm'd first.

    c1 widens to 2*ch because SimpleGate halves the channels again. That makes
    the block ~1.5x the parameters of an EDSR block, so match_width solves for a
    narrower ch to stay on the 1,368,705 budget. The comparison stays matched.
    """

    def __init__(self, ch, res_scale=0.1, norm=True):
        super().__init__()
        self.norm = LayerNorm2d(ch) if norm else None
        self.c1 = nn.Conv2d(ch, ch * 2, 3, 1, 1)
        self.sg = SimpleGate()
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        y = self.norm(x) if self.norm is not None else x
        return x + self.res_scale * self.c2(self.sg(self.c1(y)))


class ForgeXGated(_Base):
    """ForgeX skeleton, gated interior. stem and skip are ALWAYS on -- this row
    asks about the block, not about our two contributions."""

    def __init__(self, ch=54, nb=16, scale=2, res_scale=0.1, norm=True):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale, norm=norm)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[GatedResBlock(ch, res_scale, norm) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        return bicubic_up(x, self.scale) + self.out(f).float()


GATED = {
    "forgex_gated": lambda ch, nb: ForgeXGated(ch, nb, norm=True),
    "forgex_gate":  lambda ch, nb: ForgeXGated(ch, nb, norm=False),
}
GATED_NB = {"forgex_gated": 16, "forgex_gate": 16}
