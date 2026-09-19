"""E1 -- WHICH KIND of attention, if any, buys back the SSIM our trunk loses.

WHY THESE SIX ROWS

The matched-parameter study is usually quoted as "the field spans 0.12 dB", which
is a PSNR statement and it hides the result that matters. The SSIM column:

    edsr_base   0.52616 |  nafnet     0.54204
    abl_nostem  0.52700 |  swinir     0.55323
    forgex      0.52764 |  restormer  0.55388
    abl_noskip  0.52860 |
    unet        0.53084 |

The four EDSR-family rows sit inside 0.0024 SSIM of each other. NAFNet is +0.015
above that cluster and the two attention models are +0.026 -- six and eleven times
the within-family spread, at matched parameters, same data, same schedule. LPIPS
orders identically and independently. That is not noise.

And it lands on our wound. The failure measured on the new dataset is an SSIM
failure: gain over bicubic correlates -0.51 with fine-texture content, and on the
three most textured images our SSIM falls BELOW bicubic. An architecture worth
+0.026 SSIM is the same size as the entire textured-half gain (+0.030).

So attention is worth testing. But "add a Restormer block" cannot be the
experiment, because of what MDTA actually is. Measured at our trunk width
(ch 64, 128x128):

    block                          params    GMAC   vs ResBlock
    ResBlock (ours)                73,856   1.208        1.00x
    SE / channel attention         74,436   1.208        1.00x   <- FREE
    SimpleGate block              110,912   1.812        1.50x
    MDTA only                      18,116   0.330        0.27x
    TransformerBlock (MDTA+GDFN)   54,072   0.915        0.76x

and the attention matmul is 33.6 of MDTA's 330 MMAC -- TEN PERCENT. The other
90% is a 1x1 QKV projection and a 3x3 depthwise. MDTA's attention map is C x C,
over CHANNELS, not space: at ch 64 it is a 64x64 matrix that re-weights channels
using statistics pooled over the whole image. It does not connect distant pixels.

So without a channel-attention control, a win for MDTA would be reported as
"global attention helps" when the true finding may be "any channel re-weighting
helps" -- and channel attention is free while the Transformer block costs ~4x in
wall clock (memory-bound softmax/transpose/LayerNorm, not FLOPs). E1b is that
control and it is the most important row in the table.

WHAT IS HELD FIXED

Everything except the contents of the body. Stem, head conv, long skip,
body_tail, PixelShuffle tail, zero-init, bicubic skip in fp32 -- all byte-identical
to src/model.py's Restorer. Width is solved per row to 1,368,705 parameters. If a
row wins, the win is attributable to the block, because nothing else moved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import (_Base, VarianceStabilisingStem, PixelShuffleTail,
                       bicubic_up, ResBlock, SimpleGate, LayerNorm2d,
                       TARGET_PARAMS)


def _heads(ch, want=4):
    """Largest head count <= `want` that divides ch. The width solver walks ch
    one at a time, so a fixed head count would explode on most candidates."""
    for h in (want, 3, 2, 1):
        if ch % h == 0:
            return h
    return 1


# ----------------------------------------------------- E1b: channel attention
class RCAB(nn.Module):
    """Our ResBlock with squeeze-and-excite on the residual. One scalar per
    channel from the global average. +~ch^2/8 parameters, identical FLOPs."""

    def __init__(self, ch, res_scale=0.1, reduction=16):
        super().__init__()
        r = max(4, ch // reduction)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(ch, r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(r, ch, 1), nn.Sigmoid())
        self.res_scale = res_scale

    def forward(self, x):
        y = self.c2(F.relu(self.c1(x)))
        return x + self.res_scale * (y * self.se(y))


# ----------------------------------------------------------- E1f: SimpleGate
class GateBlock(nn.Module):
    """NAFNet's nonlinearity inside our block. SimpleGate MULTIPLIES two halves
    of the feature map; ReLU is piecewise linear. Our noise is multiplicative
    (speckle over Poisson by dBIC 1118), so a multiplicative nonlinearity is a
    structurally better match -- that is the hypothesis this row tests."""

    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.c1 = nn.Conv2d(ch, ch * 2, 3, 1, 1)
        self.sg = SimpleGate()
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.c2(self.sg(self.c1(self.norm(x))))


# ------------------------------------------------- E1c/E1d: Restormer's MDTA
def restormer_block(ch):
    """The official TransformerBlock (MDTA + GDFN), loaded from the vendored
    source so this row is the real thing and not a re-implementation."""
    from .external import _load
    TB = _load("restormer_official.py", "TransformerBlock")
    return TB(dim=ch, num_heads=_heads(ch, 4), ffn_expansion_factor=2.66,
              bias=False, LayerNorm_type="WithBias")


# ------------------------------------------- E1e: windowed SPATIAL attention
def _win_split(x, w):
    B, C, H, W = x.shape
    x = x.view(B, C, H // w, w, W // w, w).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(-1, C, w * w)                       # (B*nwin), C, w*w


def _win_merge(t, w, B, C, H, W):
    t = t.view(B, H // w, W // w, C, w, w).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(B, C, H, W)


class WindowAttention(nn.Module):
    """Plain (non-shifted) self-attention inside w x w windows, over SPACE.

    This is the row that separates SPATIAL long range from MDTA's channel
    mixing. SEM images contain repeating structure, so if pixel-to-pixel
    relations matter at all, this is the block that can use them and MDTA is
    not. It is the most expensive row in the table -- attention here is
    O(w^4) per window -- which is exactly why only one block is inserted.
    """

    def __init__(self, ch, heads=4, window=8, mlp=2.0, res_scale=0.1):
        super().__init__()
        self.h, self.w, self.res_scale = _heads(ch, heads), window, res_scale
        self.n1 = LayerNorm2d(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n2 = LayerNorm2d(ch)
        hid = int(ch * mlp)
        self.mlp = nn.Sequential(nn.Conv2d(ch, hid, 1), nn.GELU(), nn.Conv2d(hid, ch, 1))

    def _attend(self, y):
        B, C, H, W = y.shape
        w, h = self.w, self.h
        q, k, v = self.qkv(y).chunk(3, dim=1)
        d = C // h
        q, k, v = (_win_split(t, w).view(-1, h, d, w * w) for t in (q, k, v))
        a = (q.transpose(-2, -1) @ k) * (d ** -0.5)       # N,h,L,L
        o = a.softmax(-1) @ v.transpose(-2, -1)           # N,h,L,d
        o = o.transpose(-2, -1).reshape(-1, C, w * w)
        return self.proj(_win_merge(o, w, B, C, H, W))

    def forward(self, x):
        H, W = x.shape[-2:]
        w = self.w
        ph, pw = (w - H % w) % w, (w - W % w) % w
        y = self.n1(x)
        if ph or pw:
            y = F.pad(y, (0, pw, 0, ph), mode="reflect")
        y = self._attend(y)[..., :H, :W]
        x = x + self.res_scale * y
        return x + self.res_scale * self.mlp(self.n2(x))


# ------------------------------------------------------------------ the net
BODY = {
    "res":  lambda ch, rs: ResBlock(ch, rs),
    "se":   lambda ch, rs: RCAB(ch, rs),
    "gate": lambda ch, rs: GateBlock(ch, rs),
}
ATTN = {"mdta": lambda ch: restormer_block(ch),
        "win":  lambda ch: WindowAttention(ch)}


class E1Net(_Base):
    """src/model.py's Restorer with ONE thing swapped: the body.

    kind    which block fills the non-attention slots
    n_attn  how many slots are replaced by `attn`, spaced evenly through the trunk
    """

    def __init__(self, ch=64, nb=16, scale=2, res_scale=0.1,
                 kind="res", attn="", n_attn=0):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale,
                           kind=kind, attn=attn, n_attn=n_attn)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        at = {int(round((i + 1) * nb / (n_attn + 1))) - 1 for i in range(n_attn)} if n_attn else set()
        self.body = nn.Sequential(*[ATTN[attn](ch) if i in at else BODY[kind](ch, res_scale)
                                    for i in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)            # includes the zero-init tail

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        return bicubic_up(x, self.scale) + self.out(f).float()


E1 = {
    "e1a_base": lambda ch, nb: E1Net(ch, nb, kind="res"),
    "e1b_se":   lambda ch, nb: E1Net(ch, nb, kind="se"),
    "e1c_mdta1": lambda ch, nb: E1Net(ch, nb, kind="res", attn="mdta", n_attn=1),
    "e1d_mdta2": lambda ch, nb: E1Net(ch, nb, kind="res", attn="mdta", n_attn=2),
    "e1e_win1":  lambda ch, nb: E1Net(ch, nb, kind="res", attn="win",  n_attn=1),
    "e1f_gate":  lambda ch, nb: E1Net(ch, nb, kind="gate"),
}
E1_NB = {k: 16 for k in E1}

# MDTA and windowed attention both split ch across heads; keeping the width a
# multiple of 4 means the solver never hands them a shape they cannot reshape.
STEP = {"e1c_mdta1": 4, "e1d_mdta2": 4, "e1e_win1": 4}


def match_width(name, nb=None, target=TARGET_PARAMS, lo=8, hi=256):
    nb = E1_NB[name] if nb is None else nb
    step = STEP.get(name, 1)
    best = None
    for ch in range(lo + (-lo) % step, hi + 1, step):
        try:
            n = E1[name](ch, nb).n_params()
        except Exception:
            continue
        e = abs(n - target) / float(target)
        if best is None or e < best[2]:
            best = (ch, n, e)
        if n > target * 1.6:
            break
    return best


if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 64)
    print(f"target {TARGET_PARAMS:,} parameters\n")
    print(f"{'row':<12}{'ch':>5}{'nb':>4}{'params':>12}{'err':>8}   forward")
    print("-" * 62)
    for name in E1:
        ch, n, e = match_width(name)
        m = E1[name](ch, E1_NB[name]).eval()
        with torch.no_grad():
            y = m(x)
        ok = tuple(y.shape) == (2, 1, 128, 128) and y.dtype == torch.float32
        print(f"{name:<12}{ch:>5}{E1_NB[name]:>4}{n:>12,}{e:>7.1%}   "
              f"{tuple(y.shape)} {'OK' if ok else 'BAD'}")
