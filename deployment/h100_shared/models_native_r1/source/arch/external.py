"""
Official SwinIR / NAFNet / Restormer, adapted to this task without editing them.

The three files downloaded next to this one are left BYTE-UNTOUCHED, so their
provenance is defensible: "official implementation, adapted at load time."
Adaptation happens here instead, and it is only ever three things:

  1. basicsr is not installed, so the two helpers NAFNet imports from it are
     supplied as in-process module shims (LayerNorm2d is reproduced exactly;
     Local_Base is only used by NAFNetLocal, which we never build).
  2. img_channel / in_chans 3 -> 1. This is grayscale SEM.
  3. NAFNet and Restormer output at INPUT resolution, so their final 1x1 is
     replaced by a PixelShuffle x2 tail. SwinIR needs none of this -- it has a
     native upscale=2 path, which is why it is the cleanest transformer
     baseline available.

Every model still gets the global bicubic skip, because without it a zero-init
tail cannot learn and the comparison would measure initialisation rather than
architecture. The variance-stabilising stem is OURS and is switchable per
model: default OFF, so `nafnet` and `restormer` are the published methods, and
`nafnet_ours` / `restormer_ours` test whether our stem transfers to someone
else's backbone. That second result is worth more to the paper than winning.

    python arch/external.py        # width search + forward test, in situ
"""
import inspect
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_PARAMS = 1_368_705


# ------------------------------------------------------------ basicsr shims
class LayerNorm2d(nn.Module):
    """Reproduces basicsr.models.archs.arch_util.LayerNorm2d: normalise across
    channels at each spatial location, then a per-channel affine."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * y + self.bias[None, :, None, None]


class Local_Base:                      # only NAFNetLocal needs it; never built
    def convert(self, *a, **k):
        raise NotImplementedError("NAFNetLocal is not used in this study")


def _install_basicsr_shims():
    if "basicsr.models.archs.arch_util" in sys.modules:
        return
    for name in ("basicsr", "basicsr.models", "basicsr.models.archs"):
        sys.modules.setdefault(name, types.ModuleType(name))
    au = types.ModuleType("basicsr.models.archs.arch_util")
    au.LayerNorm2d = LayerNorm2d
    sys.modules["basicsr.models.archs.arch_util"] = au
    la = types.ModuleType("basicsr.models.archs.local_arch")
    la.Local_Base = Local_Base
    sys.modules["basicsr.models.archs.local_arch"] = la


def _load(fname, what):
    """Import a downloaded official file by path, without a package."""
    import importlib.util
    path = os.path.join(HERE, fname)
    if not os.path.isfile(path):
        raise SystemExit(
            f"missing {fname}. Download it first:\n"
            f"  cd {HERE}\n  curl.exe -L -o {fname} <official raw url>")
    _install_basicsr_shims()
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, what)


def _kw(cls, **want):
    """Pass only the kwargs this version of the class actually accepts."""
    ok = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in want.items() if k in ok}


# ------------------------------------------------------------------ shared
class Stem(nn.Module):
    def forward(self, x):
        p = x.clamp_min(0.0)
        return torch.cat([x, torch.sqrt(p + 1e-6), torch.log1p(p)], dim=1)


class Tail(nn.Module):
    def __init__(self, ch, scale=2):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1), nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, f):
        return self.tail(self.up(f))


def bicubic_up(x, scale=2):
    return F.interpolate(x.float(), scale_factor=scale, mode="bicubic", align_corners=False)


class _SR(nn.Module):
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ NAFNet
class NAFNetSR(_SR):
    """Official NAFNet UNet body; `ending` swapped for a PixelShuffle x2 tail.
    Block layout is the paper's SIDD shape; only the width is scaled to match
    our parameter budget, so the comparison is at matched capacity."""
    ENC, MID, DEC = [1, 1, 1], 2, [1, 1, 1]

    def __init__(self, ch=16, nb=None, scale=2, stem=False):
        super().__init__()
        NAF = _load("nafnet_official.py", "NAFNet")
        self.config = dict(ch=ch, nb=nb, scale=scale, stem=stem, kind="nafnet")
        self.scale, self.use_stem = scale, stem
        self.stem = Stem() if stem else None
        self.net = NAF(**_kw(NAF, img_channel=3 if stem else 1, width=ch,
                             middle_blk_num=self.MID,
                             enc_blk_nums=list(self.ENC), dec_blk_nums=list(self.DEC)))
        del self.net.ending                       # we upsample instead
        self.net.ending = nn.Identity()
        self.out = Tail(ch, scale)

    def forward(self, x):
        inp = self.stem(x) if self.use_stem else x
        n = self.net
        z = n.check_image_size(inp)
        f = n.intro(z)
        encs = []
        for enc, down in zip(n.encoders, n.downs):
            f = enc(f); encs.append(f); f = down(f)
        f = n.middle_blks(f)
        for dec, up, sk in zip(n.decoders, n.ups, encs[::-1]):
            f = up(f); f = f + sk; f = dec(f)
        f = f[:, :, :inp.shape[-2], :inp.shape[-1]]
        return bicubic_up(x, self.scale) + self.out(f).float()


# --------------------------------------------------------------- Restormer
class RestormerSR(_SR):
    """Restormer has no upsampling path -- it restores at input resolution. To
    use it for x2 SR it must therefore run at OUTPUT resolution, on the
    bicubic-upsampled image. That is 4x the compute of an LR-space design, and
    it is the honest cost of this architecture on this task: the speed column
    is measuring something real, not a handicap we imposed."""

    BLOCKS, REFINE = [2, 2, 2, 2], 2

    def __init__(self, ch=16, nb=None, scale=2, stem=False, blocks=None, refine=None):
        super().__init__()
        if blocks is not None:
            self.BLOCKS = list(blocks)
        if refine is not None:
            self.REFINE = refine
        R = _load("restormer_official.py", "Restormer")
        self.config = dict(ch=ch, nb=nb, scale=scale, stem=stem, kind="restormer",
                           blocks=list(self.BLOCKS), refine=self.REFINE)
        self.scale, self.use_stem = scale, stem
        self.stem = Stem() if stem else None
        c = 3 if stem else 1
        self.net = R(**_kw(R, inp_channels=c, out_channels=c, dim=ch,
                           num_blocks=list(self.BLOCKS), num_refinement_blocks=self.REFINE,
                           heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                           bias=False, LayerNorm_type="WithBias", dual_pixel_task=False))
        self.gain = nn.Parameter(torch.zeros(1))      # start at the bicubic base

    def forward(self, x):
        base = bicubic_up(x, self.scale)
        inp = self.stem(base) if self.use_stem else base
        h, w = inp.shape[-2:]
        pad = 16
        ph, pw = (pad - h % pad) % pad, (pad - w % pad) % pad
        if ph or pw:
            inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        y = self.net(inp)[:, :, :h, :w]
        if y.shape[1] != 1:
            y = y.mean(1, keepdim=True)
        return base + self.gain * y.float()


# ------------------------------------------------------------------ SwinIR
class SwinIRSR(_SR):
    """Official SwinIR with its native upscale=2 path -- no tail surgery needed.

    It DOES still get the global bicubic skip, like every other row: SwinIR is
    given the residual to predict, scaled by a zero-init gain so training starts
    exactly at bicubic. Without it, SwinIR would be the only architecture in the
    study learning from scratch, and the table would measure initialisation
    rather than architecture.
    """

    def __init__(self, ch=60, nb=4, scale=2, stem=False, window=8, depth=6):
        super().__init__()
        S = _load("swinir_official.py", "SwinIR")
        self.config = dict(ch=ch, nb=nb, scale=scale, stem=stem, kind="swinir")
        self.scale, self.use_stem = scale, stem
        self.stem = Stem() if stem else None
        heads = max(2, ch // 15)
        self.net = S(**_kw(S, img_size=64, patch_size=1, in_chans=3 if stem else 1,
                           embed_dim=ch, depths=[depth] * nb,
                           num_heads=[heads] * nb, window_size=window,
                           mlp_ratio=2.0, upscale=scale, img_range=1.0,
                           upsampler="pixelshuffledirect", resi_connection="1conv"))
        self.gain = nn.Parameter(torch.zeros(1))      # start at the bicubic base

    def forward(self, x):
        inp = self.stem(x) if self.use_stem else x
        h, w = inp.shape[-2:]
        ws = 8
        ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
        if ph or pw:
            inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        y = self.net(inp)[:, :, :h * self.scale, :w * self.scale]
        if y.shape[1] != 1:
            y = y.mean(1, keepdim=True)
        return bicubic_up(x, self.scale) + self.gain * y.float()


_R_CACHE = {}


def _R_BEST(name):
    """Restormer's width must stay a multiple of 8, so the budget is matched by
    searching its block layout instead. Cached -- the search builds models."""
    if name not in _R_CACHE:
        _R_CACHE[name] = match_width(name)
    return _R_CACHE[name]


EXTERNAL = {
    "nafnet":        lambda ch, nb: NAFNetSR(ch, nb, stem=False),
    "nafnet_ours":   lambda ch, nb: NAFNetSR(ch, nb, stem=True),
    "restormer":     lambda ch, nb: RestormerSR(ch, nb, stem=False,
                                                 blocks=_R_BEST("restormer")[3],
                                                 refine=_R_BEST("restormer")[4]),
    "restormer_ours": lambda ch, nb: RestormerSR(ch, nb, stem=True,
                                                 blocks=_R_BEST("restormer_ours")[3],
                                                 refine=_R_BEST("restormer_ours")[4]),
    "swinir":        lambda ch, nb: SwinIRSR(ch, nb, stem=False),
}
EXTERNAL_NB = {"nafnet": None, "nafnet_ours": None,
               "restormer": None, "restormer_ours": None, "swinir": 4}


# Restormer splits every level's channels across heads [1,2,4,8], so the width
# must stay divisible by 8 or the attention reshape fails at level 4.
STEP = {"restormer": 8, "restormer_ours": 8}


# Restormer's width must stay divisible by 8, which makes parameter count jump
# in very coarse steps (ch=16 -> 1.18 M, ch=24 -> 2.6 M). The budget is matched
# by varying the block layout at a fixed width instead.
RESTORMER_LAYOUTS = [([2, 2, 2, 2], 2), ([2, 2, 2, 2], 3), ([2, 2, 2, 2], 4),
                     ([2, 2, 2, 3], 2), ([2, 2, 2, 3], 3), ([2, 2, 3, 3], 2),
                     ([2, 2, 3, 3], 3), ([2, 3, 3, 3], 2), ([2, 3, 3, 4], 2),
                     ([3, 3, 4, 4], 4), ([4, 4, 4, 4], 4), ([4, 6, 6, 8], 4)]


def match_width(name, target=TARGET_PARAMS, lo=4, hi=200):
    if name.startswith("restormer"):
        stem = name.endswith("_ours")
        best = None
        for blocks, refine in RESTORMER_LAYOUTS:
            for ch in range(8, 65, 8):
                try:
                    n = RestormerSR(ch, None, stem=stem, blocks=blocks,
                                    refine=refine).n_params()
                except Exception:
                    continue
                e = abs(n - target) / target
                if best is None or e < best[2]:
                    best = (ch, n, e, blocks, refine)
        return best
    nb = EXTERNAL_NB[name]
    step = STEP.get(name, 1)
    lo = max(lo, step)
    best = None
    for ch in range(lo - lo % step if lo % step == 0 else lo + step - lo % step, hi + 1, step):
        try:
            n = EXTERNAL[name](ch, nb).n_params()
        except Exception:
            continue
        e = abs(n - target) / target
        if best is None or e < best[2]:
            best = (ch, n, e)
        if n > target * 1.8:
            break
    return best


if __name__ == "__main__":
    print(f"target {TARGET_PARAMS:,} parameters\n")
    print(f"{'arch':<16}{'ch':>5}{'params':>12}{'err':>8}   forward(2,1,128,128)")
    print("-" * 70)
    x = torch.randn(2, 1, 128, 128)
    for name in EXTERNAL:
        try:
            ch, n, e = match_width(name)[:3]
            m = EXTERNAL[name](ch, EXTERNAL_NB[name]).eval()
            with torch.no_grad():
                y = m(x)
            ok = tuple(y.shape) == (2, 1, 256, 256) and y.dtype == torch.float32
            print(f"{name:<16}{ch:>5}{n:>12,}{e:>7.1%}   {tuple(y.shape)} "
                  f"{'OK' if ok else 'BAD SHAPE/DTYPE'}")
        except Exception as ex:
            print(f"{name:<16}    -           -       -   FAILED: {type(ex).__name__}: {ex}")
