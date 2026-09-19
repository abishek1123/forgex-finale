"""Compute-scalable e1f_gate with explicitly trained restoration exits."""
import torch
import torch.nn as nn

from .attn import E1Net
from .registry import bicubic_up

DEFAULT_EXITS = (3, 6, 10, 13, 16)


def _identity_conv(ch):
    layer = nn.Conv2d(ch, ch, 1)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
        layer.weight[:, :, 0, 0].copy_(torch.eye(ch, dtype=layer.weight.dtype))
    return layer


class MultiExitGate(E1Net):
    """Gate backbone whose selected prefix produces a supervised restoration."""

    def __init__(self, ch=53, nb=16, scale=2, res_scale=0.1,
                 exits=DEFAULT_EXITS, variant="adapter"):
        super().__init__(ch=ch, nb=nb, scale=scale, res_scale=res_scale, kind="gate")
        exits = tuple(sorted(set(int(d) for d in exits)))
        if not exits or exits[-1] != nb or exits[0] < 1:
            raise ValueError(f"exits must be within 1..{nb} and include {nb}: {exits}")
        if variant not in ("shared", "adapter"):
            raise ValueError("variant must be 'shared' or 'adapter'")
        self.exits = exits
        self.variant = variant
        self.adapters = nn.ModuleDict()
        if variant == "adapter":
            self.adapters.update({str(d): _identity_conv(ch) for d in exits if d != nb})
        self.config.update(exits=list(exits), variant=variant, kind="multiexit_gate")

    def _reconstruct(self, stem_feature, body_feature, x, depth):
        f = stem_feature + self.body_tail(body_feature)
        key = str(depth)
        if key in self.adapters:
            f = self.adapters[key](f)
        return bicubic_up(x, self.scale) + self.out(f).float()

    def forward_all(self, x):
        stem_feature = self.head(self.stem(x))
        f = stem_feature
        outputs = {}
        wanted = set(self.exits)
        for depth, block in enumerate(self.body, start=1):
            f = block(f)
            if depth in wanted:
                outputs[depth] = self._reconstruct(stem_feature, f, x, depth)
        return outputs

    def forward_depth(self, x, depth):
        depth = int(depth)
        if depth not in self.exits:
            raise ValueError(f"depth {depth} is not one of {self.exits}")
        stem_feature = self.head(self.stem(x))
        f = stem_feature
        for block in list(self.body)[:depth]:
            f = block(f)
        return self._reconstruct(stem_feature, f, x, depth)

    def forward(self, x):
        return self.forward_depth(x, self.exits[-1])


def load_gate_backbone(model, checkpoint):
    """Load e1f_gate weights while allowing only newly added adapter tensors."""
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {f"adapters.{d}.{part}" for d in model.exits[:-1]
               for part in ("weight", "bias")}
    bad_missing = set(missing) - allowed
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={sorted(bad_missing)} "
                           f"unexpected={sorted(unexpected)}")
    return sorted(set(missing) & allowed)
