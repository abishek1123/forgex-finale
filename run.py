#!/usr/bin/env python3
"""
KLA PS01 - AI-Based Restoration of Degraded Images.  Team ForgeX.

    python run.py <input-dir> <output-dir>

Round 2 scores END-TO-END wall clock on an H100: process start, imports, CUDA
initialisation, reading inputs, inference, and writing outputs.  For a 1.35 M
parameter network the forward pass is a small fraction of that, so this script
is built to overlap the fixed costs rather than pay them in series:

  * CUDA context initialisation starts on a background thread at import time
  * input files are listed, read and parsed on a thread pool WHILE that happens
  * weights are memory-mapped instead of read + unpickled
  * there is no separate warm-up pass; the first real batch is the warm-up
  * outputs are written by a writer pool so the GPU never waits on disk

Self-contained: the network is inlined, so this file depends on nothing but
torch and numpy.  No internet, no keys, no downloads, no manual configuration.
"""
import time
_T0 = time.perf_counter()
_STAGES = []


def _mark(name):
    _STAGES.append((name, time.perf_counter() - _T0))                      # before anything else


# Per-stage timing, OFF by default and zero-overhead when off. Enabled by
# --timing. GPU work is measured with CUDA events (recorded inline, read once at
# the very end) rather than torch.cuda.synchronize() per batch, because a sync
# inside the loop would serialise the pipeline and change the very number we are
# trying to report.
_TIMING = False
_FWD = []            # (start_event, end_event, n_images) or (seconds, n_images)
_H2D = []
_D2H = 0.0
_READ = 0.0
_WRITE = 0.0

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------
# TENSORRT DISPATCH.  This runs ABOVE `import torch`, which is the entire point.
#
# Measured on an H100: `import torch` is 1.445 s of a 2.686 s scored run (53.8%)
# while the model's own arithmetic is 4.7%.  The choice of runtime therefore has
# to be made BEFORE torch is imported -- probing afterwards would already have
# spent the saving.  So this reads .npy HEADERS only (never pixels) to learn the
# shapes, and os.execv()s into the TensorRT runner, which REPLACES the process
# so nothing is imported twice.  The probe costs ~0.12 s.
#
# It falls through to the PyTorch path below -- silently, never raising -- on any
# of: --no-trt, a flag TensorRT cannot honour, no engine index, no engine at the
# requested depth, models/model.pt not matching the engine's recorded sha1, an
# input outside the engine's shape profile, or tensorrt not being installed.
# Being wrong costs a tenth of a second; it never costs a result.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_SAFE_DEPTH = 3                 # single definition; see the note further down
DATASHEET = "knob_datasheet.json"
_ENGINES = os.path.join(HERE, "models", "engines")
_TORCH_ONLY = {"--tta", "--half", "--no-fp16", "--list-knob", "--profile"}
_SHARED_OPTIONS = None


def load_datasheet():
    """-> (sheet, path) or (None, None). Defined up here because the TensorRT
    probe needs it and the probe must run before `import torch`."""
    for rel in (os.path.join("models", DATASHEET), DATASHEET):
        path = os.path.join(HERE, rel)
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    return json.load(fh), path
            except Exception:
                pass
    return None, None


def _flag(argv, name):
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def _hdr_span(d):
    """(min side, max side, count) from .npy headers alone. (0,0,0) on anything odd."""
    import numpy.lib.format as fmt
    lo, hi, n = 10 ** 9, 0, 0
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".npy"):
            continue
        try:
            with open(os.path.join(d, f), "rb") as fh:
                v = fmt.read_magic(fh)
                shp = (fmt.read_array_header_1_0(fh) if v == (1, 0)
                       else fmt.read_array_header_2_0(fh))[0]
        except Exception:
            return 0, 0, 0
        if len(shp) < 2:
            return 0, 0, 0
        lo = min(lo, shp[0], shp[1]); hi = max(hi, shp[0], shp[1]); n += 1
    return (lo, hi, n) if n else (0, 0, 0)


def _wanted_depth(argv, nb_full):
    """The depth the user asked for, resolved the same way resolve_depth() does."""
    b = _flag(argv, "--budget-ms")
    if b is not None:
        try:
            budget = float(b)
        except ValueError:
            return None
        sheet, _ = load_datasheet()
        if sheet is None:
            return None
        key = _flag(argv, "--prefer") or "psnr"
        rows = [r for r in sheet["rows"] if r["ms_per_img"] <= budget]
        if not rows:
            return min(sheet["rows"], key=lambda r: r["ms_per_img"])["depth"]
        return (max(rows, key=lambda r: (r[key], r["depth"])) if all(key in r for r in rows)
                else max(rows, key=lambda r: r["depth"]))["depth"]
    d = _flag(argv, "--depth")
    if d is None:
        return nb_full
    try:
        d = int(d)
    except ValueError:
        return None
    if d <= 0 or d >= nb_full:
        return nb_full
    if d < MIN_SAFE_DEPTH:
        # Same clamp resolve_depth() applies on the PyTorch path. Warn here too,
        # or the TensorRT path would silently honour a different depth than the
        # one the operator typed.
        print("  WARNING: depth %d is below MIN_SAFE_DEPTH=%d; the quality curve is "
              "not monotone there. Clamping to %d."
              % (d, MIN_SAFE_DEPTH, MIN_SAFE_DEPTH), flush=True)
    return max(d, MIN_SAFE_DEPTH)


def _sha1_file(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _trt_dispatch(argv):
    """Never returns if TensorRT can serve this invocation; otherwise returns None."""
    try:
        if "--no-trt" in argv or "--require-trt" in argv or _TORCH_ONLY.intersection(argv) or _flag(argv, "--weights") is not None or _flag(argv, "--knob") is not None:
            return
        if _flag(argv, "--device") == "cpu":
            return
        dirs = [a for a in argv if not a.startswith("-")]
        if not dirs or not os.path.isdir(dirs[0]):
            return
        idx = os.path.join(_ENGINES, "index.json")
        if not os.path.isfile(idx):
            return
        meta = json.load(open(idx))
        engines = meta.get("engines") or []
        if not engines:
            return
        nb_full = meta.get("nb_full") or max(e["depth"] for e in engines)
        want = _wanted_depth(argv, nb_full)
        match = [e for e in engines if e["depth"] == want]
        if not match:
            return
        e = match[0]
        plan = os.path.join(_ENGINES, e.get("plan") or "")
        if not os.path.isfile(plan):
            return
        lo, hi, n = _hdr_span(dirs[0])
        if n == 0 or not (e.get("hmin", 128) <= lo and hi <= e.get("hmax", 128)):
            return
        # An engine is compiled for ONE set of weights. Identity is the sha1,
        # never the filename and never the size.
        w = os.path.join(HERE, "models", "model.pt")
        if e.get("weights_sha1") and os.path.isfile(w):
            if _sha1_file(w) != e["weights_sha1"]:
                return
        import tensorrt  # noqa: F401
    except Exception:
        return
    print("[run.py] TensorRT: %s (depth %s, %s)  probe %.3fs"
          % (e.get("plan"), e.get("depth"), e.get("precision", "?"),
             time.perf_counter() - _T0), flush=True)
    os.execv(sys.executable,
             [sys.executable, os.path.join(HERE, "tools", "trt_infer.py")]
             + list(argv) + ["--engine", plan])


# Returns immediately unless TensorRT can serve this exact request, in which
# case it replaces the process and never returns.
_trt_dispatch(sys.argv[1:])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_mark("import torch + numpy")

# Kick the CUDA context off immediately, on its own thread, so the ~2-4 s of
# driver/cuBLAS/cuDNN initialisation overlaps with reading the input directory.
_CUDA_READY = threading.Event()


_PREWARM = os.environ.get("FORGEX_PREWARM", "") == "1"


def _init_cuda():
    """CUDA context, and optionally the cuDNN/kernel init for OUR conv shapes.

    MEASURED: on an H100 the first real forward costs ~0.199 s of one-time
    cuDNN handle creation and kernel module loading -- 53% of what the profiler
    attributes to "MODEL FORWARD", and 7.4% of the whole run. It is paid once,
    serially, AFTER the checkpoint has loaded.

    Prewarm pays it on this background thread instead, concurrently with the
    0.283 s checkpoint load on the main thread. The dummy network is built from
    the SHIPPED default config (ch 64, nb 16) with random weights and thrown
    away -- it exists only to make CUDA load the kernels. Nothing it computes is
    used, so it cannot affect output.
    """
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            if _PREWARM:
                b = int(os.environ.get("FORGEX_PREWARM_BATCH", "32"))
                warm = Restorer().eval().cuda().to(memory_format=torch.channels_last)
                # prewarm only needs to make CUDA load conv kernels; the exact
                # architecture does not matter, and the shipped one is cheapest.
                x = torch.rand(b, 1, 128, 128, device="cuda").contiguous(
                    memory_format=torch.channels_last)
                with torch.inference_mode(), torch.autocast("cuda", enabled=True):
                    warm(x)
                torch.cuda.synchronize()
                del warm, x
                _mark("prewarm: cuDNN + kernels loaded (background thread)")
    except Exception:
        pass
    finally:
        _mark("CUDA context ready (background thread)")
        _CUDA_READY.set()


# The thread is started AFTER Restorer is defined (below), because prewarm
# instantiates it. Starting it here would race the class definition.


# --------------------------------------------------------------------------
# Model (inlined so run.py has no local imports)
# --------------------------------------------------------------------------
class VarianceStabilisingStem(nn.Module):
    """Speckle variance scales with I^2. Presenting raw / sqrt / log views lets
    the first conv pick the representation where the noise is closest to
    uniform. Zero parameters."""

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


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalisation. NAFNet's, not BatchNorm."""
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.g + self.b


class SimpleGate(nn.Module):
    """a * b on the two halves of the feature map. A MULTIPLICATIVE
    nonlinearity, where ReLU is piecewise linear."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class GateBlock(nn.Module):
    """x + res_scale * c2(SimpleGate(c1(LayerNorm(x)))). c1 widens to 2*ch
    because SimpleGate halves it again, so ch is solved down to 53 to stay on
    the same parameter budget as the 64-wide ReLU network."""
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.c1 = nn.Conv2d(ch, ch * 2, 3, 1, 1)
        self.sg = SimpleGate()
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.c2(self.sg(self.c1(self.norm(x))))


class PixelShuffleTail(nn.Module):
    def __init__(self, ch, scale=2):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1),
                                nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, f):
        return self.tail(self.up(f))


class GateRestorer(nn.Module):
    """e1f_gate. Identical to Restorer except the block interior: ReLU ->
    SimpleGate, plus LayerNorm2d, width 64 -> 53. Stem, head, 16 blocks,
    res_scale 0.1, long skip, body_tail, PixelShuffle x2 and the FP32 global
    bicubic skip are all unchanged."""
    def __init__(self, ch=53, nb=16, scale=2, res_scale=0.1, **_ignored):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[GateBlock(ch, res_scale) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.out = PixelShuffleTail(ch, scale)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        base = F.interpolate(x.float(), scale_factor=self.scale,
                             mode="bicubic", align_corners=False)
        return base + self.out(f).float()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, state=None):
    """Pick the architecture from the checkpoint, not from a flag.

    The gated checkpoint carries kind='gate' in its config; the shipped one has
    no 'kind' key at all. Falling back to the state_dict keys as well means an
    older checkpoint without the marker still resolves correctly.
    """
    kind = (cfg or {}).get("kind", "")
    if not kind and state is not None:
        kind = "gate" if any(k.startswith("out.") or ".norm.g" in k for k in state) else ""
    keep = ("ch", "nb", "scale", "res_scale")
    kw = {k: cfg[k] for k in keep if k in (cfg or {})}
    if kind == "multiexit_gate":
        if cfg.get("variant") != "shared":
            raise ValueError("This runner supports shared-head multi-exit checkpoints, not adapters")
        model = GateRestorer(**kw)
        model.config.update(cfg)
        return model, "shared_gate"
    if kind == "gate":
        return GateRestorer(**kw), "e1f_gate"
    return Restorer(**kw), "forgex"


class Restorer(nn.Module):
    """All convolution at low resolution; single PixelShuffle upsample at the
    end; global bicubic skip so the network predicts only the correction."""

    def __init__(self, ch=64, nb=16, scale=2, res_scale=0.1):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch, res_scale) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1),
                                nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        residual = self.tail(self.up(f))
        base = F.interpolate(x.float(), scale_factor=self.scale,
                             mode="bicubic", align_corners=False)
        return base + residual.float()


_CUDA_THREAD = threading.Thread(target=_init_cuda, daemon=True)
_CUDA_THREAD.start()


# --------------------------------------------------------------------------
# The quality / latency knob
# --------------------------------------------------------------------------
# Truncating the residual trunk is a free knob on this architecture, because of
# three properties it already had:
#   * the body is a flat Sequential at constant width and constant resolution,
#     so block k's output is the same KIND of tensor as block 16's;
#   * res_scale 0.1 makes each block a small additive refinement, so features
#     evolve slowly rather than changing character with depth;
#   * the zero-init tail plus the global bicubic skip mean the network predicts
#     only a CORRECTION to bicubic, so a shallower run produces a SMALLER
#     correction. The failure mode is "falls back toward bicubic", never
#     "produces garbage". Measured: even depth 0 is +2.21 dB over bicubic.
# Measured on the 297-image test set, e1f_gate: 44.0 -> 4.3 GFLOPs (10.4x) costs
# 1.17 dB. Depths 0-2 are NOT monotone, so MIN_SAFE_DEPTH refuses them.
# MIN_SAFE_DEPTH, DATASHEET and load_datasheet() are defined above the torch
# import, because the TensorRT probe needs them and must run before it.


def resolve_depth(args, nb_full):
    """-> (depth or None, one-line explanation). None means full depth."""
    if args.budget_ms > 0:
        sheet, path = load_datasheet()
        if sheet is None:
            print("  WARNING: --budget-ms needs models/knob_datasheet.json; run "
                  "tools/calibrate_knob.py on this machine. Using full depth.", flush=True)
            return None, "full depth (no datasheet)"
        # The budget is resolved against MEASURED timings -- but they were measured
        # on whatever machine ran calibrate_knob.py, not on this one, and nothing
        # here is timed at run time. On different silicon the chosen setting can
        # quietly miss the budget the operator asked for, with no error. Say so.
        sheet_gpu = sheet.get("gpu", "")
        try:
            here_gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                        else "cpu")
        except Exception:
            here_gpu = "cpu"
        if sheet_gpu and here_gpu and sheet_gpu != here_gpu:
            print(f"  WARNING: --budget-ms is resolved from timings measured on "
                  f"{sheet_gpu!r}, but this machine is {here_gpu!r}. The QUALITY "
                  f"columns are hardware-independent; the milliseconds are not, so "
                  f"the selected setting may not meet {args.budget_ms:.3f} ms/img "
                  f"here. Re-measure with:  python tools/calibrate_knob.py --data "
                  f"<test-set> --rounds 5", flush=True)
        rows = [r for r in sheet["rows"] if r["ms_per_img"] <= args.budget_ms]
        if not rows:
            fastest = min(sheet["rows"], key=lambda r: r["ms_per_img"])
            print(f"  WARNING: budget {args.budget_ms:.3f} ms/img is below the fastest "
                  f"setting ({fastest['ms_per_img']:.3f} ms at depth {fastest['depth']}). "
                  f"Using that.", flush=True)
            return fastest["depth"], f"depth {fastest['depth']} (budget unreachable)"
        # Pick the BEST-QUALITY row inside the budget, not the deepest. These differ:
        # the calibration found depth 14 is Pareto-DOMINATED on PSNR -- slower than
        # depth 13 (4.219 vs 3.972 ms) and lower (23.3931 vs 23.4064 dB). "Deepest
        # that fits" would hand the operator a setting that is strictly worse on the
        # headline metric. Ties and missing quality fall back to depth order.
        key = args.prefer
        if all(key in r for r in rows):
            pick = max(rows, key=lambda r: (r[key], r["depth"]))
        else:
            pick = max(rows, key=lambda r: r["depth"])
        q = f", {pick[key].__format__('.4f')} {key}" if key in pick else ""
        return pick["depth"], (f"depth {pick['depth']} of {nb_full} -- "
                               f"{pick['ms_per_img']:.3f} ms/img measured{q}, "
                               f"fits {args.budget_ms:.3f} (max {key})")
    if args.depth > 0:
        d = args.depth
        if d >= nb_full:
            return None, "full depth"
        if d < MIN_SAFE_DEPTH:
            print(f"  WARNING: depth {d} is below MIN_SAFE_DEPTH={MIN_SAFE_DEPTH}; the "
                  f"quality curve is not monotone there. Clamping to {MIN_SAFE_DEPTH}.",
                  flush=True)
            d = MIN_SAFE_DEPTH
        return d, f"depth {d} of {nb_full}"
    return None, "full depth"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def find_weights(explicit=None):
    """models/ is the spec location; weights/ kept as a fallback."""
    if explicit:
        return explicit
    for rel in (os.path.join("models", "model.pt"),
                os.path.join("weights", "model.pt")):
        path = os.path.join(HERE, rel)
        if os.path.isfile(path):
            return path
    sys.exit("ERROR: no model weights found. Expected models/model.pt next to run.py")


def load_checkpoint(path):
    """Memory-map and skip unpickling anything but tensors where possible."""
    for kw in ({"weights_only": True, "mmap": True}, {"weights_only": True}, {}):
        try:
            return torch.load(path, map_location="cpu", **kw)
        except Exception:
            continue
    sys.exit(f"ERROR: could not load weights from {path}")


def load_npy(path):
    """-> (H, W) float32 array, plus whether the input carried a trailing axis."""
    global _READ
    _t = time.perf_counter() if _TIMING else 0.0
    a = np.load(path)
    had_channel = (a.ndim == 3 and a.shape[-1] == 1)
    if had_channel:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"expected (H,W) or (H,W,1), got shape {a.shape}")
    out = np.ascontiguousarray(a, dtype=np.float32), had_channel
    if _TIMING:
        _READ += time.perf_counter() - _t
    return out


def restore_batch(model, arrays, device, use_amp, tta):
    """arrays: list of equally-shaped (H,W) float32 -> list of (2H,2W) float32."""
    global _D2H
    # The frozen shared engines are optional. The existing PyTorch path remains
    # available for incompatible runtimes, flags, sizes and explicit weights.
    if (_SHARED_OPTIONS is not None and not _TIMING and not tta
            and device.type == "cuda" and model.config.get("variant") == "shared"):
        try:
            from tools.shared_backend import try_restore
        except ImportError:
            if getattr(_SHARED_OPTIONS, 'require_trt', False):
                raise RuntimeError('Required TensorRT helper is missing from this package')
            pass  # The minimal four-item submission can still run PyTorch.
        else:
            accelerated = try_restore(model, arrays, _SHARED_OPTIONS, HERE)
            if accelerated is not None:
                return accelerated
    _cuda = device.type == "cuda"
    if _TIMING and _cuda:
        _hs, _he = torch.cuda.Event(True), torch.cuda.Event(True); _hs.record()
    elif _TIMING:
        _t0 = time.perf_counter()
    t = torch.from_numpy(np.stack(arrays))[:, None].to(device)
    if _cuda:
        t = t.contiguous(memory_format=torch.channels_last)
    if _TIMING and _cuda:
        _he.record(); _H2D.append((_hs, _he, len(arrays)))
    elif _TIMING:
        _H2D.append((time.perf_counter() - _t0, len(arrays)))

    if tta:
        variants = [t, t.flip(-1), t.flip(-2), t.flip(-1, -2), t.transpose(-1, -2),
                    t.transpose(-1, -2).flip(-1), t.transpose(-1, -2).flip(-2),
                    t.transpose(-1, -2).flip(-1, -2)]
        undo = [lambda o: o, lambda o: o.flip(-1), lambda o: o.flip(-2),
                lambda o: o.flip(-1, -2), lambda o: o.transpose(-1, -2),
                lambda o: o.flip(-1).transpose(-1, -2),
                lambda o: o.flip(-2).transpose(-1, -2),
                lambda o: o.flip(-1, -2).transpose(-1, -2)]
    else:
        variants, undo = [t], [lambda o: o]

    outs = []
    with torch.inference_mode():
        if _TIMING and _cuda:
            _fs, _fe = torch.cuda.Event(True), torch.cuda.Event(True); _fs.record()
        elif _TIMING:
            _t1 = time.perf_counter()
        for k, v in enumerate(variants):
            with torch.autocast("cuda", enabled=use_amp):
                o = model(v.contiguous()).float()
            outs.append(undo[k](o))
        out = torch.stack(outs).mean(0)
        if _TIMING and _cuda:
            _fe.record(); _FWD.append((_fs, _fe, len(arrays)))
        elif _TIMING:
            _FWD.append((time.perf_counter() - _t1, len(arrays)))

    # Guarantee the contract: finite, float32, inside [0,1].
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    _t2 = time.perf_counter() if _TIMING else 0.0
    out = out[:, 0].cpu().numpy().astype(np.float32)   # implicit sync: D2H is real wall time
    if _TIMING:
        _D2H += time.perf_counter() - _t2
    return [out[i] for i in range(out.shape[0])]


def restore_safe(model, arrays, device, use_amp, tta):
    """restore_batch, but halve the batch and retry on CUDA out-of-memory.

    The VRAM of the evaluation machine is not known in advance. A batch that
    does not fit must produce a SLOWER run, never a crash: a crash scores zero,
    a slow run still scores. Halving recurses down to a single image, which
    needs only a few MB for a 1.35 M-parameter model at 128x128.
    """
    try:
        return restore_batch(model, arrays, device, use_amp, tta)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(arrays) == 1:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        half = len(arrays) // 2
        print(f"  CUDA out of memory at batch {len(arrays)}; retrying at {half}",
              file=sys.stderr, flush=True)
        return (restore_safe(model, arrays[:half], device, use_amp, tta) +
                restore_safe(model, arrays[half:], device, use_amp, tta))


def main():
    p = argparse.ArgumentParser(
        description="Restore degraded inspection images (2x super-resolution + denoising).")
    p.add_argument("input_dir", nargs="?", default=None,
                   help="directory containing degraded .npy files")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="directory to write restored .npy files to")
    p.add_argument("--weights", default=None,
                   help="path to model weights (default: models/model.pt)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="disable half precision on CUDA")
    p.add_argument("--tta", action="store_true",
                   help="8x self-ensemble; higher quality, ~8x slower")
    p.add_argument("--batch", type=int, default=0,
                   help="images per forward pass; 0 = auto (32 GPU / 1 CPU)")
    p.add_argument("--workers", type=int, default=8, help="threads for disk read/write")
    p.add_argument("--profile", action="store_true",
                   help="print where the end-to-end time actually goes")
    p.add_argument("--half", action="store_true",
                   help="store weights in FP16 and run without autocast. Removes "
                        "36 per-call weight casts. The global bicubic skip stays "
                        "FP32 regardless -- it is x.float() inside forward().")
    p.add_argument("--prewarm", action="store_true",
                   help="CHECK ONLY. Prewarm must be enabled with the environment "
                        "variable FORGEX_PREWARM=1, because the background thread "
                        "starts at import time -- before argparse runs, which is "
                        "exactly the point. This flag just verifies it took effect.")
    p.add_argument("--timing", action="store_true",
                   help="per-image breakdown: read, host->device, MODEL FORWARD, "
                        "device->host, write. GPU stages use CUDA events read once "
                        "at the end, so the pipeline is not serialised. Off by "
                        "default and zero-overhead when off.")
    p.add_argument("--depth", type=int, default=0,
                   help="QUALITY/SPEED KNOB: run only the first N residual blocks. "
                        "0 = full depth (default, unchanged behaviour). Lower is "
                        "faster and lower quality; the network degrades toward the "
                        "bicubic baseline, never toward garbage.")
    p.add_argument("--knob", type=int, choices=range(1, 6), default=None,
                   help="trained shared-head exits: 1/2/3/4/5 select 3/6/10/13/16 blocks")
    p.add_argument("--budget-ms", type=float, default=0.0,
                   help="QUALITY/SPEED KNOB: pick the deepest setting whose MEASURED "
                        "ms/image fits this budget, using models/knob_datasheet.json "
                        "(written by tools/calibrate_knob.py on THIS machine).")
    p.add_argument("--list-knob", action="store_true",
                   help="print the calibrated knob datasheet and exit")
    p.add_argument("--prefer", default="psnr", choices=["psnr", "ssim"],
                   help="which metric --budget-ms maximises within the budget")
    p.add_argument("--no-trt", action="store_true",
                   help="force the PyTorch path even when a TensorRT engine could "
                        "serve the request. Consumed by the dispatch probe at the "
                        "top of this file, above `import torch`; it is declared "
                        "here so argparse accepts it and --help documents it.")
    p.add_argument("--require-trt", action="store_true",
                   help="require the matching shared TensorRT engine; fail instead of falling back")
    p.set_defaults(fp16=True)
    a = p.parse_args()
    if a.require_trt and (a.no_trt or a.device == "cpu" or a.tta or a.half
                          or not a.fp16 or a.profile or a.timing):
        p.error("--require-trt conflicts with CPU/PyTorch-only or precision/profiling options")
    if a.knob is not None:
        if a.depth != 0 or a.budget_ms > 0:
            p.error("--knob cannot be combined with --depth or --budget-ms")
        a.depth = (3, 6, 10, 13, 16)[a.knob - 1]
    global _SHARED_OPTIONS
    _SHARED_OPTIONS = a
    global _TIMING
    _TIMING = a.timing

    if a.list_knob:
        sheet, path = load_datasheet()
        if sheet is None:
            sys.exit("no datasheet; run tools/calibrate_knob.py on this machine first")
        print(f"{path}\n  gpu={sheet.get('gpu')}  weights_sha1={sheet.get('weight_sha1')}  "
              f"batch={sheet.get('batch')}  dtype={sheet.get('dtype')}")
        rs = sheet["rows"]; base = rs[-1]
        print(f"\n  {'depth':>6}{'ms/img':>10}{'img/s':>9}{'speedup':>9}{'GFLOPs':>9}"
              f"{'PSNR':>9}{'dPSNR':>8}{'SSIM':>9}{'':>4}")
        for i, r in enumerate(rs):
            dom = any(y.get("psnr", -9) >= r.get("psnr", 9) for y in rs[:i])
            print(f"  {r['depth']:>6}{r['ms_per_img']:>10.3f}{1000/r['ms_per_img']:>9.1f}"
                  f"{base['ms_per_img']/r['ms_per_img']:>8.2f}x{r.get('gflops',float('nan')):>9.2f}"
                  f"{r.get('psnr',float('nan')):>9.4f}"
                  f"{r.get('psnr',float('nan'))-base.get('psnr',float('nan')):>+8.3f}"
                  f"{r.get('ssim',float('nan')):>9.5f}"
                  f"{'  DOM' if dom else '':>6}")
        print("\n  DOM = Pareto-dominated on PSNR (a shallower row is faster AND better).")
        print("  --budget-ms maximises --prefer (default psnr) inside the budget, so it")
        print("  never selects a dominated row.")
        return

    if a.input_dir is None or a.output_dir is None:
        p.error("the following arguments are required: input_dir, output_dir")
    if not os.path.isdir(a.input_dir):
        sys.exit(f"ERROR: input directory not found: {a.input_dir}")
    os.makedirs(a.output_dir, exist_ok=True)

    in_dir = a.input_dir
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".npy"))

    # A benchmark harness may hand us the DATASET directory rather than the
    # images directory -- e.g. semicon_test_data/, which holds GT/ and NoisyLR/.
    # Rather than exit with "no .npy files" and score nothing, look one level
    # down for the degraded inputs and say clearly what we did.
    if not files:
        prefer = ("noisylr", "noisy", "lr", "input", "inputs", "degraded", "test", "images")
        subs = sorted(d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d)))
        cand = [d for d in subs
                if any(f.lower().endswith(".npy") for f in os.listdir(os.path.join(in_dir, d)))]
        pick = None
        for want in prefer:                      # a name that means "the inputs"
            for d in cand:
                if d.lower() == want:
                    pick = d
                    break
            if pick:
                break
        if pick is None and len(cand) == 1:      # only one candidate: unambiguous
            pick = cand[0]
        if pick is not None:
            in_dir = os.path.join(in_dir, pick)
            files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".npy"))
            print(f"note: no .npy at the top level of {a.input_dir}; "
                  f"using {in_dir} ({len(files)} files)", flush=True)
        elif len(cand) > 1:
            sys.exit(f"ERROR: no .npy files in {a.input_dir}, and more than one "
                     f"subdirectory contains them: {cand}. Point me at one of them.")

    if not files:
        listing = sorted(os.listdir(in_dir))[:12]
        sys.exit(f"ERROR: no .npy files found in {a.input_dir}\n"
                 f"       directory contains: {listing}"
                 + ("" if len(listing) < 12 else " ..."))

    if os.path.realpath(in_dir) == os.path.realpath(a.output_dir):
        p.error("output_dir must differ from the input image directory; inputs will not be overwritten")

    # ---- read every input on a thread pool while CUDA is still initialising ----
    _mark("argparse + list input dir")
    reader = ThreadPoolExecutor(max_workers=a.workers)
    futures = [reader.submit(load_npy, os.path.join(in_dir, f)) for f in files]

    weights = find_weights(a.weights)
    ck = load_checkpoint(weights)
    state = ck.get("state_dict", ck)
    cfg = ck.get("config", {}) or {}

    _mark("load checkpoint from disk")
    _CUDA_THREAD.join()
    _mark("join CUDA thread")                       # by now it has almost certainly finished
    device = torch.device("cuda" if (a.device in ("auto", "cuda")
                                     and torch.cuda.is_available()) else "cpu")
    if a.require_trt and device.type != "cuda":
        sys.exit("ERROR: --require-trt needs CUDA and compatible TensorRT engines; no CUDA device is available")
    use_amp = a.fp16 and device.type == "cuda"
    if not a.fp16:
        # Verification requests true FP32, including convolution on CUDA.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if a.batch <= 0:
        a.batch = 32 if device.type == "cuda" else 1

    model, arch_name = build_model(cfg, state)
    model = model.eval()
    model.load_state_dict(state)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    # FP16 weights: removes the 36 per-call weight casts autocast would do.
    # forward() computes the bicubic base from x.float(), so the global skip and
    # the final addition stay FP32 whatever the weights are.
    if a.half and device.type == "cuda":
        model = model.half()
        use_amp = False
    if a.prewarm and not _PREWARM:
        print("  WARNING: --prewarm has no effect. Set FORGEX_PREWARM=1 in the "
              "environment instead.", flush=True)

    nb_full = len(model.body)
    depth, knob_note = resolve_depth(a, nb_full)
    if depth is not None:
        model.body = nn.Sequential(*list(model.body)[:depth])
    if a.require_trt and (cfg.get("variant") != "shared" or len(model.body) not in (3, 6, 10, 13, 16)):
        sys.exit("ERROR: --require-trt needs the shared checkpoint at depth 3, 6, 10, 13 or 16")

    _mark("build model + weights to device")
    n_params = sum(q.numel() for q in model.parameters())
    wdt = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"device={device}  arch={arch_name}  weights={wdt}  autocast={use_amp}  "
          f"tta={a.tta}  params={n_params/1e6:.2f}M  files={len(files)}  "
          f"batch={a.batch}  prewarm={_PREWARM}  knob={knob_note}", flush=True)

    # ---- group by shape, infer, hand writing off to the writer pool ----
    writer = ThreadPoolExecutor(max_workers=a.workers)
    writes = []
    pending = {}

    def save(name, arr):
        global _WRITE
        _t = time.perf_counter() if _TIMING else 0.0
        np.save(os.path.join(a.output_dir, name), arr)
        if _TIMING:
            _WRITE += time.perf_counter() - _t

    def flush(shape):
        group = pending.pop(shape, [])
        for i in range(0, len(group), a.batch):
            chunk = group[i:i + a.batch]
            ys = restore_safe(model, [g[1] for g in chunk], device, use_amp, a.tta)
            for (name, _, had_channel), y in zip(chunk, ys):
                writes.append(writer.submit(save, name, y[..., None] if had_channel else y))

    for name, fut in zip(files, futures):
        arr, had_channel = fut.result()
        pending.setdefault(arr.shape, []).append((name, arr, had_channel))
        if len(pending[arr.shape]) >= a.batch:
            flush(arr.shape)
    for shape in list(pending):
        flush(shape)

    _mark("read all inputs + inference + queue writes")
    for w in writes:
        w.result()
    reader.shutdown(wait=True)
    writer.shutdown(wait=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if _TIMING:
        n = len(files)
        if device.type == "cuda":
            torch.cuda.synchronize()          # first and only sync: read the events
            fwd = sum(a_.elapsed_time(b_) for a_, b_, _ in _FWD) / 1000.0
            h2d = sum(a_.elapsed_time(b_) for a_, b_, _ in _H2D) / 1000.0
        else:
            fwd = sum(x for x, _ in _FWD)
            h2d = sum(x for x, _ in _H2D)
        tot = time.perf_counter() - _T0
        St = dict(_STAGES)
        imp = St.get("import torch + numpy", 0.0)
        ckpt = (St.get("build model + weights to device", 0.0)
                - St.get("argparse + list input dir", 0.0))
        gpu = h2d + fwd + _D2H
        print(f"\n  timing over {n} images  (batch {a.batch}, "
              f"{'fp16' if use_amp else 'fp32'}, {knob_note})")
        print(f"\n  FIXED COST (paid once, independent of image count)")
        print(f"  {'stage':<36}{'seconds':>10}{'% of e2e':>10}")
        for nm, v in [("import torch + numpy", imp),
                      ("checkpoint load + model to device", ckpt)]:
            print(f"  {nm:<36}{v:>10.3f}{100*v/tot:>9.1f}%")
        print(f"\n  PER-IMAGE GPU PATH (serial -- these do add up)")
        print(f"  {'stage':<36}{'seconds':>10}{'ms/image':>11}{'% of e2e':>10}")
        for nm, v in [("host -> device transfer", h2d),
                      ("MODEL FORWARD", fwd),
                      ("device -> host + postprocess", _D2H)]:
            print(f"  {nm:<36}{v:>10.3f}{v/n*1000:>11.4f}{100*v/tot:>9.1f}%")
        print(f"  {'':<36}{'-'*10}{'-'*11}{'-'*10}")
        print(f"  {'GPU path total':<36}{gpu:>10.3f}{gpu/n*1000:>11.4f}{100*gpu/tot:>9.1f}%")
        print(f"\n  OVERLAPPED THREAD POOLS ({a.workers} workers each)")
        print(f"  These are THREAD-seconds summed across workers, not wall-seconds.")
        print(f"  They run concurrently with the GPU path and with each other, so")
        print(f"  they must NOT be added to the totals above.")
        print(f"  {'stage':<36}{'thread-s':>10}{'ms/image':>11}")
        for nm, v in [("read + parse .npy", _READ), ("write .npy", _WRITE)]:
            print(f"  {nm:<36}{v:>10.3f}{v/n*1000:>11.4f}")
        print(f"\n  {'END-TO-END':<36}{tot:>10.3f}{tot/n*1000:>11.4f}{100.0:>9.1f}%")
        print(f"\n  MODEL-ONLY INFERENCE: {fwd/n*1000:.4f} ms/image  "
              f"({n/fwd:.1f} img/s)  = {100*fwd/tot:.1f}% of end-to-end")
        sheet, _p = load_datasheet()
        if sheet:
            row = next((r for r in sheet["rows"]
                        if r["depth"] == (depth if depth is not None else nb_full)), None)
            if row and row.get("ms_per_img"):
                d_ms, r_ms = row["ms_per_img"], fwd / n * 1000
                print(f"  datasheet says {d_ms:.4f} ms/image for this setting -- "
                      f"measured here is {r_ms/d_ms:.2f}x that.")
                print(f"  (the datasheet runs with cudnn.benchmark=True on synthetic")
                print(f"   input; run.py leaves cudnn.benchmark at its default)")

    _mark("flush all writes to disk")
    dt = time.perf_counter() - _T0
    written = len([f for f in os.listdir(a.output_dir) if f.lower().endswith(".npy")])
    print(f"restored {len(files)} images  |  end-to-end {dt:.2f}s "
          f"({dt/len(files)*1000:.1f} ms/image, measured the way KLA measures it)")
    print(f"wrote {written} files to {a.output_dir}")
    if a.profile:
        print("\n  where the end-to-end time goes")
        print(f"  {'stage':<42}{'cumulative':>12}{'this stage':>12}{'% total':>9}")
        prev = 0.0
        for name, t in _STAGES:
            print(f"  {name:<42}{t:>11.3f}s{t-prev:>11.3f}s{100*(t-prev)/dt:>8.1f}%")
            prev = t
        print(f"  {'(interpreter start, before our first mark)':<42}{'':>12}{_STAGES[0][1]:>11.3f}s"
              f"{100*_STAGES[0][1]/dt:>8.1f}%" if _STAGES else "")
        print(f"\n  model compute is roughly {len(files)}x{1000*0.0:.0f} of this; everything else is fixed cost.")
    if written < len(files):
        sys.exit(f"ERROR: expected {len(files)} outputs, found {written}")


if __name__ == "__main__":
    main()
