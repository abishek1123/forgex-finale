#!/usr/bin/env python3
"""ForgeX entry point: TensorRT when it can serve the request, PyTorch otherwise.

    python run_fast.py <input-dir> <output-dir> [any run.py flag] [--no-trt]

Measured on H100: `import torch` is 1.445 s of a 2.686 s run (53.8%), while the
actual per-image compute is 4.7%. A TensorRT process starts in 0.52 s. So the
decision has to be taken BEFORE torch is imported -- probing with torch already
loaded would spend the entire saving.

This file therefore imports numpy and json only, reads .npy HEADERS (never
pixels) to learn the shapes, and os.execv()s into the right runner. execv
REPLACES the process, so nothing is imported twice; the probe costs ~0.12 s.

THE KNOB AND THE ENGINES. --depth truncates the residual trunk, which changes
the graph, so one plan cannot serve every depth. We ship one engine per
published depth (models/engines/index.json) and select by exact depth match.
--budget-ms is resolved here from models/knob_datasheet.json with the SAME rule
run.py uses -- best quality inside the budget, never merely the deepest -- so
the budget knob reaches TensorRT too.

Falls back to run.py -- unchanged, 13/13 stress -- whenever TensorRT is
missing, no engine matches the depth, the engine was built for another GPU or
weights, an input lies outside the shape profile, or a flag TensorRT cannot
honour is present (--tta, --device cpu, --half). Being wrong costs a tenth of a
second, never correctness.
"""
import os, sys, time
_T0 = time.perf_counter()
HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "models", "engines")
INDEX = os.path.join(ENG, "index.json")
MIN_SAFE_DEPTH = 3                       # mirrors run.py
TORCH_ONLY = {"--tta", "--half", "--no-fp16", "--list-knob", "--profile"}


def _flag_value(args, name):
    for i, x in enumerate(args):
        if x == name and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def _span(d):
    """(min_side, max_side, count) across the folder, from .npy headers only."""
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


def _resolve_depth(args, nb_full, allowed=None):
    """Same rule as run.py.resolve_depth, without importing torch.

    `allowed` restricts the --budget-ms search to depths that actually have
    an engine. Without it the solver would happily return depth 11, find no
    plan, and fall back to PyTorch -- throwing away the 49% startup saving to
    honour a distinction of 0.03 dB. Explicit --depth is NEVER substituted:
    the operator asked for a specific quality point and gets exactly it, on
    the PyTorch path if no engine matches."""
    import json
    b = _flag_value(args, "--budget-ms")
    if b is not None:
        try:
            budget = float(b)
        except ValueError:
            return None, "bad --budget-ms"
        sheet = None
        for rel in (os.path.join("models", "knob_datasheet.json"), "knob_datasheet.json"):
            p = os.path.join(HERE, rel)
            if os.path.isfile(p):
                try:
                    sheet = json.load(open(p)); break
                except Exception:
                    pass
        if sheet is None:
            return None, "no datasheet"
        rows = [r for r in sheet["rows"] if r["ms_per_img"] <= budget
                and (allowed is None or r["depth"] in allowed)]
        if not rows:
            pool = [r for r in sheet["rows"]
                    if allowed is None or r["depth"] in allowed]
            if not pool:
                return None, "no engine-backed row in the datasheet"
            f = min(pool, key=lambda r: r["ms_per_img"])
            return f["depth"], ("budget %.3f ms below the fastest engine "
                                "(%.3f ms at depth %d) -> that one"
                                % (budget, f["ms_per_img"], f["depth"]))
        key = _flag_value(args, "--prefer") or "psnr"
        pick = (max(rows, key=lambda r: (r[key], r["depth"]))
                if all(key in r for r in rows) else max(rows, key=lambda r: r["depth"]))
        return pick["depth"], "budget %.3f ms -> depth %d (max %s)" % (budget, pick["depth"], key)
    d = _flag_value(args, "--depth")
    if d is not None:
        try:
            d = int(d)
        except ValueError:
            return None, "bad --depth"
        if d <= 0 or d >= nb_full:
            return nb_full, "full depth"
        return max(d, MIN_SAFE_DEPTH), "depth %d" % max(d, MIN_SAFE_DEPTH)
    return nb_full, "full depth"


def _sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pick(args, lo, hi):
    """-> (plan path or None, reason)."""
    import json
    if not os.path.isfile(INDEX):
        return None, "no engine index (models/engines/index.json)"
    try:
        idx = json.load(open(INDEX))
    except Exception as e:
        return None, "unreadable index (%s)" % type(e).__name__
    engines = idx.get("engines") or []
    if not engines:
        return None, "engine index is empty"
    nb_full = idx.get("nb_full") or max(e["depth"] for e in engines)

    depth, why = _resolve_depth(args, nb_full, {e["depth"] for e in engines})
    if depth is None:
        return None, why
    match = [e for e in engines if e["depth"] == depth]
    if not match:
        return None, "%s -- no engine at depth %d (have %s)" % (
            why, depth, ",".join(str(e["depth"]) for e in engines))
    e = match[0]
    plan = os.path.join(ENG, e.get("plan") or "")
    if not os.path.isfile(plan):
        return None, "engine file missing: %s" % e.get("plan")

    if not (e.get("hmin", 128) <= lo and hi <= e.get("hmax", 128)):
        return None, "inputs %d-%d outside profile %d-%d" % (
            lo, hi, e.get("hmin", 128), e.get("hmax", 128))

    # The engine is compiled machine code for ONE set of weights. Identity is
    # the checkpoint sha1 -- never the filename, never the size.
    w = os.path.join(HERE, "models", "model.pt")
    if "--no-hash-check" not in args and e.get("weights_sha1") and os.path.isfile(w):
        if _sha1(w) != e["weights_sha1"]:
            return None, "models/model.pt sha1 does not match the engine's weights"

    try:
        import tensorrt  # noqa: F401
    except Exception as ex:
        return None, "tensorrt unavailable (%s)" % type(ex).__name__
    return plan, "%s -> %s (%s, batch %d-%d, hw %d-%d)" % (
        why, e.get("plan"), e.get("precision", "?"),
        e.get("bmin", 1), e.get("bmax", 1), e.get("hmin", 0), e.get("hmax", 0))


def main():
    args = sys.argv[1:]
    dirs = [a for a in args if not a.startswith("-")]
    blocked = sorted(TORCH_ONLY.intersection(args))
    dev = _flag_value(args, "--device")
    plan = None

    if "--no-trt" in args:
        why = "forced with --no-trt"
    elif blocked:
        why = "%s needs the PyTorch path" % ",".join(blocked)
    elif dev == "cpu":
        why = "--device cpu"
    elif not dirs or not os.path.isdir(dirs[0]):
        why = "no readable input dir"
    else:
        lo, hi, n = _span(dirs[0])
        if n == 0:
            why = "no .npy at top level; run.py will search subdirs"
        else:
            plan, why = _pick(args, lo, hi)

    print("[run_fast] %s: %s  (probe %.3fs)"
          % ("TensorRT" if plan else "PyTorch", why, time.perf_counter() - _T0),
          flush=True)
    if plan:
        os.execv(sys.executable,
                 [sys.executable, os.path.join(HERE, "tools", "trt_infer.py")]
                 + [a for a in args if a != "--no-hash-check"] + ["--engine", plan])
    os.execv(sys.executable,
             [sys.executable, os.path.join(HERE, "run.py")]
             + [a for a in args if a not in ("--no-trt", "--no-hash-check")])


if __name__ == "__main__":
    main()
