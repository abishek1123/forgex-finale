#!/usr/bin/env python3
"""ForgeX inference WITHOUT PyTorch. tensorrt + cuda-python + numpy only.

    python tools/trt_infer.py <in-dir> <out-dir> --engine models/engines/gate_d16.plan

WHY: measured on H100, framework startup is 54% of a scored 2.69 s run
(`import torch` = 1.445 s) while actual per-image compute is 4.7%. The TensorRT
stack starts in 0.52 s. This process therefore imports NO torch. The engine
carries the whole graph -- variance-stabilising stem, trunk, PixelShuffle tail
and the FP32 bicubic skip -- so nothing needs a tensor library at runtime.

WIDE: images are grouped by (H,W), so a folder may mix sizes and aspect ratios.
Any group outside the engine's optimisation profile, or any TensorRT failure
before the first output is written, EXECs into run.py -- the PyTorch path,
13/13 stress -- so a bad engine costs a tenth of a second, never a result.

Output contract identical to run.py: float32, (2H,2W), finite, clamped to
[0,1], trailing singleton axis preserved when the input had one.
"""
import time
_T0 = time.perf_counter()
_ST = []
def _mark(n): _ST.append((n, time.perf_counter() - _T0))

import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
_mark("import numpy")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fallback(why):
    """Hand the whole invocation to run.py. Never returns."""
    print("[trt_infer] falling back to PyTorch: %s" % why, flush=True)
    args, skip = [], False
    for x in sys.argv[1:]:
        if skip:
            skip = False; continue
        if x == "--engine":
            skip = True; continue
        if x.startswith("--engine="):
            continue
        args.append(x)
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "run.py")] + args)


try:
    import tensorrt as trt
    _mark("import tensorrt")
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.9
    except ImportError:
        from cuda import cudart                      # older layout
    _mark("import cuda bindings")
except Exception as _e:                              # never reached via run_fast,
    _fallback("%s: %s" % (type(_e).__name__, _e))    # but this file must stand alone


def _chk(r):
    err = r[0] if isinstance(r, tuple) else r
    if int(err) != 0:
        raise RuntimeError("CUDA error %s" % err)
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None


def load_npy(path):
    a = np.load(path)
    had = (a.ndim == 3 and a.shape[-1] == 1)
    if had:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError("expected (H,W), got %s" % (a.shape,))
    return np.ascontiguousarray(a, dtype=np.float32), had


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir"); p.add_argument("output_dir")
    p.add_argument("--engine", default=os.path.join(HERE, "models", "engines",
                                                    "gate_d16.plan"))
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timing", action="store_true")
    a, _unknown = p.parse_known_args()
    os.makedirs(a.output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(a.input_dir) if f.lower().endswith(".npy"))
    if not files:
        _fallback("no .npy at top level of %s" % a.input_dir)

    meta = {}
    if os.path.isfile(a.engine + ".json"):
        meta = json.load(open(a.engine + ".json"))
    hmin = meta.get("hmin", 128); hmax = meta.get("hmax", 128)
    bmax = meta.get("bmax", 32)
    batch = a.batch if a.batch > 0 else min(32, bmax)
    batch = max(1, min(batch, bmax))

    reader = ThreadPoolExecutor(max_workers=a.workers)
    futs = [reader.submit(load_npy, os.path.join(a.input_dir, f)) for f in files]
    _mark("list + queue reads")

    try:
        _chk(cudart.cudaFree(0))                      # create the CUDA context
        _mark("cuda context")
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(a.engine, "rb") as fh:
            eng = rt.deserialize_cuda_engine(fh.read())
        if eng is None:
            raise RuntimeError("engine deserialise failed (wrong GPU arch or TRT version)")
        ctx = eng.create_execution_context()
        _mark("engine load")
    except Exception as e:
        reader.shutdown(wait=False)
        _fallback("%s: %s" % (type(e).__name__, e))

    arrs = [f.result() for f in futs]
    _mark("all inputs read")

    # Group by (H,W): a folder may mix sizes. Every group must fit the profile.
    groups = {}
    for i, (arr, _had) in enumerate(arrs):
        groups.setdefault(arr.shape, []).append(i)
    bad = [s for s in groups if not (hmin <= min(s) and max(s) <= hmax)]
    if bad:
        reader.shutdown(wait=False)
        _fallback("shape(s) %s outside engine profile %d-%d"
                  % (sorted(bad)[:3], hmin, hmax))

    writer = ThreadPoolExecutor(max_workers=a.workers)
    writes = []
    stream = _chk(cudart.cudaStreamCreate())
    dev_in = dev_out = None
    cap_in = cap_out = 0
    try:
        for (H, W), idxs in sorted(groups.items()):
            for s in range(0, len(idxs), batch):
                sel = idxs[s:s + batch]
                b = len(sel)
                x = np.ascontiguousarray(
                    np.stack([arrs[i][0] for i in sel])[:, None])     # (b,1,H,W)
                y = np.empty((b, 1, H * 2, W * 2), dtype=np.float32)
                if x.nbytes > cap_in:
                    if dev_in is not None:
                        _chk(cudart.cudaFree(dev_in))
                    dev_in = _chk(cudart.cudaMalloc(x.nbytes)); cap_in = x.nbytes
                if y.nbytes > cap_out:
                    if dev_out is not None:
                        _chk(cudart.cudaFree(dev_out))
                    dev_out = _chk(cudart.cudaMalloc(y.nbytes)); cap_out = y.nbytes
                _chk(cudart.cudaMemcpyAsync(dev_in, x.ctypes.data, x.nbytes,
                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
                ctx.set_input_shape("lr", (b, 1, H, W))
                ctx.set_tensor_address("lr", int(dev_in))
                ctx.set_tensor_address("sr", int(dev_out))
                if not ctx.execute_async_v3(stream):
                    raise RuntimeError("execute_async_v3 returned False")
                _chk(cudart.cudaMemcpyAsync(y.ctypes.data, dev_out, y.nbytes,
                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
                _chk(cudart.cudaStreamSynchronize(stream))
                y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                np.clip(y, 0.0, 1.0, out=y)
                for k, i in enumerate(sel):
                    arr = y[k, 0]
                    writes.append(writer.submit(
                        np.save, os.path.join(a.output_dir, files[i]),
                        arr[..., None] if arrs[i][1] else arr))
    except Exception as e:
        if not writes:
            reader.shutdown(wait=False); writer.shutdown(wait=False)
            _fallback("%s: %s" % (type(e).__name__, e))
        raise
    _mark("inference + queue writes")

    for w in writes:
        w.result()
    reader.shutdown(wait=True); writer.shutdown(wait=True)
    _mark("flush writes")

    dt = time.perf_counter() - _T0
    print("restored %d images  |  end-to-end %.2fs (%.1f ms/image, %.1f img/s, NO PYTORCH)"
          % (len(files), dt, dt / len(files) * 1000, len(files) / dt), flush=True)
    print("  engine %s  depth %s  batch %d  shapes %s"
          % (os.path.basename(a.engine), meta.get("depth", "?"), batch,
             sorted({"%dx%d" % s for s in groups})), flush=True)
    if a.timing:
        print("\n  %-28s%12s%12s%8s" % ("stage", "cumulative", "this stage", "%"))
        prev = 0.0
        for n, t in _ST:
            print("  %-28s%11.3fs%11.3fs%7.1f%%" % (n, t, t - prev, 100 * (t - prev) / dt))
            prev = t
        print("  %-28s%11.3fs%11s%7.1f%%" % ("END-TO-END", dt, "", 100.0))
    n = len([f for f in os.listdir(a.output_dir) if f.endswith(".npy")])
    if n < len(files):
        sys.exit("ERROR: expected %d outputs, found %d" % (len(files), n))


if __name__ == "__main__":
    main()
