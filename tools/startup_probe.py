#!/usr/bin/env python3
"""Is a non-PyTorch runtime actually cheaper to START? Measure before building.

The entire "replace PyTorch with TensorRT" thesis rests on one number nobody has
measured: what the TRT stack costs to import and initialise, versus torch's
1.445 s. If the TRT stack costs about the same, the idea is dead and we have
spent five minutes instead of an afternoon.

Each candidate is timed in a FRESH SUBPROCESS, because import cost is a
once-per-process cost and measuring it in-process measures nothing.

    python tools/startup_probe.py --reps 5
"""
import argparse, json, statistics, subprocess, sys, time

# Each snippet prints: <import_seconds> <cuda_ctx_seconds> <note>
SNIPPETS = {
"numpy only": r'''
import time; t=time.perf_counter()
import numpy as np
print(f"{time.perf_counter()-t:.4f} 0.0 numpy-{np.__version__}")
''',

"torch (current baseline)": r'''
import time; t=time.perf_counter()
import torch, numpy
ti=time.perf_counter()-t
t2=time.perf_counter(); torch.zeros(1,device="cuda"); torch.cuda.synchronize()
print(f"{ti:.4f} {time.perf_counter()-t2:.4f} torch-{torch.__version__}")
''',

"tensorrt + cuda-python": r'''
import time; t=time.perf_counter()
import numpy, tensorrt as trt
from cuda import cudart
ti=time.perf_counter()-t
t2=time.perf_counter()
cudart.cudaFree(0)                      # forces CUDA context creation
print(f"{ti:.4f} {time.perf_counter()-t2:.4f} trt-{trt.__version__}")
''',

"tensorrt + pycuda": r'''
import time; t=time.perf_counter()
import numpy, tensorrt as trt
ti=time.perf_counter()-t
t2=time.perf_counter()
import pycuda.driver as cuda, pycuda.autoinit
print(f"{ti:.4f} {time.perf_counter()-t2:.4f} trt-{trt.__version__}+pycuda")
''',

"onnxruntime-gpu": r'''
import time; t=time.perf_counter()
import numpy, onnxruntime as ort
ti=time.perf_counter()-t
t2=time.perf_counter()
_=ort.get_available_providers()
print(f"{ti:.4f} {time.perf_counter()-t2:.4f} ort-{ort.__version__}")
''',

"cupy (for bicubic without torch)": r'''
import time; t=time.perf_counter()
import numpy, cupy as cp
ti=time.perf_counter()-t
t2=time.perf_counter(); cp.zeros(1); cp.cuda.Stream.null.synchronize()
print(f"{ti:.4f} {time.perf_counter()-t2:.4f} cupy-{cp.__version__}")
''',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    print(f"  fresh subprocess per measurement, {a.reps} interleaved reps\n")
    acc = {k: {"imp": [], "ctx": [], "note": ""} for k in SNIPPETS}
    for _ in range(a.reps):
        for name, code in SNIPPETS.items():          # interleaved
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True)
            if r.returncode != 0:
                acc[name]["note"] = "NOT AVAILABLE"
                continue
            p = r.stdout.split()
            acc[name]["imp"].append(float(p[0]))
            acc[name]["ctx"].append(float(p[1]))
            acc[name]["note"] = p[2] if len(p) > 2 else ""

    base = None
    print(f"  {'runtime':<34}{'import s':>10}{'cuda ctx s':>12}{'TOTAL s':>10}"
          f"{'vs torch':>10}  note")
    rows = []
    for name in SNIPPETS:
        d = acc[name]
        if not d["imp"]:
            print(f"  {name:<34}{'--':>10}{'--':>12}{'--':>10}{'--':>10}  {d['note']}")
            rows.append(dict(runtime=name, available=False)); continue
        i, c = statistics.median(d["imp"]), statistics.median(d["ctx"])
        tot = i + c
        if name.startswith("torch"): base = tot
        rows.append(dict(runtime=name, available=True, import_s=round(i, 4),
                         cuda_ctx_s=round(c, 4), total_s=round(tot, 4), note=d["note"]))
        vs = f"{tot-base:+.3f}" if base else ""
        print(f"  {name:<34}{i:>10.4f}{c:>12.4f}{tot:>10.4f}{vs:>10}  {d['note']}")

    if base:
        print(f"\n  PyTorch startup to beat: {base:.4f} s "
              f"(plus 0.283 s checkpoint load = 1.728 s of a 2.686 s run)")
        best = min((r for r in rows if r.get("available") and not r["runtime"].startswith("torch")
                    and not r["runtime"].startswith("numpy")),
                   key=lambda r: r["total_s"], default=None)
        if best:
            save = base - best["total_s"]
            print(f"  best non-torch stack: {best['runtime']} at {best['total_s']:.4f} s")
            print(f"  -> would save {save:+.3f} s = {100*save/2.686:+.1f}% of end-to-end")
            print(f"  -> VERDICT: {'WORTH BUILDING' if save > 0.4 else 'NOT WORTH IT'}"
                  f"  (threshold: must save >0.4 s to justify the integration risk)")
    json.dump(rows, open("startup_probe.json", "w"), indent=2)
    print("\n  wrote startup_probe.json")


if __name__ == "__main__":
    main()
