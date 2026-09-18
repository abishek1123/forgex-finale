#!/usr/bin/env python3
"""The two measurements the ladder does NOT cover.

A) PRIORITY 5 cold cost of cudnn.benchmark. The ladder measures WARM latency, but
   algorithm search runs at startup and a scored run may be a single fresh
   process. Measured as fresh subprocesses, benchmark off vs on, interleaved.

B) PRIORITY 6 host<->device transfer. Tests pinned memory and non_blocking=True
   against the current synchronous .to(device) / .cpu(). Only adopted if it
   moves the number.
"""
import argparse, os, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import os, sys, time
t0 = time.perf_counter()
import torch, numpy as np
sys.path.insert(0, %r)
import run as R
torch.backends.cudnn.benchmark = (os.environ.get("BENCH") == "1")
ck = R.load_checkpoint(R.find_weights(None))
m = R.Restorer(**(ck.get("config", {}) or {})).eval().cuda().to(memory_format=torch.channels_last)
x = torch.rand(%d, 1, 128, 128, device="cuda").contiguous(memory_format=torch.channels_last)
t_setup = time.perf_counter() - t0
torch.cuda.synchronize(); t1 = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", enabled=True):
    m(x)                       # FIRST call -- this is where autotune is paid
torch.cuda.synchronize(); t_first = time.perf_counter() - t1
t2 = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", enabled=True):
    for _ in range(20): m(x)
torch.cuda.synchronize(); t_warm = (time.perf_counter() - t2) / 20
print(f"{t_setup:.4f} {t_first:.4f} {t_warm:.6f}")
'''


def cold(batch, reps):
    out = {}
    for b in ("0", "1"):
        out[b] = []
    for _ in range(reps):
        for b in ("0", "1"):                      # interleaved
            env = dict(os.environ, BENCH=b)
            r = subprocess.run([sys.executable, "-c", CHILD % (HERE, batch)],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0:
                print(r.stderr[-500:]); sys.exit("child failed")
            out[b].append([float(v) for v in r.stdout.split()])
    print(f"\n  PRIORITY 5 -- cudnn.benchmark cold cost (batch {batch}, "
          f"{reps} interleaved fresh processes)")
    print(f"  {'benchmark':<12}{'setup s':>10}{'FIRST call s':>14}{'warm ms/img':>14}")
    for b, lbl in (("0", "OFF"), ("1", "ON")):
        v = np.array(out[b])
        print(f"  {lbl:<12}{statistics.median(v[:,0]):>10.4f}"
              f"{statistics.median(v[:,1]):>14.4f}"
              f"{statistics.median(v[:,2])*1000/batch:>14.4f}")
    a0, a1 = np.array(out["0"]), np.array(out["1"])
    d_first = statistics.median(a1[:, 1]) - statistics.median(a0[:, 1])
    d_warm = (statistics.median(a0[:, 2]) - statistics.median(a1[:, 2])) * 1000 / batch
    print(f"\n  autotune costs {d_first:+.4f} s on the first call")
    print(f"  and saves      {d_warm:+.4f} ms/image warm")
    if d_warm > 0:
        print(f"  break-even at  {d_first/(d_warm/1000):.0f} images "
              f"-- worth it above that, a regression below it")
    else:
        print("  warm is not faster: reject cudnn.benchmark")


def io(batch, reps):
    import torch
    print(f"\n  PRIORITY 6 -- host<->device transfer (batch {batch})")
    a = np.random.rand(batch, 1, 128, 128).astype(np.float32)
    res = {}
    for name in ("plain", "pinned+nonblocking"):
        ts = []
        for _ in range(reps):
            t = torch.from_numpy(a)
            if name.startswith("pinned"):
                t = t.pin_memory()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            g = t.to("cuda", non_blocking=name.startswith("pinned"))
            back = g.cpu()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            del g, back
        res[name] = statistics.median(ts)
        print(f"  {name:<22}{res[name]*1000:>9.3f} ms  "
              f"({res[name]/batch*1000:.4f} ms/image round trip)")
    d = (res["plain"] - res["pinned+nonblocking"]) / batch * 1000
    print(f"\n  pinned+non_blocking saves {d:+.4f} ms/image on the round trip")
    print(f"  (for 297 images that is {d*297/1000:+.3f} s of a ~4 s run)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", default="", choices=["", "cold", "io"])
    x = ap.parse_args()
    if x.only in ("", "cold"): cold(x.batch, x.reps)
    if x.only in ("", "io"):   io(x.batch, x.reps)
