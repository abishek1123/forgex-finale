#!/usr/bin/env python3
"""Does splitting the folder across GPUs actually help? Measured, not modelled.

Amdahl on the measured H100 budget says 71.7% of a 297-image run is serial
(import 53.8%, checkpoint load 10.5%, first-forward kernel init 7.4%), so the
hard ceiling for ANY parallelisation at 297 images is 13.1%. At 1,197 images the
fixed cost amortises and 2 GPUs are worth ~21%.

This runs the REAL entry point, N processes in parallel, each pinned to one GPU
with a disjoint slice of the input, and compares against the single-process run.
Outputs are then checked to be byte-identical to the single-GPU result -- a
throughput win that changes the output is not a win.

    python tools/multigpu_test.py --data ../semicon_test_data/NoisyLR --reps 3
"""
import argparse, os, shutil, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_split(files, data, nproc, workroot, extra_env=None):
    """Launch nproc processes, each with its own input dir and GPU. -> wall seconds."""
    shutil.rmtree(workroot, ignore_errors=True)
    parts = [files[i::nproc] for i in range(nproc)]
    ins, outs = [], []
    for i, part in enumerate(parts):
        d = os.path.join(workroot, f"in{i}"); o = os.path.join(workroot, f"out{i}")
        os.makedirs(d, exist_ok=True); os.makedirs(o, exist_ok=True)
        for f in part:
            os.link(os.path.join(data, f), os.path.join(d, f))   # hardlink: no copy cost
        ins.append(d); outs.append(o)
    procs = []
    t0 = time.perf_counter()
    for i in range(nproc):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), **(extra_env or {}))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "run.py"), ins[i], outs[i]],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env))
    errs = [p.communicate()[1] for p in procs]
    wall = time.perf_counter() - t0
    for i, p in enumerate(procs):
        if p.returncode != 0:
            print(f"  proc {i} FAILED:\n{errs[i][-400:].decode(errors='ignore')}")
            return None, outs
    return wall, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--procs", default="1,2")
    ap.add_argument("--workroot", default="/tmp/_mgpu")
    a = ap.parse_args()

    import torch
    ngpu = torch.cuda.device_count()
    files = sorted(f for f in os.listdir(a.data) if f.endswith(".npy"))
    print(f"  {len(files)} images   {ngpu} GPU(s) visible   {a.reps} interleaved reps\n")

    procs = [int(x) for x in a.procs.split(",") if int(x) <= max(ngpu, 1)]
    acc = {n: [] for n in procs}
    ref_out = None
    for rep in range(a.reps):
        for n in procs:                                   # interleaved
            wall, outs = run_split(files, a.data, n, a.workroot)
            if wall is None: sys.exit("a worker failed")
            acc[n].append(wall)
            if n == 1 and ref_out is None:
                ref_out = {}
                for f in os.listdir(outs[0]):
                    ref_out[f] = np.load(os.path.join(outs[0], f))
            elif ref_out is not None:
                mx = 0.0
                for o in outs:
                    for f in os.listdir(o):
                        mx = max(mx, float(np.abs(np.load(os.path.join(o, f)) - ref_out[f]).max()))
                if rep == 0:
                    print(f"  {n} proc: output max|diff| vs 1 proc = {mx:.3e}"
                          f"  {'IDENTICAL' if mx == 0.0 else '!! DIFFERS'}")
        print(f"  round {rep+1}/{a.reps}", flush=True)
    shutil.rmtree(a.workroot, ignore_errors=True)

    base = statistics.median(acc[procs[0]])
    print(f"\n  {'procs':>6}{'wall s':>10}{'spread':>8}{'speedup':>9}{'img/s':>10}")
    for n in procs:
        m = statistics.median(acc[n])
        sp = (max(acc[n]) - min(acc[n])) / m * 100 if len(acc[n]) > 1 else 0.0
        print(f"  {n:>6}{m:>10.3f}{sp:>7.1f}%{base/m:>8.2f}x{len(files)/m:>10.1f}")
    if len(procs) > 1:
        g = 100 * (base - statistics.median(acc[procs[-1]])) / base
        print(f"\n  {procs[-1]} GPUs give {g:+.1f}% on {len(files)} images.")
        print(f"  Amdahl predicted +7.7% at 297 and +21.2% at 1197 -- the gap between")
        print(f"  those two is why the evaluation's image count decides this.")


if __name__ == "__main__":
    main()
