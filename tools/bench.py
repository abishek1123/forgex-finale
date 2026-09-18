#!/usr/bin/env python3
"""Benchmark two entry-point scripts end-to-end and write the verdict to a file.

    python tools/bench.py --a run.py --b <challenger>.py --input testin

Times the WHOLE PROCESS (which is what KLA measures), alternating between the
two scripts so GPU clock and thermal drift hit both equally -- a 1.9x spread
from thermal state alone bit us in round 1. Then compares the outputs.

Writes bench_results.txt next to the repo root.
"""
import argparse, os, subprocess, sys, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY   = sys.executable

_ap = argparse.ArgumentParser()
_ap.add_argument("--a", default="run.py", help="baseline script")
_ap.add_argument("--b", required=True, help="challenger script to compare against --a")
_ap.add_argument("--input", default="testin", help="directory of .npy inputs")
_ap.add_argument("--reps", type=int, default=9)
_A = _ap.parse_args()
REPS = _A.reps
IN   = _A.input if os.path.isabs(_A.input) else os.path.join(ROOT, _A.input)

def wall(script, outdir):
    t = time.perf_counter()
    r = subprocess.run([PY, os.path.join(ROOT, script), IN, outdir],
                       capture_output=True, text=True, cwd=ROOT)
    dt = time.perf_counter() - t
    return dt, r.returncode, (r.stdout or "").strip().splitlines()[-2:] , (r.stderr or "")[-300:]

def load(d):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(ROOT, d, "*.npy")))]

lines = []
def say(s=""):
    print(s); lines.append(s)

if not os.path.isdir(IN):
    sys.exit(f"missing {IN} -- create it first")
say(f"input: {len(glob.glob(os.path.join(IN,'*.npy')))} files   reps: {REPS}   python: {PY}")
say()

times = {_A.a: [], _A.b: []}
errs  = {}
for i in range(REPS):
    for s, o in ((_A.a, "bench_old"), (_A.b, "bench_new")):
        dt, rc, tail, err = wall(s, o)
        times[s].append(dt)
        if rc != 0:
            errs[s] = err
        say(f"  rep {i+1}  {s:<13} {dt:7.2f}s   rc={rc}   {tail[-1] if tail else ''}")
say()

for s in times:
    v = times[s]
    say(f"{s:<13} best {min(v):6.2f}s   median {sorted(v)[len(v)//2]:6.2f}s   worst {max(v):6.2f}s")
for s, e in errs.items():
    say(f"\n!! {s} exited non-zero:\n{e}")

a, b = load("bench_old"), load("bench_new")
say()
if len(a) != len(b) or not a:
    say(f"!! output count mismatch: {len(a)} vs {len(b)}")
else:
    md = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
    mse = np.mean([float(np.mean((x - y) ** 2)) for x, y in zip(a, b)])
    p = 10 * np.log10(1 / max(mse, 1e-20))
    say(f"outputs: {len(a)} files   max abs diff {md:.3e}   PSNR between them {p:.1f} dB")
    say(f"shapes:  {sorted({x.shape for x in b})}")
    say(f"range:   [{min(float(x.min()) for x in b):.4f}, {max(float(x.max()) for x in b):.4f}]")

# --- does the fp16-level output difference move a SCORED metric? ---
GT = os.path.join(os.path.dirname(ROOT), "semicon_train_data", "semicon_train_data", "GT")
if os.path.isdir(GT):
    def score(d):
        ps = []
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.npy"))):
            g = os.path.join(GT, os.path.basename(f))
            if not os.path.isfile(g):
                continue
            x, y = np.load(f), np.load(g)
            if x.shape != y.shape:
                continue
            ps.append(10 * np.log10(1 / max(float(np.mean((x - y) ** 2)), 1e-20)))
        return float(np.mean(ps)), len(ps)
    po, no = score("bench_old"); pn, nn = score("bench_new")
    say()
    say(f"PSNR vs ground truth   {_A.a} {po:.6f}   {_A.b} {pn:.6f}   "
        f"difference {abs(po-pn):.6f} dB  (n={no})")
    say(f"  -> the numeric difference is {'IRRELEVANT to the score' if abs(po-pn) < 0.001 else 'MOVING THE SCORE - investigate'}")

import statistics as st
fo, fn = times[_A.a], times[_A.b]
say()
say(f"{'':<13}{'median':>9}{'mean':>9}{'stdev':>9}{'best':>9}{'worst':>9}")
for k, v in ((_A.a, fo), (_A.b, fn)):
    say(f"{k:<13}{st.median(v):>9.2f}{st.mean(v):>9.2f}{st.stdev(v):>9.2f}{min(v):>9.2f}{max(v):>9.2f}")
d = st.median(fo) - st.median(fn)
pooled = (st.stdev(fo) + st.stdev(fn)) / 2
say()
say(f"median difference {d:+.2f}s against a typical run-to-run spread of {pooled:.2f}s")
if abs(d) < pooled:
    say(f"  -> INSIDE THE NOISE. Not a measurable difference; keep {_A.a} (the verified artifact).")
else:
    say(f"  -> {_A.b + ' is genuinely faster - adopt it' if d > 0 else _A.a + ' is genuinely faster - keep it'}")

fast_best, old_best = min(times[_A.b]), min(times[_A.a])
say()
say(f"VERDICT: {_A.b} is {old_best/fast_best:.2f}x the speed of {_A.a} "
    f"({'FASTER - merge it' if fast_best < old_best*0.97 else 'NOT faster - keep run.py'})")

out = os.path.join(ROOT, "bench_results.txt")
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"\nwritten to {out}")
