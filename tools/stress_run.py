#!/usr/bin/env python3
"""Does run.py survive whatever the judges hand it? Twelve hostile cases.

    python tools/stress_run.py --run ../forgex-kla-ps01/run.py

A crash scores ZERO. Every experiment on this project is worth hundredths of a
decibel; this test is worth the whole submission, and it costs two minutes.

The contract every case must satisfy:
    * exit code 0
    * one output .npy per input .npy, SAME FILENAME
    * output is (2H, 2W) when the input is (H, W)
    * output dtype float32, all finite, all inside [0, 1]

The cases are not invented. Each one is something a benchmark harness plausibly
does, and several are things this repo has already been bitten by:

  square_128        the expected case -- if this fails nothing else matters
  single_image      a harness that calls us once per file, not once per folder
  size_64 / 256     the brief fixes 128x128, but nothing stops them resizing
  size_512          four times the memory of the expected case
  non_square        180x220 -- PixelShuffle and the bicubic skip must agree
  odd_dims          127x129 -- odd sizes are where x2 upsamplers go wrong
  trailing_axis     (H, W, 1) instead of (H, W); run.py is meant to preserve it
  float64           a harness that saved with the wrong dtype
  out_of_range      values below 0 and above 1 -- REAL KLA data does this
                    (measured range of the new set is -0.048 .. 1.62)
  nan_inf           a corrupt file must not poison the whole batch
  mixed_sizes       different shapes in one folder -- exercises the grouping path
  nested_dir        we are handed the DATASET dir holding GT/ and NoisyLR/

Failures are reported, not raised, so one bad case never hides the other eleven.
"""
import argparse, os, shutil, subprocess, sys, tempfile, time
import numpy as np

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def _img(h, w, seed=0):
    g = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 0.45 + 0.28 * np.sin(x / 7.0) * np.cos(y / 9.0)          # structure, not noise
    return np.clip(base + g.normal(0, 0.06, (h, w)), 0, 1).astype(np.float32)


@case
def square_128(d):   return {"a.npy": _img(128, 128)}
@case
def single_image(d): return {"only.npy": _img(128, 128, 1)}
@case
def size_64(d):      return {"a.npy": _img(64, 64, 2)}
@case
def size_256(d):     return {"a.npy": _img(256, 256, 3)}
@case
def size_512(d):     return {"a.npy": _img(512, 512, 4)}
@case
def non_square(d):   return {"a.npy": _img(180, 220, 5)}
@case
def odd_dims(d):     return {"a.npy": _img(127, 129, 6)}
@case
def trailing_axis(d): return {"a.npy": _img(128, 128, 7)[..., None]}
@case
def float64(d):      return {"a.npy": _img(128, 128, 8).astype(np.float64)}


@case
def out_of_range(d):
    a = _img(128, 128, 9) * 1.7 - 0.12               # the real LR range is -0.048 .. 1.62
    return {"a.npy": a.astype(np.float32)}


@case
def nan_inf(d):
    a, b = _img(128, 128, 10), _img(128, 128, 11)
    b[10, 10], b[20, 20] = np.nan, np.inf
    return {"good.npy": a, "bad.npy": b}


@case
def mixed_sizes(d):
    return {"s.npy": _img(64, 64, 12), "m.npy": _img(128, 128, 13), "l.npy": _img(192, 192, 14)}


@case
def nested_dir(d):
    os.makedirs(os.path.join(d, "GT"), exist_ok=True)
    np.save(os.path.join(d, "GT", "a.npy"), _img(256, 256, 15))
    sub = os.path.join(d, "NoisyLR")
    os.makedirs(sub, exist_ok=True)
    np.save(os.path.join(sub, "a.npy"), _img(128, 128, 16))
    return None                                       # files already written


def check(inp, out):
    """-> list of complaints. Empty means the contract held."""
    bad = []
    ins = {f: np.load(os.path.join(inp, f)) for f in os.listdir(inp) if f.endswith(".npy")}
    if not ins:                                       # nested case: look one level down
        for s in sorted(os.listdir(inp)):
            p = os.path.join(inp, s)
            if os.path.isdir(p) and s.lower() in ("noisylr", "lr", "input", "inputs"):
                ins = {f: np.load(os.path.join(p, f)) for f in os.listdir(p) if f.endswith(".npy")}
                break
    outs = sorted(f for f in os.listdir(out) if f.endswith(".npy"))
    if len(outs) != len(ins):
        bad.append(f"{len(ins)} in -> {len(outs)} out")
    for name, a in ins.items():
        if name not in outs:
            bad.append(f"missing {name}")
            continue
        y = np.load(os.path.join(out, name))
        ih, iw = a.shape[:2]
        yh, yw = y.shape[:2]
        if (yh, yw) != (ih * 2, iw * 2):
            bad.append(f"{name}: {a.shape} -> {y.shape}, expected {(ih*2, iw*2)}")
        if y.dtype != np.float32:
            bad.append(f"{name}: dtype {y.dtype}")
        if not np.isfinite(y).all():
            bad.append(f"{name}: {int((~np.isfinite(y)).sum())} non-finite values")
        elif y.min() < -1e-6 or y.max() > 1 + 1e-6:
            bad.append(f"{name}: range [{y.min():.3f}, {y.max():.3f}] outside [0,1]")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run.py", help="path to the submission entry point")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--extra", default="", help="extra flags passed through to run.py")
    a = ap.parse_args()

    run = os.path.abspath(a.run)
    if not os.path.isfile(run):
        sys.exit(f"no such file: {run}")
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(CASES)

    print(f"stress-testing {run}\n")
    print(f"{'case':<16}{'result':<9}{'secs':>7}   detail")
    print("-" * 78)
    failed = []
    for n in names:
        tmp = tempfile.mkdtemp(prefix=f"stress_{n}_")
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inp)
        files = CASES[n](inp)
        if files:
            for fn, arr in files.items():
                np.save(os.path.join(inp, fn), arr)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, run, inp, out] + a.extra.split(),
                               capture_output=True, text=True, timeout=a.timeout,
                               cwd=os.path.dirname(run))
            dt = time.perf_counter() - t0
            if p.returncode != 0:
                tail = (p.stderr or p.stdout or "").strip().splitlines()
                bad = ["exit %d: %s" % (p.returncode, tail[-1] if tail else "no output")]
            else:
                bad = check(inp, out) if os.path.isdir(out) else ["no output directory"]
        except subprocess.TimeoutExpired:
            dt, bad = a.timeout, [f"TIMEOUT after {a.timeout}s"]
        except Exception as e:
            dt, bad = time.perf_counter() - t0, [f"{type(e).__name__}: {e}"]

        ok = not bad
        print(f"{n:<16}{'PASS' if ok else 'FAIL':<9}{dt:>7.1f}   {'' if ok else bad[0]}")
        for extra in bad[1:4]:
            print(f"{'':<32}{extra}")
        if not ok:
            failed.append(n)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failed:
        print(f"{len(failed)}/{len(names)} FAILED: {', '.join(failed)}")
        print("\nA crash on the judges' machine scores zero. Fix these before anything else.")
        sys.exit(1)
    print(f"all {len(names)} cases passed -- run.py holds its contract under every input we")
    print("could think of. This is the result to keep; it protects the whole submission.")


if __name__ == "__main__":
    main()
