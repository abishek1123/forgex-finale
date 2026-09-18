#!/usr/bin/env python3
"""Live demo of the quality/latency knob. Runs the REAL entry point (run.py) at
several settings on the same folder and prints measured wall clock and quality
side by side, so nothing here is a simulation of the product.

    python tools/demo_knob.py --data ../semicon_test_data --settings 4,8,12,16

Prints one table, and writes demo_knob.csv for the figure.
"""
import argparse, csv, os, subprocess, sys, shutil, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score(out_dir, gt_dir, files):
    sys.path.insert(0, os.path.join(HERE, "src"))
    import torch
    from metrics import psnr, ssim
    ps, ss = [], []
    for f in files:
        o = torch.from_numpy(np.load(os.path.join(out_dir, f)))[None, None]
        g = torch.from_numpy(np.load(os.path.join(gt_dir, f)))[None, None]
        ps.append(psnr(o.clamp(0, 1), g)); ss.append(ssim(o.clamp(0, 1), g))
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with GT/ and NoisyLR/")
    ap.add_argument("--settings", default="4,8,12,16", help="depths to demo")
    ap.add_argument("--reps", type=int, default=5,
                   help="repetitions per setting. End-to-end wall clock on this "
                        "entry point has a run-to-run stdev of ~1.17 s (n=9, "
                        "docs/bench_results.txt) while the WHOLE knob span is only "
                        "~0.9 s of model compute, so a single rep cannot resolve "
                        "the settings. Medians over several reps can.")
    ap.add_argument("--workdir", default="outputs/_knob_demo")
    ap.add_argument("--keep", action="store_true", help="keep the restored .npy files")
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    in_dir = os.path.join(a.data, "NoisyLR")
    files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
    depths = [int(d) for d in a.settings.split(",")]

    print(f"\n  ForgeX quality/latency knob -- live demo")
    print(f"  {len(files)} images from {a.data}")
    print(f"  every row is a real `python run.py` invocation, timed end to end\n")

    import statistics
    # INTERLEAVED: one rep of every setting, then the next round. Grouping all reps
    # of one setting together lets a thermal or background-load drift land entirely
    # on one row and masquerade as a knob effect.
    walls = {d: [] for d in depths}
    qual = {}
    for rep in range(a.reps):
        for d in depths:
            out = os.path.join(HERE, a.workdir, f"d{d}")
            shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, os.path.join(HERE, "run.py"), in_dir, out, "--depth", str(d)]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            if r.returncode != 0:
                print(f"  depth {d}: FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}"); continue
            walls[d].append(wall)
            if d not in qual:
                qual[d] = score(out, gt_dir, files)
            if not a.keep:
                shutil.rmtree(out, ignore_errors=True)
        print(f"  round {rep+1}/{a.reps} done", flush=True)

    rows = []
    for d in depths:
        if not walls[d]:
            continue
        w = walls[d]
        p, s = qual[d]
        rows.append(dict(depth=d, wall_median_s=round(statistics.median(w), 3),
                         wall_min_s=round(min(w), 3), wall_max_s=round(max(w), 3),
                         reps=len(w),
                         stdev_s=round(statistics.stdev(w), 3) if len(w) > 1 else 0.0,
                         psnr=round(p, 4), ssim=round(s, 5)))

    if not rows:
        sys.exit("no settings completed")
    base = max(rows, key=lambda r: r["depth"])
    print(f"\n  {'setting':>9}{'median':>9}{'min':>8}{'max':>8}{'sd':>7}"
          f"{'speedup':>9}{'PSNR':>9}{'cost':>9}{'SSIM':>9}")
    for r in rows:
        print(f"  depth {r['depth']:>3}{r['wall_median_s']:>9.2f}{r['wall_min_s']:>8.2f}"
              f"{r['wall_max_s']:>8.2f}{r['stdev_s']:>7.2f}"
              f"{base['wall_median_s']/r['wall_median_s']:>8.2f}x{r['psnr']:>9.4f}"
              f"{r['psnr']-base['psnr']:>+9.3f}{r['ssim']:>9.5f}")

    # --- can this measurement resolve the knob at all? ---
    sd = max((r["stdev_s"] for r in rows), default=0.0)
    span = base["wall_median_s"] - min(r["wall_median_s"] for r in rows)
    n = min(r["reps"] for r in rows)
    se = sd / (n ** 0.5) if n else float("inf")
    print(f"\n  RESOLVABILITY")
    print(f"    observed span across settings      {span:.3f} s")
    print(f"    worst per-setting stdev            {sd:.3f} s   (n={n})")
    print(f"    standard error of each median      {se:.3f} s")
    if span > 4 * se:
        print(f"    -> span is {span/se:.1f}x the standard error: the settings ARE resolved.")
    else:
        print(f"    -> span is only {span/se:.1f}x the standard error: NOT RESOLVED.")
        need = int((sd / (span / (2 * 1.96))) ** 2) + 1 if span > 0 else 0
        print(f"       End-to-end wall clock cannot separate these settings at n={n}.")
        print(f"       Either raise --reps to about {need}, or quote the model-only")
        print(f"       numbers from models/knob_datasheet.json, which are measured")
        print(f"       directly with CUDA sync and have 0.2-1.1% spread.")
    with open(os.path.join(HERE, "demo_knob.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n  wrote demo_knob.csv")
    print(f"\n  NOTE: these are END-TO-END times, so they include ~{100*0.82:.0f}% fixed "
          f"cost (import torch,\n  CUDA init, disk I/O). The knob only scales the forward "
          f"pass, which is why the\n  speedup here is far smaller than the FLOP ratio. "
          f"That is the honest number.\n")


if __name__ == "__main__":
    main()
