#!/usr/bin/env python3
"""The knob x backend report: five TensorRT engines, the PyTorch fallback, one table.

    python tools/engine_report.py --test ../semicon_test_data \
        --depths 3,6,10,13,16 --reps 5 --out docs/engine_report

For every (depth, backend) cell it runs the REAL entry point -- run_fast.py, the
same command a judge would type -- as a subprocess and times it end to end, the
way the task is scored. Quality is measured once per cell from the outputs that
run produced, with src/metrics.py (the same PSNR/SSIM/LPIPS that picked the
shipped model), so quality and speed describe the same artefact.

Two things make the timing honest:

  INTERLEAVED. Reps go round-robin across cells, never all reps of one cell in a
  row, so clock and thermal drift hit every row equally instead of landing on
  whichever row happened to run last.

  RESOLVABILITY. It reports the knob's span against the run-to-run spread. A
  knob whose span is inside its own noise is not a knob, and the table says so
  instead of letting a single lucky run tell the story.
"""
import argparse, csv, json, os, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_inputs(test):
    for sub in ("NoisyLR", "noisylr", "LR", "inputs"):
        p = os.path.join(test, sub)
        if os.path.isdir(p):
            return p
    return test


def find_gt(test):
    for sub in ("GT", "gt", "HR"):
        p = os.path.join(test, sub)
        if os.path.isdir(p):
            return p
    return None


def mad_sigma(xs):
    """Robust 1-sigma: 1.4826 x median-absolute-deviation.

    The median is a robust estimator; pairing it with pstdev is a mistake. On a
    shared pod one run in five stalls ~0.5 s on I/O, and that single sample
    dominates the standard deviation of a cell whose other four agree to 0.02 s
    -- which then reports a 24x-resolvable knob as unresolvable. MAD ignores the
    stall the same way the median does."""
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return 1.4826 * statistics.median([abs(x - m) for x in xs])


def median_se(xs):
    """Standard error OF THE MEDIAN: 1.2533 x sigma / sqrt(n), sigma robust.

    A knob stop is a MEDIAN of several reps, so the question "are two stops
    distinguishable" compares two medians -- and the uncertainty on a median
    shrinks with sqrt(n). Comparing a difference of medians against a
    PER-SAMPLE sigma (and pooling with the worst sigma over every cell, at
    that) understates the separation by roughly sqrt(n), which is how a knob
    whose stops are 5-10 sigma apart got reported as 'publish fewer stops'."""
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.2533 * mad_sigma(xs) / (n ** 0.5)


def run_once(in_dir, out_dir, depth, backend, extra):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, os.path.join(HERE, "run_fast.py"), in_dir, out_dir]
    if depth:
        cmd += ["--depth", str(depth)]
    if backend == "torch":
        cmd += ["--no-trt"]
    cmd += extra
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("cell failed: depth=%s backend=%s" % (depth, backend))
    served = "TensorRT" if "[run_fast] TensorRT" in r.stdout else "PyTorch"
    return dt, served, r.stdout.strip().splitlines()


def score(out_dir, gt_dir, device):
    import numpy as np, torch
    sys.path.insert(0, os.path.join(HERE, "src"))
    from metrics import psnr, ssim, lpips                          # noqa: E402
    names = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
    P, S, L = [], [], []
    for i in range(0, len(names), 32):
        ch = names[i:i + 32]
        g = torch.from_numpy(np.stack([np.load(os.path.join(gt_dir, n)).astype("float32")
                                       for n in ch]).squeeze()[:, None]).to(device)
        o = torch.from_numpy(np.stack([np.load(os.path.join(out_dir, n)).astype("float32")
                                       for n in ch]).squeeze()[:, None]).to(device)
        for k in range(g.shape[0]):
            P.append(psnr(o[k:k+1], g[k:k+1]))
            S.append(ssim(o[k:k+1], g[k:k+1]))
            # PER IMAGE, like PSNR and SSIM above. Appending one value per BATCH
            # and then taking an unweighted mean over batches is wrong whenever
            # the last batch is short: 297 images at batch 32 gives nine batches
            # of 32 and one of 9, and weighting that final batch 1/10 instead of
            # 9/297 moved LPIPS from 0.18126 to 0.17935. Measured, not theorised.
            L.append(lpips(o[k:k+1], g[k:k+1], device=device))
    return (sum(P) / len(P), sum(S) / len(S),
            sum(L) / len(L) if L else float("nan"), len(P))


def _report(rows, backends, depths, a):
        print("\n  timing from %d of %d reps per cell (first %d discarded as warmup)"
              % (rows[0]["reps_used"], rows[0]["reps_total"], a.warmup))
        print("  %-7s%-9s%-10s%10s%9s%9s%9s%9s%10s%9s%10s"
              % ("depth", "backend", "served", "t_med(s)", "t_mad", "t_std", "t_cold",
                 "img/s", "PSNR", "SSIM", "LPIPS"))
        for r in rows:
            print("  %-7d%-9s%-10s%10.3f%9.3f%9.3f%9.3f%9.1f%10.4f%9.5f%10.5f"
                  % (r["depth"], r["backend"], r["served"], r["t_median"], r["t_mad"],
                     r["t_std"], r["t_cold"], r["img_per_s"], r["psnr"], r["ssim"],
                     r["lpips"]))
        print("  t_mad = robust 1-sigma (1.4826 x MAD). t_std is kept for transparency;")
        print("  where the two disagree, one rep in the cell stalled on I/O.")

        if len(backends) == 2:
            print("\n  %-7s%12s%12s%10s%12s" % ("depth", backends[0], backends[1],
                                                "speedup", "dPSNR"))
            for d in depths:
                x = next(r for r in rows if r["depth"] == d and r["backend"] == backends[0])
                y = next(r for r in rows if r["depth"] == d and r["backend"] == backends[1])
                print("  %-7d%12.3f%12.3f%9.2fx%12.4f"
                      % (d, x["t_median"], y["t_median"],
                         y["t_median"] / x["t_median"], x["psnr"] - y["psnr"]))

        for b in backends:
            rs = sorted((r for r in rows if r["backend"] == b), key=lambda r: r["depth"])
            if len(rs) < 2:
                continue
            span = rs[-1]["t_median"] - rs[0]["t_median"]
            noise = max(r["t_mad"] for r in rs)
            print("\n  %s: knob span %.3f s, worst robust sigma %.3f s -> %s"
                  % (b, span, noise,
                     "RESOLVABLE (%.0fx)" % (span / noise) if noise > 0 and span > 3 * noise
                     else "NOT resolvable"))
            print("     adjacent stops (difference of medians vs pooled SE of the medians):")
            worst = None
            for i in range(len(rs) - 1):
                lo, hi = rs[i], rs[i + 1]
                dt = hi["t_median"] - lo["t_median"]
                se = (lo["t_se"] ** 2 + hi["t_se"] ** 2) ** 0.5
                k = dt / se if se > 0 else float("inf")
                worst = k if worst is None else min(worst, k)
                print("       d%-3d -> d%-3d  %+7.3f s  +/- %.3f  = %6.1f sigma  %s"
                      % (lo["depth"], hi["depth"], dt, se, k,
                         "OK" if k >= 3 else ("INVERTED" if dt < 0 else "too close")))
            print("     -> %s" % ("every published stop is distinguishable (worst %.1f sigma)" % worst
                                  if worst is not None and worst >= 3
                                  else "some stops are not separable; drop or re-space them"))
        fastest = min(rows, key=lambda r: r["t_median"])
        slowest = max(rows, key=lambda r: r["t_median"])
        print("  end-to-end range %.3f s (%s d%d) .. %.3f s (%s d%d)  = %.2fx"
              % (fastest["t_median"], fastest["backend"], fastest["depth"],
                 slowest["t_median"], slowest["backend"], slowest["depth"],
                 slowest["t_median"] / fastest["t_median"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--depths", default="3,6,10,13,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1,
                    help="reps discarded from the TIMING statistics. The first "
                         "pass over a cell deserialises a cold plan, faults the "
                         ".npy files into page cache and builds the kernel cache; "
                         "including it inflates the spread by ~20x and makes a "
                         "real knob look unresolvable. Outputs are unaffected.")
    ap.add_argument("--backends", default="trt,torch")
    ap.add_argument("--out", default=os.path.join(HERE, "docs", "engine_report"))
    ap.add_argument("--extra", default="", help="extra flags passed to run_fast.py")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--from-json", default="",
                    help="recompute the tables and verdicts from a previous run's "
                         ".json (it stores every rep time). Costs no GPU time.")
    a = ap.parse_args()

    if a.from_json:
        rows = json.load(open(a.from_json))
        for r in rows:
            t = sorted(r["t_all"][a.warmup:]) or sorted(r["t_all"])
            r["t_median"] = round(statistics.median(t), 4)
            r["t_mad"] = round(mad_sigma(t), 4)
            r["t_std"] = round(statistics.pstdev(t), 4) if len(t) > 1 else 0.0
            r["t_se"] = round(median_se(t), 4)
            r["reps_used"] = len(t)
            r["img_per_s"] = round(max(r["imgs"], 1) / r["t_median"], 2)
        backends, depths = [], []
        for r in rows:
            if r["backend"] not in backends: backends.append(r["backend"])
            if r["depth"] not in depths: depths.append(r["depth"])
        _report(rows, backends, sorted(depths), a)
        return

    in_dir = find_inputs(a.test); gt_dir = find_gt(a.test)
    extra = a.extra.split() if a.extra else []
    depths = [int(x) for x in a.depths.split(",")]
    backends = a.backends.split(",")
    full = max(depths)
    cells = [(d, b) for d in depths for b in backends]
    tmp = os.path.join(HERE, "_report_out")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    print("input  %s" % in_dir)
    print("GT     %s" % gt_dir)
    print("cells  %d  x  %d reps, interleaved\n" % (len(cells), a.reps))

    times = {c: [] for c in cells}
    served = {}
    kept = {}
    for rep in range(a.reps):
        for c in cells:
            d, b = c
            out = tmp + "_%s_d%d" % (b, d)
            dt, srv, log = run_once(in_dir, out, 0 if d == full else d, b, extra)
            times[c].append(dt)
            served[c] = srv
            if rep == 0:
                kept[c] = out
            print("  rep %d  depth %-3d %-6s %6.3f s   (%s)" % (rep + 1, d, b, dt, srv),
                  flush=True)
        print()

    rows = []
    for c in cells:
        d, b = c
        used = times[c][a.warmup:] if len(times[c]) > a.warmup else times[c]
        t = sorted(used)
        med = statistics.median(t)
        q = dict(psnr=float("nan"), ssim=float("nan"), lpips=float("nan"), n=0)
        if not a.no_score and gt_dir:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            p, s, l, n = score(kept[c], gt_dir, dev)
            q = dict(psnr=p, ssim=s, lpips=l, n=n)
        rows.append(dict(depth=d, backend=b, served=served[c],
                         t_median=round(med, 4), t_min=round(t[0], 4),
                         t_max=round(t[-1], 4),
                         t_std=round(statistics.pstdev(t), 4) if len(t) > 1 else 0.0,
                         t_mad=round(mad_sigma(t), 4), t_se=round(median_se(t), 4), t_all=[round(v, 4) for v in times[c]],
                         t_cold=round(times[c][0], 4), reps_used=len(t),
                         reps_total=len(times[c]),
                         imgs=q["n"],
                         ms_per_img=round(med / max(q["n"], 1) * 1000, 4),
                         img_per_s=round(max(q["n"], 1) / med, 2),
                         psnr=round(q["psnr"], 4), ssim=round(q["ssim"], 5),
                         lpips=round(q["lpips"], 5)))

    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(a.out + ".json", "w"), indent=2)

    _report(rows, backends, depths, a)


    print("\n  wrote %s.csv / .json" % a.out)


if __name__ == "__main__":
    main()
