#!/usr/bin/env python3
"""Measured forward time vs knob depth, ON THE SHIPPING PATH.

models/knob_datasheet.json is a microbenchmark: synthetic input, cudnn.benchmark
forced ON. run.py leaves cudnn.benchmark at its default, so the datasheet is a
CEILING, not the number the submission actually achieves. This sweep runs the
real entry point at each depth and records what it really does.

Repetitions are INTERLEAVED (one rep of every depth, then the next round) so
thermal drift cannot land on one depth and masquerade as a knob effect. The
forward time itself was measured to vary ~19% run-to-run, so a single rep per
depth is not enough.

    python tools/timing_sweep.py --data ../semicon_test_data/NoisyLR \
        --depths 3,8,13,16 --reps 5

Writes docs/timing_sweep.csv and prints a table with medians and spread.
"""
import argparse, csv, os, re, shutil, statistics, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = {
    "fwd_ms":  re.compile(r"MODEL-ONLY INFERENCE:\s+([\d.]+) ms/image"),
    "fwd_pct": re.compile(r"=\s+([\d.]+)% of end-to-end"),
    "e2e_s":   re.compile(r"^\s+END-TO-END\s+([\d.]+)", re.M),
    "imp_s":   re.compile(r"import torch \+ numpy\s+([\d.]+)"),
    "h2d_ms":  re.compile(r"host -> device transfer\s+[\d.]+\s+([\d.]+)"),
    "d2h_ms":  re.compile(r"device -> host \+ postprocess\s+[\d.]+\s+([\d.]+)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of degraded .npy inputs")
    ap.add_argument("--depths", default="3,8,13,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workdir", default="outputs/_timing_sweep")
    ap.add_argument("--out", default="docs/timing_sweep.csv")
    a = ap.parse_args()

    depths = [int(d) for d in a.depths.split(",")]
    acc = {d: {k: [] for k in PAT} for d in depths}
    out_dir = os.path.join(HERE, a.workdir)

    print(f"  sweep: depths {depths} x {a.reps} interleaved reps, real run.py\n")
    for rep in range(a.reps):
        for d in depths:
            shutil.rmtree(out_dir, ignore_errors=True); os.makedirs(out_dir, exist_ok=True)
            cmd = [sys.executable, os.path.join(HERE, "run.py"), a.data, out_dir,
                   "--timing", "--depth", str(d)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  depth {d}: FAILED\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
                continue
            for k, pat in PAT.items():
                m = pat.search(r.stdout)
                if m:
                    acc[d][k].append(float(m.group(1)))
        print(f"  round {rep+1}/{a.reps}", flush=True)
    shutil.rmtree(out_dir, ignore_errors=True)

    def med(d, k):
        v = acc[d][k]
        return statistics.median(v) if v else float("nan")

    def spread(d, k):
        v = acc[d][k]
        return (max(v) - min(v)) / statistics.median(v) * 100 if len(v) > 1 else 0.0

    sheet = {}
    sp = os.path.join(HERE, "models", "knob_datasheet.json")
    if os.path.isfile(sp):
        import json
        sheet = {r["depth"]: r for r in json.load(open(sp))["rows"]}

    rows = []
    print(f"\n  {'depth':>6}{'fwd ms/img':>12}{'spread':>8}{'img/s':>9}"
          f"{'fwd s':>8}{'e2e s':>8}{'fwd %':>8}{'import s':>10}"
          f"{'sheet ms':>10}{'ratio':>8}")
    for d in depths:
        f = med(d, "fwd_ms"); e = med(d, "e2e_s")
        ds = sheet.get(d, {}).get("ms_per_img")
        n_img = len([x for x in os.listdir(a.data) if x.lower().endswith(".npy")])
        row = dict(depth=d, reps=len(acc[d]["fwd_ms"]),
                   fwd_ms_per_img=round(f, 4), fwd_spread_pct=round(spread(d, "fwd_ms"), 1),
                   fwd_img_per_s=round(1000 / f, 1) if f == f else None,
                   fwd_total_s=round(f * n_img / 1000, 3),
                   e2e_s=round(e, 3), e2e_spread_pct=round(spread(d, "e2e_s"), 1),
                   fwd_pct_of_e2e=round(med(d, "fwd_pct"), 1),
                   import_s=round(med(d, "imp_s"), 3),
                   h2d_ms=round(med(d, "h2d_ms"), 4), d2h_ms=round(med(d, "d2h_ms"), 4),
                   datasheet_ms=ds, datasheet_ratio=round(f / ds, 3) if ds else None)
        rows.append(row)
        print(f"  {d:>6}{f:>12.4f}{spread(d,'fwd_ms'):>7.1f}%{1000/f:>9.1f}"
              f"{row['fwd_total_s']:>8.3f}{e:>8.3f}{row['fwd_pct_of_e2e']:>7.1f}%"
              f"{row['import_s']:>10.3f}"
              f"{ds if ds else float('nan'):>10.4f}"
              f"{row['datasheet_ratio'] if ds else float('nan'):>7.2f}x")

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lo, hi = rows[0], rows[-1]
    print(f"\n  knob span in FORWARD time : {hi['fwd_total_s'] - lo['fwd_total_s']:.3f} s "
          f"({hi['fwd_ms_per_img'] / lo['fwd_ms_per_img']:.2f}x)")
    print(f"  knob span in END-TO-END    : {hi['e2e_s'] - lo['e2e_s']:.3f} s "
          f"({100 * (hi['e2e_s'] - lo['e2e_s']) / hi['e2e_s']:.1f}% of the full-depth run)")
    print(f"\n  wrote {op}")


if __name__ == "__main__":
    main()
