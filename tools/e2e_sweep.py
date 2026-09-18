#!/usr/bin/env python3
"""END-TO-END sweep of the shippable configurations. Fresh process every run.

Warm microbenchmarks mislead here: the measured budget on H100 is 53.8% import,
10.5% checkpoint load, 7.4% first-forward kernel init and only 4.7% actual
per-image compute. So a config is judged on fresh-process wall clock, not on
throughput of a resident tensor.

Runs are INTERLEAVED (one rep of every config, then the next round) so drift
cannot land on one config. Medians reported, with spread.

    python tools/e2e_sweep.py --data ../semicon_test_data/NoisyLR --reps 5
"""
import argparse, csv, itertools, os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FWD = re.compile(r"MODEL-ONLY INFERENCE:\s+([\d.]+) ms/image")
E2E = re.compile(r"^\s+END-TO-END\s+([\d.]+)", re.M)
IMP = re.compile(r"import torch \+ numpy\s+([\d.]+)")
CKT = re.compile(r"checkpoint load \+ model to device\s+([\d.]+)")

CONFIGS = [
    ("baseline b32",        [],                              {}),
    ("b64",                 ["--batch", "64"],               {}),
    ("b128",                ["--batch", "128"],              {}),
    ("b297 (single batch)", ["--batch", "297"],              {}),
    ("prewarm b32",         [],                              {"FORGEX_PREWARM": "1"}),
    ("half b32",            ["--half"],                      {}),
    ("prewarm+half b32",    ["--half"],                      {"FORGEX_PREWARM": "1"}),
    ("prewarm+half b297",   ["--half", "--batch", "297"],    {"FORGEX_PREWARM": "1",
                                                              "FORGEX_PREWARM_BATCH": "297"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="docs/e2e_sweep.csv")
    ap.add_argument("--workdir", default="/tmp/_e2e_sweep")
    a = ap.parse_args()

    acc = {n: {"e2e": [], "fwd": [], "imp": [], "ckt": []} for n, _, _ in CONFIGS}
    print(f"  {len(CONFIGS)} configs x {a.reps} interleaved reps, fresh process each\n")
    for rep in range(a.reps):
        for name, args, env in CONFIGS:
            shutil.rmtree(a.workdir, ignore_errors=True); os.makedirs(a.workdir, exist_ok=True)
            e = dict(os.environ, **env)
            r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"),
                                a.data, a.workdir, "--timing"] + args,
                               capture_output=True, text=True, env=e)
            if r.returncode != 0:
                print(f"  {name}: FAILED\n{r.stdout[-500:]}\n{r.stderr[-500:]}"); continue
            for key, pat in (("e2e", E2E), ("fwd", FWD), ("imp", IMP), ("ckt", CKT)):
                m = pat.search(r.stdout)
                if m: acc[name][key].append(float(m.group(1)))
        print(f"  round {rep+1}/{a.reps}", flush=True)
    shutil.rmtree(a.workdir, ignore_errors=True)

    def med(n, k):
        v = acc[n][k]; return statistics.median(v) if v else float("nan")
    def spr(n, k):
        v = acc[n][k]
        return (max(v)-min(v))/statistics.median(v)*100 if len(v) > 1 else 0.0

    base = med("baseline b32", "e2e")
    rows = []
    print(f"\n  {'config':<22}{'e2e s':>9}{'spread':>8}{'vs base':>9}"
          f"{'fwd ms/img':>12}{'import s':>10}{'ckpt s':>9}")
    for name, _, _ in CONFIGS:
        e, f = med(name, "e2e"), med(name, "fwd")
        rows.append(dict(config=name, e2e_s=round(e, 3), spread_pct=round(spr(name, "e2e"), 1),
                         delta_pct=round(100*(e-base)/base, 2),
                         fwd_ms_per_img=round(f, 4),
                         import_s=round(med(name, "imp"), 3),
                         ckpt_s=round(med(name, "ckt"), 3),
                         reps=len(acc[name]["e2e"])))
        print(f"  {name:<22}{e:>9.3f}{spr(name,'e2e'):>7.1f}%"
              f"{100*(e-base)/base:>+8.2f}%{f:>12.4f}"
              f"{med(name,'imp'):>10.3f}{med(name,'ckt'):>9.3f}")

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    with open(op, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    best = min(rows, key=lambda r: r["e2e_s"])
    print(f"\n  FASTEST: {best['config']}  {best['e2e_s']:.3f} s  ({best['delta_pct']:+.2f}%)")
    print(f"  wrote {op}")
    print(f"\n  NOTE: verify the winner's OUTPUT before adopting it -- speed without")
    print(f"  an identical output is not a win. Compare against the reference outputs.")


if __name__ == "__main__":
    main()
