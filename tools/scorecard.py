#!/usr/bin/env python3
"""The two-number scorecard. This is what picks the model we ship.

    python tools/scorecard.py --data data --ckpts runs/*/last.pt

The organisers stated the test set is half in-distribution and half
out-of-distribution "from different sources". So the quantity to maximise is
the MEAN of the two halves, not either alone -- and every model we have wins
one and loses the other:

    preal1   23.27 in-dist / 21.74 OOD-band -> 22.51
    long120  23.13 / 21.91 -> 22.52
    newgen   23.22 / 22.26 -> 22.74   <- incumbent

Columns:
  in_dist     PSNR on KLA's real held-out pairs             (the easy half)
  ood_band    mean PSNR over sigma 0.15/0.19/0.25, the range the real data
              actually occupies, measured on synthesised inputs
  collapse    PSNR(sigma 0.19) - PSNR(sigma 0.45). How hard it falls off the
              edge of what it was trained on. Lower is better.
  FINAL       mean(in_dist, ood_band) -- the selection number

This shells out to validate.py and noise_sweep.py rather than reimplementing
the metrics. Both are already tested, and a scorer with its own quiet bug would
be the worst possible thing to choose a model with.
"""
import argparse, csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY   = sys.executable
BAND = (0.15, 0.19, 0.25)          # where KLA's real data lives
EDGE = (0.19, 0.45)                # the collapse endpoints


def sh(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {' '.join(cmd[-6:])}\n{(r.stderr or '')[-400:]}")
    return r.returncode == 0


def last_row(tag):
    p = os.path.join(ROOT, "results.csv")
    if not os.path.isfile(p): return None
    hit = [r for r in csv.DictReader(open(p)) if r.get("tag") == tag]
    return hit[-1] if hit else None


def sweep(tag):
    p = os.path.join(ROOT, f"noise_sweep_{tag}.csv")
    if not os.path.isfile(p): return None
    return {round(float(r["sigma_mul"]), 3): r for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60, help="images per noise level")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    rows = []
    for ck in a.ckpts:
        tag = os.path.basename(os.path.dirname(ck)) or os.path.basename(ck)
        print(f"\n>>> {tag}", flush=True)
        if not (a.skip_existing and last_row(tag)):
            print("    in-distribution...", flush=True)
            sh([PY, "src/validate.py", "--data", a.data, "--ckpt", ck, "--tag", tag])
        if not (a.skip_existing and sweep(tag)):
            print("    noise sweep...", flush=True)
            sh([PY, "tools/noise_sweep.py", "--data", a.data, "--ckpt", ck,
                "--tag", tag, "--n", str(a.n)])

        r, s = last_row(tag), sweep(tag)
        if not r or not s:
            print("    no result, skipping"); continue
        band = [float(s[k]["model_psnr"]) for k in BAND if k in s]
        if not band:
            print("    sweep missing the band levels, skipping"); continue
        ind = float(r["psnr"])
        ood = sum(band) / len(band)
        coll = (float(s[EDGE[0]]["model_psnr"]) - float(s[EDGE[1]]["model_psnr"])
                if EDGE[0] in s and EDGE[1] in s else float("nan"))
        rows.append((tag, ind, float(r["ssim"]), float(r["lpips"]), ood, coll,
                     (ind + ood) / 2))

    if not rows:
        print("\nnothing scored."); return
    rows.sort(key=lambda x: -x[-1])
    print("\n" + "=" * 82)
    print(f"{'model':<16}{'in_dist':>9}{'SSIM':>8}{'LPIPS':>8}{'ood_band':>10}"
          f"{'collapse':>10}{'FINAL':>9}")
    print("-" * 82)
    for t, p, ss, lp, o, c, f in rows:
        print(f"{t:<16}{p:>9.4f}{ss:>8.4f}{lp:>8.4f}{o:>10.4f}{c:>10.2f}{f:>9.4f}")
    print("=" * 82)
    print(f"\nSHIP: {rows[0][0]}   (FINAL {rows[0][-1]:.4f})")
    if len(rows) > 1:
        d = rows[0][-1] - rows[1][-1]
        print(f"margin over {rows[1][0]}: {d:+.4f} dB"
              f"{'  -- inside the 0.010 dB noise floor, so pick the simpler one' if abs(d) < 0.010 else ''}")
    print("\nincumbent to beat: r2-newgen at FINAL 22.74")


if __name__ == "__main__":
    main()
