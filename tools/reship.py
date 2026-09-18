#!/usr/bin/env python3
"""Deploy a new checkpoint and re-measure EVERYTHING that depends on it.

    python tools/reship.py --ckpt <path-to-new-last.pt> \
        --data ../semicon_train_data/semicon_train_data \
        --test ../semicon_test_data \
        --d512 ../data512 \
        --nffa ../nffa

Backs up the current models/model.pt, installs the new one, then runs every
measurement whose value depends on which weights are shipped, plus the four
analysis tools whose output we need as committed evidence.

Each step is independent: a failure is reported and the run continues, so one
missing dataset does not cost you the other twenty minutes. A summary table is
printed at the end and written to docs_tmp/reship_summary.txt.
"""
import argparse, os, shutil, subprocess, sys, time, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
OUT = os.path.join(ROOT, "docs_tmp")
RESULTS = []


def sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def step(name, args, logfile, need=None):
    """Run one measurement. `need` is a list of paths that must exist."""
    for p in (need or []):
        if not os.path.exists(p):
            print(f"  SKIP  {name}  (missing {p})")
            RESULTS.append((name, "SKIPPED", f"missing {p}"))
            return
    print(f"\n=== {name}")
    t0 = time.time()
    lp = os.path.join(OUT, logfile)
    with open(lp, "w", encoding="utf-8") as f:
        r = subprocess.run([PY] + args, cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    tail = open(lp, encoding="utf-8", errors="replace").read().strip().splitlines()
    if r.returncode == 0:
        print("\n".join(tail[-6:]))
        print(f"  OK  {dt:.0f}s  ->  docs_tmp/{logfile}")
        RESULTS.append((name, "OK", f"docs_tmp/{logfile}"))
    else:
        print("\n".join(tail[-12:]))
        print(f"  FAILED rc={r.returncode}  ->  docs_tmp/{logfile}")
        RESULTS.append((name, f"FAILED rc={r.returncode}", f"docs_tmp/{logfile}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the new checkpoint to ship")
    ap.add_argument("--data", default="../semicon_train_data/semicon_train_data")
    ap.add_argument("--test", default="../semicon_test_data")
    ap.add_argument("--d512", default="../data512")
    ap.add_argument("--nffa", default="../nffa")
    ap.add_argument("--tag", default="ship")
    ap.add_argument("--no-deploy", action="store_true",
                    help="measure without replacing models/model.pt")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mp = os.path.join(ROOT, "models", "model.pt")
    new = os.path.abspath(os.path.join(ROOT, a.ckpt)) if not os.path.isabs(a.ckpt) else a.ckpt
    if not os.path.isfile(new):
        sys.exit(f"no such checkpoint: {new}")

    print("=" * 66)
    print(f"new checkpoint : {new}")
    print(f"               sha1 {sha(new)}   {os.path.getsize(new)/1e6:.2f} MB")
    if os.path.isfile(mp):
        print(f"current model.pt sha1 {sha(mp)}")

    if not a.no_deploy:
        if os.path.isfile(mp):
            bak = os.path.join(ROOT, "models", "model-PREVIOUS.pt")
            shutil.copy2(mp, bak)
            print(f"backed up      : models/model-PREVIOUS.pt")
        shutil.copy2(new, mp)
        assert sha(mp) == sha(new), "copy verification failed"
        print(f"DEPLOYED       : models/model.pt  sha1 {sha(mp)}")
    print("=" * 66)

    T, D, F, N = a.test, a.data, a.d512, a.nffa
    tg = a.tag

    # ---- the four numbers that depend on which weights we ship -------------
    step("in-distribution : organisers' 297-image test set",
         ["src/validate.py", "--data", T, "--ckpt", "models/model.pt",
          "--n-val", "1000000", "--baseline", "--tag", f"TESTSET-{tg}"],
         "reship_testset.txt", [os.path.join(T, "GT")])

    step("in-distribution : our 200-image held-out split",
         ["src/validate.py", "--data", D, "--ckpt", "models/model.pt",
          "--baseline", "--tag", f"SPLIT-{tg}"],
         "reship_split.txt", [os.path.join(D, "GT")])

    step("OOD : withheld content block 2868-3345",
         ["src/validate.py", "--data", D, "--ckpt", "models/model.pt",
          "--holdout-range", "2868-3345", "--baseline", "--tag", f"{tg}-on-block"],
         "reship_onblock.txt", [os.path.join(D, "GT")])

    step("OOD : 256 -> 512 with no training at that scale",
         ["src/validate.py", "--data", F, "--ckpt", "models/model.pt",
          "--n-val", "1000000", "--baseline", "--tag", f"{tg}-512"],
         "reship_512.txt", [os.path.join(F, "GT")])

    step("OOD : ten labelled NFFA morphologies",
         ["tools/per_category.py", "--src", N, "--ckpt", "models/model.pt", "--n", "24"],
         "reship_per_category.txt", [N])

    step("OOD : nine-level noise sweep",
         ["tools/noise_sweep.py", "--data", D, "--ckpt", "models/model.pt", "--tag", tg],
         "reship_noise_sweep.txt", [os.path.join(D, "GT")])

    step("OOD : nine-axis suite, scored on the ORGANISERS' test set",
         ["tools/ood_suite.py", "--data", T, "--ckpts", "models/model.pt"],
         "reship_ood_testset.txt", [os.path.join(T, "GT")])

    step("hallucination guard : energy above the input Nyquist",
         ["tools/hf_energy.py", "--data", T, "--ckpts", "models/model.pt"],
         "reship_hf_energy.txt", [os.path.join(T, "GT")])

    # ---- analysis tools we need a committed output file for ---------------
    step("degradation fit (R^2)", ["tools/calibrate.py", "--data", D],
         "reship_calibrate.txt", [os.path.join(D, "GT")])

    step("noise physics : speckle vs Poisson by BIC",
         ["tools/noise_physics.py", "--data", D],
         "reship_noise_physics.txt", [os.path.join(D, "GT")])

    # ---- submission readiness --------------------------------------------
    # Requirement 3: denoised outputs for the ORGANISERS' test set, all 297.
    step("packaging : outputs/ for the organisers' test set + clean-directory run",
         ["tools/package_check.py", "--data", T],
         "reship_package_check.txt", [os.path.join(T, "GT")])

    # ---- summary ----------------------------------------------------------
    w = max(len(r[0]) for r in RESULTS) + 2
    lines = ["", "=" * 78, "SUMMARY", "=" * 78]
    for n, st, where in RESULTS:
        lines.append(f"  {n:<{w}} {st:<16} {where}")
    ok = sum(1 for r in RESULTS if r[1] == "OK")
    lines += ["", f"  {ok} of {len(RESULTS)} steps OK",
              "", "  results.csv now holds the new rows. Send me:",
              "     docs_tmp/reship_summary.txt", "     results.csv",
              "     docs_tmp/per_category.csv", f"     noise_sweep_{tg}.csv", ""]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "reship_summary.txt"), "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
