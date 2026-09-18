#!/usr/bin/env python3
"""Tonight's three jobs, in dependency order, unattended.

    python tools\tonight.py --data <train-root>

1. r2-newgen   Does the REBUILT degradation generator beat training on real pairs
               alone?  120 epochs, p_real 0.3.                            ~2h
2. r2-ood      What does GENERALISATION cost?  Holds out ids 2868-3345 entirely
               -- a content block with structure 3x coarser than the rest -- so
               the model is scored on morphology it has never seen.        ~2h
3. capacity    How far can the model shrink before quality moves?  7 configs
               spanning 78x of parameter count, 40 epochs each.            ~3h

Steps 2 and 3 take their --p-real from step 1's verdict, so both measure the data
configuration we would actually ship. Every step skips work already complete, so
this is safe to re-run after an interruption.

Order is deliberate: if the night is cut short, steps 1 and 2 are the two results
that speak to the score. Step 3 speaks to latency and can wait.
"""
import argparse, csv, os, statistics as st, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY   = sys.executable
HOLD = "2868-3345"

def log(run):
    p = os.path.join(ROOT, "runs", run, "log.csv")
    if not os.path.isfile(p): return None
    r = list(csv.DictReader(open(p)))
    return r or None

def final5(run):
    r = log(run)
    if not r: return None
    return (st.mean(float(x["val_psnr"]) for x in r[-5:]),
            st.mean(float(x["val_ssim"]) for x in r[-5:]), len(r))

def sh(cmd, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)
    t = time.perf_counter()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"--- {label}: {(time.perf_counter()-t)/60:.0f} min, rc={rc} ---", flush=True)
    return rc

def train(out, epochs, extra):
    return [PY, os.path.join(ROOT,"src","train.py"), "--data", A.data,
            "--out", os.path.join("runs",out), "--epochs", str(epochs),
            "--seed","0", "--split-seed","0", "--amp"] + extra

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=120)
A = ap.parse_args()

# ======================= STEP 1 : the rebuilt generator =======================
if (f := final5("r2-newgen")) and f[2] >= A.epochs:
    print("step 1 (r2-newgen) already complete, skipping")
else:
    sh(train("r2-newgen", A.epochs, ["--p-real","0.3"]), "STEP 1  r2-newgen  (rebuilt generator, p_real 0.3)")

new, ref = final5("r2-newgen"), final5("r2-preal1")
print(f"\n{'run':<14}{'final-5 PSNR':>14}{'final-5 SSIM':>14}")
print(f"{'r2-preal1':<14}{ref[0]:>14.4f}{ref[1]:>14.4f}   100% real pairs")
print(f"{'r2-newgen':<14}{new[0]:>14.4f}{new[1]:>14.4f}   70% rebuilt synthetic")
d = new[0] - ref[0]
print(f"{'delta':<14}{d:>+14.4f}{new[1]-ref[1]:>+14.4f}   noise floor 0.010 dB")
if d > 0.010:
    P_REAL = "0.3"; print("\nVERDICT: the rebuilt generator EARNS ITS PLACE. Synthetic data works once")
    print("         the model of the data is right. Steps 2-3 use p_real 0.3.")
elif d < -0.010:
    P_REAL = "1.0"; print("\nVERDICT: real pairs win even against a correctly-fitted generator.")
    print("         A clean negative result. Steps 2-3 use p_real 1.0.")
else:
    P_REAL = "1.0"; print("\nVERDICT: inside the noise floor. Ship the simpler option (no synthetic).")
    print("         Steps 2-3 use p_real 1.0.")

# ======================= STEP 2 : the generalisation cost =====================
if (f := final5("r2-ood")) and f[2] >= A.epochs:
    print("\nstep 2 (r2-ood) already complete, skipping")
else:
    sh(train("r2-ood", A.epochs, ["--p-real",P_REAL,"--holdout-range",HOLD]),
       f"STEP 2  r2-ood  (holding out ids {HOLD}, p_real {P_REAL})")

# score BOTH models on the held-out block. preal1 trained on those images, so it
# is the in-distribution ceiling; r2-ood never saw them. The gap is what
# generalisation to unseen morphology actually costs.
for ck, tag in ((os.path.join("runs","r2-preal1","last.pt"), "preal1-on-block"),
                (os.path.join("runs","r2-ood","last.pt"),    "ood-on-block")):
    if os.path.isfile(os.path.join(ROOT, ck)):
        sh([PY, os.path.join(ROOT,"src","validate.py"), "--data", A.data,
            "--ckpt", ck, "--tag", tag, "--holdout-range", HOLD], f"scoring {tag}")

# ======================= STEP 3 : the efficiency frontier =====================
sh([PY, os.path.join(ROOT,"tools","capacity_sweep.py"), "--data", A.data, "--p-real", P_REAL],
   f"STEP 3  capacity sweep  (7 configs, p_real {P_REAL})")

print("\n" + "="*70)
print("ALL DONE.  results.csv has the two block scores; capacity_sweep.csv has the frontier.")
print("="*70)
