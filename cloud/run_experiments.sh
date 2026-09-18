#!/usr/bin/env bash
# The one-day experiment queue. Sequential on one GPU; split across pods if you
# have more than one.
#
#   bash cloud/run_experiments.sh [epochs]
#
# Every run: same architecture, same seed, same validation split. One variable.
# Skips anything already complete, so it is safe to re-run after an interruption.
set -uo pipefail
E="${1:-40}"
D="${DATA:-data}"
COMMON="--data $D --epochs $E --seed 0 --split-seed 0 --amp"

run () {                       # run <name> <extra args...>
  local name="$1"; shift
  if [ -f "runs/$name/log.csv" ] && [ "$(tail -n +2 "runs/$name/log.csv" | wc -l)" -ge "$E" ]; then
    echo ">>> $name already complete, skipping"; return
  fi
  echo -e "\n=============================================================="
  echo ">>> $name   $*"
  echo "=============================================================="
  python src/train.py $COMMON --out "runs/$name" "$@"
}

# --- the p_real / variety grid -------------------------------------------
# newgen (p_real 0.3, calibrated only) is the incumbent at a 50/50 score of
# 22.74. Each of these is a different bet on how far the test set strays.
run pr30-w50  --p-real 0.3 --wide-p 0.5     # newgen + long120's variety
run pr50-w50  --p-real 0.5 --wide-p 0.5     # more real anchoring
run pr30-w100 --p-real 0.3 --wide-p 1.0     # all-wide synthetic
run pr50-w00  --p-real 0.5 --wide-p 0.0     # calibrated only, more real

# --- loss variants --------------------------------------------------------
# w-lpips has been 0.0 all project: a metric possibly worth a third of the
# score with zero weight on it.
run loss-lp05 --p-real 0.3 --wide-p 0.5 --w-lpips 0.05
run loss-lp15 --p-real 0.3 --wide-p 0.5 --w-lpips 0.15
run loss-ss30 --p-real 0.3 --wide-p 0.5 --w-ssim 0.30

echo -e "\n\n=============== in-distribution summary ==============="
python - <<'PY'
import csv, os, statistics as st
rows=[]
for r in sorted(os.listdir("runs")):
    p=f"runs/{r}/log.csv"
    if not os.path.isfile(p): continue
    d=list(csv.DictReader(open(p)))
    if len(d) < 20: continue
    t=d[-5:]
    rows.append((r, st.mean(float(x["val_psnr"]) for x in t),
                    st.mean(float(x["val_ssim"]) for x in t), len(d)))
rows.sort(key=lambda x: -x[1])
print(f"{'run':<16}{'PSNR':>10}{'SSIM':>9}{'epochs':>8}")
for r,p,s,n in rows: print(f"{r:<16}{p:>10.4f}{s:>9.4f}{n:>8}")
print("\nNOTE: this is the IN-DISTRIBUTION half only. Sweep each candidate with")
print("tools/noise_sweep.py and score on the 50/50 rule before choosing.")
PY
