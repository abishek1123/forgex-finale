#!/usr/bin/env bash
# Score EVERY finished run on all three axes, in one pass, on one machine.
#
#   bash cloud/score_all.sh [min_epochs]        # default 100
#
# Why one pass: comparing numbers produced by different script versions against
# different test sets is exactly the mistake in docs/ENGINEERING_LOG.md 11.2.
# Every model here is scored by the same code against the same inputs, and the
# shared bicubic column is the checksum.
set -uo pipefail
cd "$(dirname "$0")/.."
MIN="${1:-100}"

finished () {                     # finished <run-name> -> 0 if scoreable
  local n="$1"
  [ -f "runs/$n/last.pt" ] || return 1
  [ -f "runs/$n/log.csv" ] || return 1
  local e; e=$(tail -n +2 "runs/$n/log.csv" | wc -l)
  [ "$e" -ge "$MIN" ]
}

RUNS=()
for d in runs/*/; do n=$(basename "$d"); finished "$n" && RUNS+=("$n"); done
[ "${#RUNS[@]}" -gt 0 ] || { echo "no runs with >= $MIN epochs"; exit 1; }

echo "=============== runs found ($MIN+ epochs) ==============="
for n in "${RUNS[@]}"; do
  printf "  %-26s %s epochs\n" "$n" "$(tail -n +2 "runs/$n/log.csv" | wc -l)"
done

echo
echo "=============== 1. in-distribution, final-5 mean ==============="
python - "${RUNS[@]}" <<'PY'
import csv, sys, statistics as st
rows = []
for r in sys.argv[1:]:
    d = list(csv.DictReader(open(f"runs/{r}/log.csv")))
    t = d[-5:]
    rows.append((r, st.mean(float(x["val_psnr"]) for x in t),
                    st.mean(float(x["val_ssim"]) for x in t), len(d)))
rows.sort(key=lambda x: -x[1])
print(f"{'run':<26}{'PSNR':>10}{'SSIM':>9}{'ep':>6}")
for r, p, s, n in rows:
    print(f"{r:<26}{p:>10.4f}{s:>9.4f}{n:>6}")
PY

echo
echo "=============== 2. real held-out incl. LPIPS ==============="
for n in "${RUNS[@]}"; do
  python src/validate.py --data data --ckpt "runs/$n/last.pt" --tag "$n" --baseline 2>&1 | tail -2
done

echo
echo "=============== 3. nine-axis OOD suite ==============="
CK=(); for n in "${RUNS[@]}"; do CK+=("runs/$n/last.pt"); done
python tools/ood_suite.py --data data --ckpts "${CK[@]}"

echo
echo "=============== 4. nine-level noise sweeps ==============="
for n in "${RUNS[@]}"; do
  echo "--- $n"
  python tools/noise_sweep.py --data data --ckpt "runs/$n/last.pt" --tag "$n"
done

echo
echo "=============== done ==============="
echo "Bring it home with:"
echo "  tar czf scores.tar.gz noise_sweep_*.csv results.csv && runpodctl send scores.tar.gz"
