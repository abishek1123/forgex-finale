#!/usr/bin/env bash
# Score every finished checkpoint on a NEW paired dataset, in one pass.
#
#   bash cloud/score_testset.sh <data-dir> [min_epochs]
#
# <data-dir> must contain GT/ and NoisyLR/. Every image is scored (n-val is set
# far above the set size), all models see identical inputs, and the shared
# bicubic row is the checksum.
set -uo pipefail
D="${1:?usage: score_testset.sh <data-dir> [min_epochs]}"
MIN="${2:-100}"
[ -d "$D/GT" ] && [ -d "$D/NoisyLR" ] || { echo "!! $D needs GT/ and NoisyLR/"; ls "$D"; exit 1; }

N=$(ls "$D/GT" | wc -l); M=$(ls "$D/NoisyLR" | wc -l)
echo "test set: $N GT, $M NoisyLR"
[ "$N" = "$M" ] || { echo "!! count mismatch -- did an unzip finish?"; exit 1; }

python - "$D" <<'PY'
import numpy as np, os, sys, collections
d=sys.argv[1]
g=sorted(os.listdir(d+"/GT")); l=sorted(os.listdir(d+"/NoisyLR"))
assert [x[:-4] for x in g]==[x[:-4] for x in l], "GT and NoisyLR filenames differ"
sg=collections.Counter(); sl=collections.Counter(); mn=[]; mx=[]
for n in g[:400]:
    a=np.load(f"{d}/GT/{n}"); b=np.load(f"{d}/NoisyLR/{n}")
    sg[a.shape]+=1; sl[b.shape]+=1; mn.append(float(b.min())); mx.append(float(b.max()))
print("GT shapes     ", dict(sg))
print("NoisyLR shapes", dict(sl))
print("NoisyLR range  min %.4f  max %.4f  (outside [0,1] is expected)"%(min(mn),max(mx)))
PY

RUNS=()
for p in runs/*/; do n=$(basename "$p")
  [ -f "runs/$n/last.pt" ] && [ -f "runs/$n/log.csv" ] \
    && [ "$(tail -n +2 runs/$n/log.csv | wc -l)" -ge "$MIN" ] && RUNS+=("$n")
done
[ "${#RUNS[@]}" -gt 0 ] || { echo "!! no runs with >= $MIN epochs"; ls runs; exit 1; }
echo; echo "scoring ${#RUNS[@]} checkpoints on the FULL test set"

rm -f testset_results.csv
for n in "${RUNS[@]}"; do
  echo "--- $n"
  python src/validate.py --data "$D" --ckpt "runs/$n/last.pt" --tag "$n" \
         --n-val 1000000 --baseline --csv testset_results.csv 2>&1 | tail -2
done

echo; echo "================ RANKED ON THE ORGANISERS' TEST SET ================"
python - <<'PY'
import csv
r=list(csv.DictReader(open("testset_results.csv")))
seen={}
for x in r: seen[x["tag"]]=x          # last row per tag wins
bic=[x for x in seen.values() if x["tag"].startswith("bicubic")]
rows=[(k,float(v["psnr"]),float(v["ssim"]),float(v["lpips"])) for k,v in seen.items()]
rows.sort(key=lambda x:-x[1])
print(f"{'run':<26}{'PSNR':>10}{'SSIM':>9}{'LPIPS':>9}")
for k,p,s,l in rows: print(f"{k:<26}{p:>10.4f}{s:>9.5f}{l:>9.5f}")
print("\nby LPIPS:")
for k,p,s,l in sorted(rows,key=lambda x:x[3]): print(f"{k:<26}{p:>10.4f}{s:>9.5f}{l:>9.5f}")
PY
