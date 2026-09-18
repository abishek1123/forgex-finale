#!/usr/bin/env bash
# One command, from a fresh pod to a finished experiment queue.
#
#   nohup bash cloud/go.sh <runpodctl-code> > go.log 2>&1 &
#   tail -f go.log
#
# Survives the Jupyter terminal dropping, which it will. Every stage checks its
# own result and stops loudly rather than continuing on bad data.
set -uo pipefail
cd "$(dirname "$0")/.."
CODE="${1:-}"
EPOCHS="${2:-40}"

say () { echo -e "\n\033[1m=== $* ===\033[0m"; }
die () { echo -e "\n!!! $*"; exit 1; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU"

# ---------------------------------------------------------------- data
if [ -d data/GT ]; then
  say "data already present"
elif [ -n "$CODE" ]; then
  say "receiving data  ($CODE)"
  runpodctl receive "$CODE" || die "transfer failed -- get a fresh code with 'runpodctl send' on the laptop"
  [ -f semicon_train_data.zip ] && { say "unzipping"; unzip -q -o semicon_train_data.zip; }
  [ -d semicon_train_data ] && mv semicon_train_data data
  [ -d data/semicon_train_data/GT ] && mv data/semicon_train_data/* data/ 2>/dev/null
else
  die "no data and no transfer code. Run: bash cloud/go.sh <code>"
fi

N=$(ls data/GT 2>/dev/null | wc -l)
M=$(ls data/NoisyLR 2>/dev/null | wc -l)
say "data check"
echo "GT $N   NoisyLR $M"
[ "$N" -gt 4000 ] && [ "$N" = "$M" ] || die "expected ~4785 matched pairs, got GT=$N NoisyLR=$M"

# ---------------------------------------------------------------- deps
say "deps"
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu126
python -c "import lpips" 2>/dev/null || pip install -q lpips
python -c "import numpy, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

say "smoke"
python src/train.py --data data --smoke --wide-p 0.5 2>&1 | tail -3 || die "smoke test failed"

# ---------------------------------------------------------------- run
CORES=$(nproc)
W=$(( CORES / 2 )); [ "$W" -gt 8 ] && W=8; [ "$W" -lt 2 ] && W=2
say "launching queue   ${CORES} cores -> --workers $W   ${EPOCHS} epochs each"
sed -i "s/--amp\"/--amp --workers $W\"/" cloud/run_experiments.sh 2>/dev/null
grep -q "workers" cloud/run_experiments.sh || echo "(worker patch skipped -- already set or pattern changed)"

bash cloud/run_experiments.sh "$EPOCHS"

say "DONE"
echo "Checkpoints in runs/. Send them back with:"
echo "  tar czf best.tar.gz runs/*/last.pt && runpodctl send best.tar.gz"
