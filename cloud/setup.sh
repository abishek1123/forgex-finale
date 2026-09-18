#!/usr/bin/env bash
# Bring a fresh RunPod (or any CUDA box) up to speed. Run once, from the repo root.
#
#   bash cloud/setup.sh
#
# Assumes a PyTorch CUDA image. Verifies the GPU before installing anything, so
# a CPU-only pod fails here rather than four hours into a training run.
set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "NO GPU VISIBLE -- stop, this pod is useless for training."; exit 1; }

echo -e "\n=== python ==="
python -c "import sys; print(sys.version)"

echo -e "\n=== deps ==="
python -m pip install -q --upgrade pip
python -c "import torch" 2>/dev/null || python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -q numpy lpips pillow

echo -e "\n=== torch sees the GPU? ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
assert torch.cuda.is_available(), "CUDA not available -- fix the pod image before training"
PY

echo -e "\n=== data ==="
if [ -d "data/GT" ]; then
  echo "GT pairs: $(ls data/GT | wc -l)   NoisyLR: $(ls data/NoisyLR | wc -l)"
else
  echo "NO DATA at ./data -- send it over before running anything:"
  echo "  on the laptop:  runpodctl send semicon_train_data"
  echo "  on the pod:     runpodctl receive <code>  &&  mv semicon_train_data data"
fi

echo -e "\n=== smoke ==="
python src/train.py --data data --smoke --wide-p 0.5 2>&1 | tail -4
echo -e "\nsetup OK."
