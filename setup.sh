#!/usr/bin/env bash
# ForgeX one-command setup (Linux / macOS / RunPod pod).
#
#   git clone https://github.com/abishek1123/forgex-kla-ps01.git
#   cd forgex-kla-ps01
#   bash setup.sh
#
# Creates .venv, installs dependencies, then runs the laptop check.
# Safe to re-run.
set -u
cd "$(dirname "$0")"

echo
echo "  ForgeX setup"
echo "  ------------------------------------------------------------"

PY=""
for c in python3.13 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;sys.exit(sys.version_info[0]!=3)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "  FAIL: no Python 3 on PATH."; exit 1
fi
echo "  python      : $PY ($($PY --version 2>&1))"

if [ -x .venv/bin/python ]; then
  echo "  venv        : reusing .venv"
else
  echo "  venv        : creating .venv ..."
  "$PY" -m venv .venv || { echo "  FAIL: could not create .venv"; exit 1; }
fi
VPY=.venv/bin/python

if command -v nvidia-smi >/dev/null 2>&1; then
  REQ=requirements.txt;      echo "  gpu         : nvidia-smi found"
else
  REQ=requirements-cpu.txt;  echo "  gpu         : none -- installing CPU build"
fi
[ -f "$REQ" ] || REQ=requirements.txt
echo "  installing  : $REQ  (this takes a few minutes)"

"$VPY" -m pip install --upgrade pip --quiet
if ! "$VPY" -m pip install -r "$REQ" --quiet; then
  echo
  echo "  Pinned install failed -- most likely this Python is not 3.13,"
  echo "  so the exact torch wheel does not exist for it."
  echo "  Falling back to an unpinned install, enough to run inference."
  echo
  "$VPY" -m pip install torch numpy --quiet || {
    echo "  FAIL: could not install torch. Tell Abishek."; exit 1; }
  echo "  NOTE: this machine is INFERENCE-ONLY (no training deps)."
fi

mkdir -p outputs runs

echo
"$VPY" tools/doctor.py
code=$?

echo "  From now on, use this python for everything:"
echo "      ./.venv/bin/python <script>"
echo
exit $code
