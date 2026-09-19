#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  '') REQ=requirements-inference.txt ;;
  --cpu) REQ=requirements-cpu.txt ;;
  --h100) REQ=requirements-h100.txt
    [ "$(uname -s)" = Linux ] || { echo 'H100 engines require Linux'; exit 1; } ;;
  *) echo 'Usage: bash setup.sh [--cpu|--h100]'; exit 1 ;;
esac
PY=''
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))'; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo 'Install Python 3.12'; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -r "$REQ"
.venv/bin/python tools/doctor.py
