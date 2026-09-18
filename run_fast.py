#!/usr/bin/env python3
"""Kept for compatibility. run.py now dispatches to TensorRT by itself.

Earlier versions of this submission split the runtime choice into a separate
entry point. It is no longer separate: the TensorRT probe lives at the top of
run.py, above `import torch`, because that is the only place it can pay for
itself. This file simply forwards, so any script or note that still says
`run_fast.py` keeps working.

    python run.py <input-dir> <output-dir> [--no-trt]
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.execv(sys.executable, [sys.executable, os.path.join(HERE, "run.py")] + sys.argv[1:])
