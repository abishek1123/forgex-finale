"""Real weights at all 20 depth/size combinations; simulated TRT failures.

This never renames or damages an engine. It injects failure at the loader and
checks the actual PyTorch fallback's output against a CPU reference.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from knob_runtime import DEPTHS, SUPPORTED_SIZES, ForgeXKnob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(44)
    obj = ForgeXKnob(a.engines)
    rows = []
    for knob, depth in enumerate(DEPTHS, 1):
        for size in SUPPORTED_SIZES:
            x = rng.uniform(0, 1.5, (size, size)).astype(np.float32)
            ref = obj.restore(x, knob, backend="cpu", clamp=False)
            for failure in (FileNotFoundError("injected absent engine"), RuntimeError("injected TRT failure")):
                with patch.object(obj, "load_engine", side_effect=failure):
                    actual, info = obj.restore(x, knob, clamp=False, return_info=True)
                assert info["backend"] in ("pytorch_cuda", "pytorch_cpu") and info["fallback"]
                np.testing.assert_allclose(actual, ref, atol=1e-4, rtol=1e-3)
                rows.append(dict(depth=depth, size=size, failure=type(failure).__name__,
                                 backend=info["backend"], max_abs=float(np.max(np.abs(actual-ref)))))
            print(f"PASS fallback depth={depth} size={size}", flush=True)
    # CPU-only path: real model while CUDA availability is hidden from routing.
    with patch.object(torch.cuda, "is_available", return_value=False):
        _, info = obj.restore(np.zeros((32, 32), np.float32), knob=3, return_info=True)
        assert info["backend"] == "pytorch_cpu" and info["depth"] == 10
    path = Path(a.engines) / "fallback_validation.json"
    path.write_text(json.dumps(dict(passed=True, cases=rows, cpu_only_passed=True), indent=2))
    print("Fallback validation PASSED; real weights, 20 depth/size combinations.", flush=True)


if __name__ == "__main__":
    main()
