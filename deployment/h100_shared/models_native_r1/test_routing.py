"""Unit tests for routing contracts and fallback; no GPU/model is required.

These tests inject backend failures. They do not establish actual TensorRT or
PyTorch numeric equivalence; validate_fallback.py does that on the pod.
"""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

if importlib.util.find_spec("torch") is None:
    sys.modules["torch"] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True), inference_mode=nullcontext)
import knob_runtime as kr


class RoutingTests(unittest.TestCase):
    def make(self, minimum=32, maximum=512):
        obj = kr.ForgeXKnob.__new__(kr.ForgeXKnob)
        obj.device = 0
        obj.manifest = {"identity": {"profile": {"min": [1, 1, minimum, minimum], "max": [1, 1, maximum, maximum]}}}
        obj.engines, obj.failed_engines, obj.models = {}, {}, {}
        obj.cuda_failed, obj.last_info = False, None
        return obj

    def test_knob_depths(self):
        self.assertEqual([kr.depth_for_knob(k) for k in range(1, 6)], [3, 6, 10, 13, 16])
        for k in (0, 6, True, 1.0, "3"):
            with self.assertRaises(ValueError):
                kr.depth_for_knob(k)

    def test_profile_boundaries(self):
        for s in kr.SUPPORTED_SIZES:
            self.assertTrue(kr.shape_supported((1, 1, s, s), (1, 1, 32, 32), (1, 1, 512, 512)))
        self.assertFalse(kr.shape_supported((1, 1, 513, 513), (1, 1, 32, 32), (1, 1, 512, 512)))

    def test_bad_inputs_fail_before_backend(self):
        obj = self.make()
        for shape in ((64, 64), (512, 256), (1024, 1024), (32, 32, 3)):
            with self.assertRaises(ValueError):
                obj.restore(np.zeros(shape, np.float32))
        with self.assertRaises(ValueError):
            obj.restore(np.full((32, 32), np.nan, np.float32))

    def test_missing_engine_falls_back_at_every_depth(self):
        for knob, depth in enumerate(kr.DEPTHS, 1):
            obj = self.make()
            with patch.object(kr.torch.cuda, "is_available", return_value=True), \
                 patch.object(obj, "load_engine", side_effect=FileNotFoundError("injected missing engine")), \
                 patch.object(obj, "_torch", return_value=np.full((64, 64), 1.25, np.float32)) as fallback:
                out, info = obj.restore(np.zeros((32, 32), np.float32), knob, return_info=True)
                self.assertEqual(fallback.call_args.args[1:], (depth, "cuda:0"))
                self.assertEqual(info["backend"], "pytorch_cuda")
                self.assertTrue(info["fallback"])
                self.assertEqual(float(out.max()), 1.0)

    def test_strict_tensorrt_does_not_hide_failure(self):
        obj = self.make()
        with patch.object(kr.torch.cuda, "is_available", return_value=True), \
             patch.object(obj, "load_engine", side_effect=RuntimeError("injected failure")), \
             patch.object(obj, "_torch") as fallback:
            with self.assertRaisesRegex(RuntimeError, "Strict TensorRT"):
                obj.restore(np.zeros((32, 32), np.float32), backend="tensorrt")
            fallback.assert_not_called()

    def test_gpu_failure_uses_cpu_preserving_depth(self):
        obj = self.make()
        with patch.object(kr.torch.cuda, "is_available", return_value=True), \
             patch.object(obj, "load_engine", side_effect=RuntimeError("injected TRT failure")), \
             patch.object(obj, "_torch", side_effect=[RuntimeError("injected CUDA OOM"), np.zeros((64, 64), np.float32)]) as fallback:
            _, info = obj.restore(np.zeros((32, 32), np.float32), knob=2, return_info=True)
            self.assertEqual(info["backend"], "pytorch_cpu")
            self.assertEqual(fallback.call_args.args[1:], (6, "cpu"))
            self.assertTrue(obj.cuda_failed)

    def test_no_cuda_routes_cpu(self):
        obj = self.make()
        with patch.object(kr.torch.cuda, "is_available", return_value=False), \
             patch.object(obj, "_torch", return_value=np.zeros((256, 256), np.float32)) as fallback:
            _, info = obj.restore(np.zeros((128, 128), np.float32), knob=4, return_info=True)
            self.assertEqual(info["backend"], "pytorch_cpu")
            self.assertEqual(fallback.call_args.args[1:], (13, "cpu"))

    def test_all_paths_fail_explicitly(self):
        obj = self.make()
        with patch.object(kr.torch.cuda, "is_available", return_value=False), \
             patch.object(obj, "_torch", side_effect=MemoryError("injected CPU failure")):
            with self.assertRaisesRegex(RuntimeError, "All available inference paths failed"):
                obj.restore(np.zeros((32, 32), np.float32))


if __name__ == "__main__":
    unittest.main()
