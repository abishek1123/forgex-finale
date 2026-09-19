"""Root-runner compatibility and optional-engine dispatch contracts."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from tools import shared_backend as backend
sys.path.insert(0, str(ROOT / 'deployment/h100_shared'))
from knob_runtime import ForgeXKnob


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.directory = ROOT / 'deployment/h100_shared/models_native_r1'
        cls.ck = torch.load(cls.directory / 'model_weights.pt', weights_only=True)
        cls.reference = ForgeXKnob(cls.directory).load_model('cpu')

    def options(self, **changes):
        values = dict(no_trt=False, half=False, fp16=True, profile=False,
                      weights=str(self.directory / 'model_weights.pt'))
        values.update(changes)
        return SimpleNamespace(**values)

    def test_root_matches_shared_reference_at_all_exits(self):
        model, name = run.build_model(self.ck['config'], self.ck['state_dict'])
        model.load_state_dict(self.ck['state_dict'])
        self.assertEqual(name, 'shared_gate')
        body = model.body
        x = torch.rand(1, 1, 32, 32)
        with torch.inference_mode():
            for depth in backend.DEPTHS:
                model.body = torch.nn.Sequential(*list(body)[:depth])
                torch.testing.assert_close(model(x), self.reference.forward_depth(x, depth), rtol=0, atol=0)

    def test_legacy_gate_loading_remains_available(self):
        cfg = dict(self.ck['config'], kind='gate')
        model, name = run.build_model(cfg, self.ck['state_dict'])
        model.load_state_dict(self.ck['state_dict'], strict=True)
        self.assertEqual(name, 'e1f_gate')

    def test_adapters_are_rejected_instead_of_ignored(self):
        with self.assertRaises(ValueError):
            run.build_model(dict(self.ck['config'], variant='adapter'))

    def test_options_and_other_weights_bypass_engines(self):
        model = SimpleNamespace(body=list(range(16)))
        inputs = [np.zeros((128, 128), np.float32)]
        for changes in [dict(no_trt=True), dict(half=True), dict(fp16=False), dict(profile=True)]:
            self.assertIsNone(backend.try_restore(model, inputs, self.options(**changes), ROOT))
        with patch.object(backend, 'digest', side_effect=['custom', 'engine']):
            self.assertIsNone(backend.try_restore(model, inputs, self.options(), ROOT))

    def test_engine_batches_preserve_depth_and_use_profile_limit(self):
        model = SimpleNamespace(body=list(range(6)))
        calls = []
        def restore(images, knob, backend):
            calls.append((len(images), knob, backend))
            return [np.zeros((256, 256), np.float32) for _ in images], []
        runtime = SimpleNamespace(restore_batch=restore)
        stream = SimpleNamespace(synchronize=lambda: None)
        with patch.object(backend, '_runtime', runtime), patch.object(backend, '_stream', stream), \
             patch.object(backend, '_failed', set()), \
             patch.dict(sys.modules, {'tensorrt': SimpleNamespace(__version__='10.13.3.9')}), \
             patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
             patch.object(torch.cuda, 'stream', return_value=nullcontext()):
            output = backend.try_restore(model, [np.zeros((128, 128), np.float32)] * 33,
                                         self.options(), ROOT)
        self.assertEqual(len(output), 33)
        self.assertEqual(calls, [(16, 2, 'tensorrt'), (16, 2, 'tensorrt'), (1, 2, 'tensorrt')])

    def test_incompatible_engine_runtime_falls_back(self):
        with patch.object(backend, '_failed', set()), \
             patch.dict(sys.modules, {'tensorrt': SimpleNamespace(__version__='incompatible')}):
            self.assertIsNone(backend.try_restore(SimpleNamespace(body=list(range(16))),
                [np.zeros((128, 128), np.float32)], self.options(), ROOT))
            self.assertIn(16, backend._failed)


if __name__ == '__main__':
    unittest.main()
