"""Dynamic-size ForgeX routing with visible, depth-preserving fallback.

Use one instance per concurrent worker. TensorRT is imported lazily, so a missing
TensorRT installation does not prevent PyTorch CPU/GPU inference.
"""
from pathlib import Path
import copy
import hashlib
import importlib
import importlib.util
import json
import logging
import sys
import time
import numpy as np
import torch
from profiles import choose_profile

DEPTHS = (3, 6, 10, 13, 16)
SUPPORTED_SIZES = (32, 128, 256, 512)


def shape_supported(shape, minimum, maximum):
    return (len(shape) == len(minimum) == len(maximum)
            and all(lo <= n <= hi for n, lo, hi in zip(shape, minimum, maximum)))


def depth_for_knob(knob):
    if isinstance(knob, bool) or not isinstance(knob, (int, np.integer)) or knob not in range(1, 6):
        raise ValueError("knob must be an integer 1 through 5")
    return DEPTHS[int(knob) - 1]


class Engine:
    def __init__(self, path, device=0):
        import tensorrt as trt
        self.device = torch.device(f"cuda:{device}")
        with torch.cuda.device(self.device):
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise RuntimeError("Engine load failed; rebuild for this GPU/TensorRT/platform")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT context allocation failed")
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            assert ins == ["input"] and outs == ["output"], (ins, outs)
            assert all(self.engine.get_tensor_dtype(n) == trt.float32 for n in names)
            self.minimum, _, self.maximum = [tuple(s) for s in self.engine.get_tensor_profile_shape("input", 0)]
            self.profiles = []
            for index in range(self.engine.num_optimization_profiles):
                lo, opt, hi = self.engine.get_tensor_profile_shape("input", index)
                self.profiles.append(dict(min=list(lo), opt=list(opt), max=list(hi)))
            self.profile_index = None
            self.shape = None
            self.output = None

    def __call__(self, x):
        shape = tuple(x.shape)
        profile_index = choose_profile(shape, self.profiles)
        if x.device != self.device or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError("Input must be contiguous float32 on the engine's CUDA device")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self.profile_index != profile_index:
                stream.synchronize()
                if not self.context.set_optimization_profile_async(profile_index, stream.cuda_stream):
                    raise RuntimeError("TensorRT profile selection failed")
                self.profile_index = profile_index
                self.shape = None
            if self.shape != shape:
                # Complete prior uses of reusable buffers before changing allocation.
                stream.synchronize()
                if not self.context.set_input_shape("input", shape):
                    raise RuntimeError(f"TensorRT rejected shape {shape}")
                out_shape = tuple(self.context.get_tensor_shape("output"))
                if out_shape != (shape[0], 1, 2 * shape[2], 2 * shape[3]):
                    raise RuntimeError(f"Unexpected output shape {out_shape}")
                self.output = torch.empty(out_shape, device=self.device, dtype=torch.float32)
                self.context.set_tensor_address("output", self.output.data_ptr())
                self.shape = shape
            self.context.set_tensor_address("input", x.data_ptr())
            self._input = x
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT enqueue failed")
        return self.output  # Reused next call; use restore() for an owned CPU result.


class ForgeXKnob:
    def __init__(self, directory, device=0):
        self.directory = Path(directory)
        self.device = device
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.engines = {}
        self.failed_engines = {}
        self.models = {}
        self.cuda_failed = False
        self.last_info = None

    def load_model(self, device="cpu"):
        device = str(device)
        if device in self.models:
            return self.models[device]
        if device != "cpu":
            self.models[device] = copy.deepcopy(self.load_model("cpu")).to(device).eval()
            return self.models[device]
        # Namespace the packaged architecture to avoid importing a host app's `arch`.
        source = self.directory / "source" / "arch"
        namespace = "_forgex_arch_" + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(namespace, source / "__init__.py",
                                                         submodule_search_locations=[str(source)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = package
            spec.loader.exec_module(package)
        klass = importlib.import_module(namespace + ".multiexit").MultiExitGate
        ck = torch.load(self.directory / "model_weights.pt", map_location="cpu", weights_only=True)
        cfg = ck["config"]
        model = klass(**{k: cfg[k] for k in ("ch", "nb", "scale", "res_scale", "exits", "variant")})
        model.load_state_dict(ck["state_dict"], strict=True)
        self.models["cpu"] = model.eval()
        return self.models["cpu"]

    def load_engine(self, depth):
        if depth in self.failed_engines:
            raise RuntimeError(self.failed_engines[depth])
        if depth not in self.engines:
            entry = self.manifest["engines"].get(str(depth))
            if not entry or entry["status"] != "smoke_passed_pending_full_evaluation":
                raise RuntimeError(f"Exit {depth} has no smoke-checked engine")
            path = self.directory / entry["engine"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["engine_sha256"]:
                raise RuntimeError("Engine checksum mismatch")
            self.engines[depth] = Engine(path, self.device)
        return self.engines[depth]

    def _torch(self, arr, depth, device):
        with torch.inference_mode():
            x = torch.from_numpy(arr)[None, None].to(device)
            y = self.load_model(device).forward_depth(x, depth)
            result = y.detach().float().cpu().numpy()[0, 0].copy()
        if not np.isfinite(result).all():
            raise RuntimeError("PyTorch produced nonfinite output")
        return result

    def restore_batch(self, images, knob=5, backend="auto"):
        """Same-shape batch; return owned outputs and actual backend per image."""
        depth = depth_for_knob(knob)
        if backend not in ("auto", "tensorrt", "pytorch", "cpu"):
            raise ValueError("Invalid backend")
        arrays = [np.ascontiguousarray(x, dtype=np.float32) for x in images]
        if not arrays:
            raise ValueError("Empty batch")
        shape = arrays[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] not in SUPPORTED_SIZES:
            raise ValueError("Unsupported input size")
        if any(x.shape != shape or not np.isfinite(x).all() for x in arrays):
            raise ValueError("Batch inputs must be finite and have identical shapes")
        reason = None
        if backend in ("auto", "tensorrt") and torch.cuda.is_available() and not self.cuda_failed:
            try:
                with torch.inference_mode():
                    x = torch.from_numpy(np.stack(arrays)[:, None]).to(f"cuda:{self.device}")
                    result = self.load_engine(depth)(x).detach().float().cpu().numpy()[:, 0].copy()
                if result.shape != (len(arrays), shape[0]*2, shape[1]*2) or not np.isfinite(result).all():
                    raise RuntimeError("Invalid batched TensorRT output")
                np.clip(result, 0, 1, out=result)
                return list(result), [dict(depth=depth, knob=knob, backend="tensorrt", fallback=False,
                                          reasons=[], batch=len(arrays)) for _ in arrays]
            except Exception as exc:
                if backend == "tensorrt":
                    raise
                reason = f"batch_tensorrt_failed: {type(exc).__name__}: {exc}"
                logging.warning(reason)
        outputs, records = [], []
        for arr in arrays:
            result, info = self.restore(arr, knob, backend=backend, return_info=True)
            if reason:
                info["reasons"].insert(0, reason)
                info["fallback"] = True
            outputs.append(result)
            records.append(info)
        return outputs, records

    def restore(self, image, knob=5, backend="auto", clamp=True, return_info=False):
        depth = depth_for_knob(knob)
        if backend not in ("auto", "tensorrt", "pytorch", "cpu"):
            raise ValueError("backend must be auto, tensorrt, pytorch or cpu")
        arr = np.ascontiguousarray(image, dtype=np.float32)
        if arr.ndim != 2 or min(arr.shape) < 1 or not np.isfinite(arr).all():
            raise ValueError("Expected a finite, nonempty 2D grayscale array")
        if arr.shape[0] != arr.shape[1] or arr.shape[0] not in SUPPORTED_SIZES:
            raise ValueError("Supported inputs are 32x32, 128x128, 256x256 and 512x512")
        start = time.perf_counter()
        reasons = []
        result, used = None, None
        profile = self.manifest["identity"]["profile"]
        supported = shape_supported((1, 1, *arr.shape), profile["min"], profile["max"])
        cuda = torch.cuda.is_available() and not self.cuda_failed
        if backend in ("auto", "tensorrt"):
            if not supported:
                reasons.append("outside_tensorrt_profile")
            elif not cuda:
                reasons.append("cuda_unavailable")
            else:
                try:
                    engine = self.load_engine(depth)
                    with torch.inference_mode():
                        x = torch.from_numpy(arr)[None, None].to(f"cuda:{self.device}")
                        # CPU transfer synchronizes; asynchronous errors are caught here.
                        result = engine(x).detach().cpu().numpy()[0, 0].copy()
                    if not np.isfinite(result).all():
                        raise RuntimeError("TensorRT produced nonfinite output")
                    used = "tensorrt"
                except Exception as exc:
                    result = None
                    engine = None
                    x = None
                    reason = f"tensorrt_failed: {type(exc).__name__}: {exc}"
                    reasons.append(reason)
                    self.failed_engines[depth] = reason
                    self.engines.pop(depth, None)
            if backend == "tensorrt" and result is None:
                raise RuntimeError("Strict TensorRT mode: " + "; ".join(reasons))
        if result is None and backend != "cpu" and cuda:
            try:
                result = self._torch(arr, depth, f"cuda:{self.device}")
                used = "pytorch_cuda"
            except Exception as exc:
                reasons.append(f"pytorch_cuda_failed: {type(exc).__name__}: {exc}")
                self.cuda_failed = True
                self.models.pop(f"cuda:{self.device}", None)
                self.engines.clear()
                # Subsequent calls avoid a possibly damaged CUDA context.
        if result is None:
            try:
                result = self._torch(arr, depth, "cpu")
                used = "pytorch_cpu"
            except Exception as exc:
                raise RuntimeError("All available inference paths failed. " + "; ".join(reasons)) from exc
        if result.shape != (arr.shape[0] * 2, arr.shape[1] * 2):
            raise RuntimeError(f"Wrong output dimensions: {result.shape}")
        if clamp:
            np.clip(result, 0, 1, out=result)
        info = dict(depth=depth, knob=int(knob), backend=used, input_shape=list(arr.shape),
                    output_shape=list(result.shape), fallback=bool(reasons), reasons=reasons,
                    wall_ms=(time.perf_counter() - start) * 1000)
        self.last_info = info
        if reasons:
            logging.warning("ForgeX depth=%s backend=%s reason=%s", depth, used, "; ".join(reasons))
        return (result, info) if return_info else result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engines", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--knob", type=int, choices=range(1, 6), default=5)
    p.add_argument("--backend", choices=("auto", "tensorrt", "pytorch", "cpu"), default="auto")
    a = p.parse_args()
    y, info = ForgeXKnob(a.engines).restore(np.load(a.input), a.knob, a.backend, return_info=True)
    np.save(a.output, y)
    print(json.dumps(info, indent=2))
