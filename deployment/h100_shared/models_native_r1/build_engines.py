"""Export five true prefixes; compare PyTorch/ONNX/TensorRT before accepting.

Run on the deployment GPU with TensorRT 10.13.3.9. No training is performed.
Smoke checks are numerical deployment checks, not full held-out quality scores.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import tensorrt as trt
from knob_runtime import DEPTHS, SUPPORTED_SIZES, Engine
from profiles import PROFILES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


class Prefix(nn.Module):
    def __init__(self, model, depth, bicubic):
        super().__init__()
        self.stem = copy.deepcopy(model.stem)
        self.head = copy.deepcopy(model.head)
        self.body = nn.Sequential(*[copy.deepcopy(b) for b in list(model.body)[:depth]])
        self.body_tail = copy.deepcopy(model.body_tail)
        self.out = copy.deepcopy(model.out)
        self.scale = model.scale
        self.bicubic = bicubic

    def forward(self, x):
        stem = self.head(self.stem(x))
        features = stem + self.body_tail(self.body(stem))
        return self.bicubic(x, self.scale) + self.out(features).float()


def compare(ref, actual, atol, rtol, label):
    if not np.isfinite(actual).all():
        raise RuntimeError(f"{label}: nonfinite output")
    err = actual.astype(np.float64) - ref.astype(np.float64)
    stats = dict(max_abs=float(np.max(np.abs(err))), rmse=float(np.sqrt(np.mean(err**2))))
    if atol is not None:
        np.testing.assert_allclose(actual, ref, atol=atol, rtol=rtol, err_msg=label)
    return stats


def parse_and_build(path, dest, fp32, hardware):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(path.read_bytes()):
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    if hardware == "ampere_plus":
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    for bounds in PROFILES:
        profile = builder.create_optimization_profile()
        profile.set_shape("input", bounds["min"], bounds["opt"], bounds["max"])
        actual = [tuple(s) for s in profile.get_shape("input")]
        expected = [tuple(bounds[key]) for key in ("min", "opt", "max")]
        if actual != expected or not profile:
            raise RuntimeError(f"Invalid optimization profile: {actual}, expected {expected}")
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("Could not register optimization profile")
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not fp32:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        sensitive = {trt.LayerType.REDUCE, trt.LayerType.ELEMENTWISE,
                     trt.LayerType.UNARY, trt.LayerType.RESIZE}
        for layer in network:
            # Keep normalization, multiplication, sqrt/log and bicubic/addition FP32.
            # Conv layers may use FP16 tactics. Inputs/outputs stay FP32.
            float_outputs = [i for i in range(layer.num_outputs)
                             if layer.get_output(i).dtype == trt.float32]
            if layer.type in sensitive and float_outputs:
                layer.precision = trt.float32
                for i in float_outputs:
                    layer.set_output_type(i, trt.float32)
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.float32
    start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT build failed; inspect the preceding builder error")
    tmp = Path(str(dest) + ".tmp")
    tmp.write_bytes(bytes(plan))
    tmp.replace(dest)
    print(f"BUILT {dest.name}: {time.perf_counter()-start:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/restored_backup/kla2")
    p.add_argument("--ckpt", default="/workspace/restored_backup/kla2/runs/mx120-s0-shared/best_frontier.pt")
    p.add_argument("--data", default="/workspace/test1197")
    p.add_argument("--out", default="/workspace/forgex_h100_v3/models_native")
    p.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS), choices=DEPTHS)
    p.add_argument("--fp32", action="store_true", help="Diagnostic alternative; use a separate --out")
    p.add_argument("--hardware", choices=("native", "ampere_plus"), default="native")
    a = p.parse_args()
    if trt.__version__ != "10.13.3.9":
        raise RuntimeError(f"Expected TensorRT 10.13.3.9, found {trt.__version__}")
    sys.path.insert(0, str(Path(a.root).resolve()))
    from arch.multiexit import MultiExitGate
    from arch.registry import bicubic_up
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["kind"] == "multiexit_gate" and cfg["variant"] == "shared", cfg
    assert tuple(cfg["exits"]) == DEPTHS and cfg["scale"] == 2, cfg
    model = MultiExitGate(**{k: cfg[k] for k in ("ch", "nb", "scale", "res_scale", "exits", "variant")})
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    precision = "fp32" if a.fp32 else "mixed_fp16"
    source_hashes = {name: sha(Path(a.root) / "arch" / name)
                     for name in ("multiexit.py", "registry.py", "attn.py")}
    identity = dict(checkpoint_sha256=sha(a.ckpt), config=cfg, source_hashes=source_hashes,
                    builder_sha256=sha(__file__), runtime_sha256=sha(Path(__file__).with_name("knob_runtime.py")),
                    precision=precision, hardware_compatibility=a.hardware,
                    target_deployment_gpu="NVIDIA H100 (final performance must be measured there)",
                    tensorrt=trt.__version__, torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    compute_capability=list(torch.cuda.get_device_capability(0)),
                    profiles=PROFILES, input_shape=["N", 1, "H", "W"], output_shape=["N", 1, "2H", "2W"],
                    supported_square_sizes=list(SUPPORTED_SIZES),
                    profile={"min": [1, 1, 32, 32], "opt": [1, 1, 128, 128], "max": [1, 1, 512, 512]})
    manifest_path = out / "manifest.json"
    manifest = dict(identity=identity, engines={})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise RuntimeError("Output belongs to another build; choose a new --out directory")
    write_json(manifest_path, manifest)
    shutil.copyfile(Path(__file__).with_name("knob_runtime.py"), out / "knob_runtime.py")
    shutil.copyfile(__file__, out / "build_engines.py")
    readme = Path(__file__).with_name("README.md")
    if readme.exists():
        shutil.copyfile(readme, out / "README.md")
    for filename in ("evaluate_dataset.py", "benchmark.py", "test_routing.py", "validate_fallback.py", "profiles.py"):
        sibling = Path(__file__).with_name(filename)
        if sibling.exists():
            shutil.copyfile(sibling, out / filename)
    # Save inference weights only, and the architecture needed to reproduce export.
    torch.save(dict(config=cfg, state_dict=ck["state_dict"]), out / "model_weights.pt")
    # Network volumes can reject timestamp/permission preservation. Copy bytes only.
    source_arch = Path(a.root) / "arch"
    for source_file in source_arch.rglob("*"):
        if source_file.is_file() and "__pycache__" not in source_file.parts and source_file.suffix != ".pyc":
            destination = out / "source" / "arch" / source_file.relative_to(source_arch)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
    (out / "requirements-lock.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))

    lrdir = Path(a.data) / "NoisyLR"
    files = sorted(lrdir.glob("*.npy"))
    assert len(files) == 1197, f"Expected 1197 test inputs, got {len(files)}"
    indices = np.linspace(0, len(files)-1, 8, dtype=int)
    samples = [(files[i].name, np.load(files[i]).astype(np.float32)) for i in indices]
    samples += [("probe_zero", np.zeros((128, 128), np.float32)),
                ("probe_above_one", np.full((128, 128), 1.5, np.float32))]
    rng = np.random.default_rng(0)
    for size in SUPPORTED_SIZES:
        if size != 128:
            samples.append((f"size_probe_{size}", rng.uniform(0, 1.5, (size, size)).astype(np.float32)))
    for name, arr in samples:
        assert arr.shape[0] == arr.shape[1] and arr.shape[0] in SUPPORTED_SIZES and np.isfinite(arr).all(), name
    samples = [(name, torch.from_numpy(np.ascontiguousarray(arr))[None, None]) for name, arr in samples]
    for size, batch in ((128, 4), (256, 2), (128, 16), (256, 8)):
        batch_array = rng.uniform(0, 1.5, (batch, 1, size, size)).astype(np.float32)
        samples.append((f"batch_probe_{batch}x{size}", torch.from_numpy(batch_array)))
    dummy = samples[0][1]
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 4
    ort_options.inter_op_num_threads = 1
    ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    for depth in a.depths:
        entry = manifest["engines"].get(str(depth), {})
        if (entry.get("status") == "smoke_passed_pending_full_evaluation"
            and (out / entry["engine"]).exists() and (out / entry["onnx"]).exists()
            and sha(out / entry["engine"]) == entry["engine_sha256"]
            and sha(out / entry["onnx"]) == entry["onnx_sha256"]):
            print(f"REUSE depth={depth}: already built and smoke checked", flush=True)
            continue
        print(f"EXPORT depth={depth}", flush=True)
        wrapper = Prefix(model, depth, bicubic_up).eval()
        assert len(wrapper.body) == depth
        with torch.inference_mode():
            for name, x in samples:
                ref = model.forward_depth(x, depth)
                torch.testing.assert_close(wrapper(x), ref, atol=1e-6, rtol=1e-5)
        onnx_path = out / f"forgex_exit{depth:02d}.onnx"
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, opset_version=17,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "output": {0: "N", 2: "H2", 3: "W2"}},
                          do_constant_folding=True)
        graph = onnx.load(str(onnx_path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(onnx_path), sess_options=ort_options,
                                       providers=["CPUExecutionProvider"])
        refs, onnx_errors = [], []
        with torch.inference_mode():
            for name, x in samples:
                ref = wrapper(x).numpy()
                actual = session.run(["output"], {"input": x.numpy()})[0]
                onnx_errors.append(dict(sample=name, **compare(ref, actual, 1e-4, 1e-3, f"ONNX exit{depth}/{name}")))
                refs.append(ref)
        del session
        print(f"ONNX PASSED depth={depth}; {len(samples)} probes", flush=True)
        engine_path = out / f"forgex_exit{depth:02d}_{precision}.engine"
        parse_and_build(onnx_path, engine_path, a.fp32, a.hardware)
        runtime = Engine(engine_path)
        errors = []
        with torch.inference_mode():
            for (name, x), ref in zip(samples, refs):
                actual = runtime(x.cuda()).cpu().numpy().copy()
                stats = compare(ref, actual, None, 0, f"TensorRT exit{depth}/{name}")
                errors.append(dict(sample=name, **stats))
                # Conservative initial screen only. Full metrics must follow.
                max_allowed, rmse_allowed = ((5e-4, 1e-4) if a.fp32 else (0.02, 0.002))
                if stats["max_abs"] > max_allowed or stats["rmse"] > rmse_allowed:
                    raise RuntimeError(f"TensorRT numeric check failed: exit{depth}/{name} {stats}. "
                                       "Engine is NOT approved; send this log for diagnosis.")
        inspector = runtime.engine.create_engine_inspector()
        (out / f"exit{depth:02d}_layers.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        manifest["engines"][str(depth)] = dict(
            status="smoke_passed_pending_full_evaluation", depth=depth,
            active_parameters=sum(v.numel() for v in wrapper.parameters()),
            onnx=onnx_path.name, onnx_sha256=sha(onnx_path),
            engine=engine_path.name, engine_sha256=sha(engine_path),
            onnx_checks=onnx_errors, tensorrt_checks=errors)
        write_json(manifest_path, manifest)
        print(f"PASS depth={depth}; TensorRT max_abs={max(e['max_abs'] for e in errors):.6g}; "
              f"rmse={max(e['rmse'] for e in errors):.6g}", flush=True)
        del inspector, runtime, wrapper
        torch.cuda.empty_cache()
    ready = sorted(int(d) for d, e in manifest["engines"].items()
                   if e["status"] == "smoke_passed_pending_full_evaluation")
    print(f"READY EXITS: {ready}. Next: full 1197-image quality and isolated latency evaluation.", flush=True)


if __name__ == "__main__":
    main()
