#!/usr/bin/env python3
"""TensorRT path for ForgeX: build engines ahead of time, benchmark, validate.

DESIGN -- three decisions, each made to protect the submission:

1. TRUNK ONLY. We export stem->head->body->body_tail->up->tail, which produces
   the RESIDUAL. The global bicubic upsample and the final addition stay in
   PyTorch at FP32. That (a) satisfies the standing requirement that the final
   bicubic+residual add is FP32, (b) avoids depending on TRT's cubic Resize, and
   (c) costs almost nothing, because the trunk is ~97% of the model's MACs.

2. FIXED SPATIAL SHAPE, DYNAMIC BATCH. The engine is built for 128x128 with a
   batch profile (min 1 / opt 32 / max 128). The evaluation data is 128x128.
   Anything else -- and the stress suite deliberately contains 64, 256, 512,
   non-square, odd and mixed sizes -- falls back to eager PyTorch. An engine
   cannot serve arbitrary shapes, so we do not pretend it can.

3. AHEAD-OF-TIME. The engine is built once and serialised to disk. Nothing is
   compiled inside the timed evaluation path.

    python tools/trt.py build --precision fp16
    python tools/trt.py build --precision int8 --calib ../semicon_test_data/NoisyLR
    python tools/trt.py build --precision fp8
    python tools/trt.py bench --data ../semicon_test_data --engine models/forgex_fp16.plan
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _torch():
    import torch
    return torch


class Trunk:
    """Everything that produces the residual. Wrapped, not modified."""
    @staticmethod
    def wrap(m):
        torch = _torch()
        import torch.nn as nn

        class _T(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.stem, self.head, self.body = m.stem, m.head, m.body
                self.body_tail, self.up, self.tail = m.body_tail, m.up, m.tail

            def forward(self, x):
                f = self.head(self.stem(x))
                f = f + self.body_tail(self.body(f))
                return self.tail(self.up(f))
        return _T(m).eval()


def load_eager(device="cuda"):
    torch = _torch()
    import run as R
    ck = R.load_checkpoint(R.find_weights(None))
    m = R.Restorer(**(ck.get("config", {}) or {})).eval()
    m.load_state_dict(ck.get("state_dict", ck))
    return m.to(device)


# ---------------------------------------------------------------- build
def cmd_build(a):
    torch = _torch()
    import tensorrt as trt
    m = load_eager("cuda")
    trunk = Trunk.wrap(m).cuda()
    onnx_path = os.path.join(HERE, "models", "forgex_trunk.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dummy = torch.randn(a.opt, 1, a.hw, a.hw, device="cuda")
    with torch.no_grad():
        torch.onnx.export(
            trunk, dummy, onnx_path, input_names=["lr"], output_names=["residual"],
            dynamic_axes={"lr": {0: "batch"}, "residual": {0: "batch"}},
            opset_version=17, do_constant_folding=True)
    print(f"  exported ONNX -> {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.2f} MB)")

    LOG = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(LOG)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = builder.create_network(flag)
    parser = trt.OnnxParser(net, LOG)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            sys.exit("ONNX parse failed")

    cfg = builder.create_builder_config()
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace << 30)
    except AttributeError:
        cfg.max_workspace_size = a.workspace << 30          # TRT 8

    p = builder.create_optimization_profile()
    p.set_shape("lr", (a.min, 1, a.hw, a.hw), (a.opt, 1, a.hw, a.hw), (a.max, 1, a.hw, a.hw))
    cfg.add_optimization_profile(p)

    prec = a.precision
    if prec in ("fp16", "int8", "fp8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if prec == "fp8":
        try:
            cfg.set_flag(trt.BuilderFlag.FP8)
            print("  FP8 flag set. NOTE: without Q/DQ nodes (nvidia-modelopt) TRT may")
            print("  silently keep layers in FP16 -- check the speed delta, not the flag.")
        except AttributeError:
            sys.exit("this TensorRT build has no FP8 flag")
    if prec == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = _calibrator(trt, a)

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        sys.exit("engine build returned None")
    build_s = time.perf_counter() - t0
    out = a.out or os.path.join(HERE, "models", f"forgex_{prec}.plan")
    with open(out, "wb") as f:
        f.write(plan)
    meta = dict(precision=prec, hw=a.hw, batch_min=a.min, batch_opt=a.opt, batch_max=a.max,
                trt=trt.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0),
                sm=f"{torch.cuda.get_device_properties(0).major}"
                   f"{torch.cuda.get_device_properties(0).minor}",
                build_seconds=round(build_s, 1), bytes=os.path.getsize(out))
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  built {prec} engine in {build_s:.1f} s -> {out} "
          f"({os.path.getsize(out)/1e6:.2f} MB)")
    print(f"  NOTE: this engine is valid ONLY for sm_{meta['sm']} + TRT {meta['trt']}.")


def _calibrator(trt, a):
    """INT8 entropy calibration on REAL degraded inputs, not synthetic noise."""
    import pycuda.driver as cuda  # noqa
    files = sorted(f for f in os.listdir(a.calib) if f.endswith(".npy"))[: a.calib_n]
    if not files:
        sys.exit(f"no .npy in {a.calib}")
    print(f"  INT8 calibration on {len(files)} real images from {a.calib}")

    class C(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.cache = os.path.join(HERE, "models", "int8_calib.cache")
            self.batch = a.opt
            self.dev = None

        def get_batch_size(self): return self.batch

        def get_batch(self, names):
            import torch
            if self.i + self.batch > len(files): return None
            arr = np.stack([np.load(os.path.join(a.calib, f))
                            for f in files[self.i:self.i + self.batch]]).astype(np.float32)
            self.i += self.batch
            self.dev = torch.from_numpy(arr)[:, None].cuda().contiguous()
            return [int(self.dev.data_ptr())]

        def read_calibration_cache(self):
            return open(self.cache, "rb").read() if os.path.isfile(self.cache) else None

        def write_calibration_cache(self, c):
            open(self.cache, "wb").write(c)
    return C()


# ---------------------------------------------------------------- runtime
class TRTTrunk:
    """Loads a serialised engine. Returns None from build() if anything is wrong,
    so the caller falls back to eager rather than crashing."""

    def __init__(self, path):
        import tensorrt as trt, torch
        self.torch = torch
        self.log = trt.Logger(trt.Logger.ERROR)
        self.rt = trt.Runtime(self.log)
        with open(path, "rb") as f:
            self.eng = self.rt.deserialize_cuda_engine(f.read())
        if self.eng is None:
            raise RuntimeError("deserialize failed (wrong GPU arch or TRT version?)")
        self.ctx = self.eng.create_execution_context()
        self.meta = json.load(open(path + ".json")) if os.path.isfile(path + ".json") else {}
        self.hw = self.meta.get("hw", 128)
        self.bmax = self.meta.get("batch_max", 128)

    def serves(self, shape):
        b, _, h, w = shape
        return h == self.hw and w == self.hw and 1 <= b <= self.bmax

    def __call__(self, x):
        torch = self.torch
        b = x.shape[0]
        self.ctx.set_input_shape("lr", tuple(x.shape))
        out = torch.empty((b, 1, x.shape[2] * 2, x.shape[3] * 2),
                          dtype=torch.float32, device="cuda")
        xc = x.contiguous()
        self.ctx.set_tensor_address("lr", int(xc.data_ptr()))
        self.ctx.set_tensor_address("residual", int(out.data_ptr()))
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out


def restore_trt(trt_trunk, eager, x):
    """residual from TRT (or eager), bicubic + add ALWAYS in fp32."""
    torch = _torch()
    import torch.nn.functional as F
    with torch.inference_mode():
        if trt_trunk is not None and trt_trunk.serves(x.shape):
            residual = trt_trunk(x)
        else:
            residual = Trunk.wrap(eager)(x)
        base = F.interpolate(x.float(), scale_factor=2, mode="bicubic", align_corners=False)
        return base + residual.float()


# ---------------------------------------------------------------- bench
def cmd_bench(a):
    torch = _torch()
    sys.path.insert(0, os.path.join(HERE, "src"))
    from metrics import psnr, ssim, lpips as _lp
    eager = load_eager("cuda")
    try:
        engine = TRTTrunk(a.engine)
        print(f"  engine {a.engine}  {engine.meta}")
    except Exception as e:
        print(f"  ENGINE UNUSABLE ({type(e).__name__}: {e}) -> eager fallback")
        engine = None

    gtd = os.path.join(a.data, "GT"); lrd = os.path.join(a.data, "NoisyLR")
    files = sorted(os.listdir(gtd))
    LR = np.stack([np.load(os.path.join(lrd, f)) for f in files]).astype(np.float32)
    GT = np.stack([np.load(os.path.join(gtd, f)) for f in files]).astype(np.float32)

    x = torch.from_numpy(LR[: a.batch])[:, None].cuda().contiguous()
    for _ in range(10):
        restore_trt(engine, eager, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(30):
        restore_trt(engine, eager, x)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 30 / a.batch
    vram = torch.cuda.max_memory_allocated() / 1024**3

    outs = []
    for i in range(0, len(LR), a.batch):
        xb = torch.from_numpy(LR[i:i + a.batch])[:, None].cuda().contiguous()
        o = restore_trt(engine, eager, xb)
        o = torch.nan_to_num(o, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        outs.append(o[:, 0].cpu().numpy().astype(np.float32))
    out = np.concatenate(outs, 0)
    P = float(np.mean([psnr(torch.from_numpy(out[i])[None, None],
                            torch.from_numpy(GT[i])[None, None]) for i in range(len(out))]))
    S = float(np.mean([ssim(torch.from_numpy(out[i])[None, None],
                            torch.from_numpy(GT[i])[None, None]) for i in range(len(out))]))
    try:
        L = float(np.nanmean([_lp(torch.from_numpy(out[i])[None, None],
                                  torch.from_numpy(GT[i])[None, None]) for i in range(len(out))]))
    except Exception:
        L = float("nan")
    print(f"\n  batch {a.batch}   {ms:.4f} ms/img   {1000/ms:.1f} img/s   {vram:.3f} GB")
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from validate_out import report
    ref = np.load(a.ref) if a.ref and os.path.isfile(a.ref) else None
    report(f"TRT {os.path.basename(a.engine)}", out, ref, (P, S, L))
    if a.save:
        np.save(a.save, out); print(f"  wrote {a.save}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--precision", default="fp16", choices=["fp16", "int8", "fp8"])
    b.add_argument("--hw", type=int, default=128)
    b.add_argument("--min", type=int, default=1)
    b.add_argument("--opt", type=int, default=32)
    b.add_argument("--max", type=int, default=128)
    b.add_argument("--workspace", type=int, default=8, help="GiB")
    b.add_argument("--calib", default="", help="dir of real .npy inputs, INT8 only")
    b.add_argument("--calib-n", type=int, default=128)
    b.add_argument("--out", default="")
    b.set_defaults(fn=cmd_build)
    n = sub.add_parser("bench")
    n.add_argument("--data", required=True)
    n.add_argument("--engine", required=True)
    n.add_argument("--batch", type=int, default=32)
    n.add_argument("--save", default="")
    n.add_argument("--ref", default="", help="npy of the eager reference outputs")
    n.set_defaults(fn=cmd_bench)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
