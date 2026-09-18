#!/usr/bin/env python3
"""OFFLINE: export ForgeX/e1f_gate to ONNX and build WIDE, PORTABLE TensorRT engines.

One engine per KNOB DEPTH. The knob truncates the residual trunk
(`model.body = model.body[:depth]`), which changes the graph -- so a depth
cannot be selected at runtime inside a single plan. Five plans, one per
published depth, is the honest implementation: run_fast.py picks the plan that
matches the requested depth and falls back to PyTorch when none does.

WIDE:     one optimisation profile spanning batch [bmin..bmax] and square or
          rectangular H,W in [hmin..hmax]. Inputs outside it fall back.
PORTABLE: --portable sets HardwareCompatibilityLevel.AMPERE_PLUS so the plan
          deserialises on any sm_80+ GPU instead of only the build GPU.

PRECISION: TensorRT >= 10.7 networks are STRONGLY TYPED -- BuilderFlag.FP16 no
longer exists and the engine's arithmetic comes from the ONNX dtypes. We export
FP32 and let TRT use TF32 tensor cores (on by default), which keeps the bicubic
skip + residual addition exactly FP32 as the spec requires. Where an older TRT
still exposes BuilderFlag.FP16 it is used, with the Resize and the final Add
pinned back to FP32.

    python tools/trt_native.py export --depth 16
    python tools/trt_native.py build  --depth 16 --portable
    python tools/trt_native.py all --depths 4,8,12,15,16 --portable
"""
import argparse, glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ENG = os.path.join(HERE, "models", "engines")


def _sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(weights):
    import torch.nn as nn
    import run as R
    path = R.find_weights(weights)
    ck = R.load_checkpoint(path)
    state = ck.get("state_dict", ck)
    model, arch = R.build_model(ck.get("config", {}) or {}, state)
    model.load_state_dict(state, strict=True)
    return model.eval(), arch, path, len(model.body)


def _tag(arch, depth):
    return "%s_d%d" % ("gate" if "gate" in arch else "forgex", depth)


# ---------------------------------------------------------------- export
def cmd_export(a):
    import torch, torch.nn as nn, numpy as np
    model, arch, wpath, nb_full = _load(a.weights)
    depth = nb_full if a.depth in (0, None) or a.depth >= nb_full else a.depth
    if depth < nb_full:
        model.body = nn.Sequential(*list(model.body)[:depth])
    model = model.to(a.device)
    os.makedirs(ENG, exist_ok=True)
    tag = _tag(arch, depth)
    out = os.path.join(ENG, tag + ".onnx")
    dummy = torch.rand(a.opt, 1, a.hw, a.hw, device=a.device)
    kw = dict(input_names=["lr"], output_names=["sr"],
              # H and W are dynamic too -- a batch-only dynamic axis bakes 128 in
              # and TensorRT then rejects any profile wider than 128.
              dynamic_axes={"lr": {0: "batch", 2: "height", 3: "width"},
                            "sr": {0: "batch", 2: "height2", 3: "width2"}},
              opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        # torch >= 2.9 defaults to the dynamo exporter, which needs onnxscript
        # and reinterprets dynamic_axes. Pin the TorchScript exporter where the
        # argument exists so the graph is the same on every pod image.
        try:
            torch.onnx.export(model, dummy, out, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, dummy, out, **kw)
        ref = model(dummy).float().cpu().numpy()
    np.save(os.path.join(ENG, tag + ".ref_in.npy"), dummy.cpu().numpy())
    np.save(os.path.join(ENG, tag + ".ref_out.npy"), ref)
    json.dump(dict(arch=arch, depth=depth, nb_full=nb_full,
                   weights=os.path.basename(wpath), weights_sha1=_sha1(wpath),
                   params=sum(p.numel() for p in model.parameters())),
              open(os.path.join(ENG, tag + ".export.json"), "w"), indent=2)
    print("  exported %-14s %6.2f MB   arch=%s depth=%d/%d params=%d"
          % (tag + ".onnx", os.path.getsize(out) / 1e6, arch, depth, nb_full,
             sum(p.numel() for p in model.parameters())))
    return tag


# ---------------------------------------------------------------- build
def cmd_build(a, tag=None):
    import tensorrt as trt
    if tag is None:
        cands = sorted(glob.glob(os.path.join(ENG, "*_d%d.onnx" % a.depth)))
        if not cands:
            sys.exit("no ONNX for depth %d -- run `export` first" % a.depth)
        tag = os.path.basename(cands[0])[:-5]
    onnx = os.path.join(ENG, tag + ".onnx")
    LOG = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(LOG)

    # TRT >= 10.7 removed EXPLICIT_BATCH (it is always explicit now).
    flags = 0
    f = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if f is not None:
        flags |= 1 << int(f)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, LOG)
    with open(onnx, "rb") as fh:
        if not parser.parse(fh.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            sys.exit("parse failed")

    cfg = builder.create_builder_config()
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace << 30)
    except AttributeError:
        cfg.max_workspace_size = a.workspace << 30

    strongly_typed = not hasattr(trt.BuilderFlag, "FP16")
    precision = "fp32(tf32)"
    if a.precision == "fp16" and hasattr(trt.BuilderFlag, "FP16"):
        cfg.set_flag(trt.BuilderFlag.FP16)
        pinned = 0
        for i in range(net.num_layers):
            l = net.get_layer(i)
            if l.type == trt.LayerType.RESIZE or \
               (l.type == trt.LayerType.ELEMENTWISE and i >= net.num_layers - 3):
                try:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)
                    pinned += 1
                except Exception:
                    pass
        if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        precision = "fp16 (%d layers pinned fp32)" % pinned

    portable = False
    if a.portable:
        hc = getattr(trt, "HardwareCompatibilityLevel", None)
        if hc is not None and hasattr(hc, "AMPERE_PLUS"):
            cfg.hardware_compatibility_level = hc.AMPERE_PLUS
            portable = True
        else:
            print("  WARNING: this TRT has no HardwareCompatibilityLevel.AMPERE_PLUS; "
                  "engine will be locked to the build GPU")

    p = builder.create_optimization_profile()
    p.set_shape("lr", (a.bmin, 1, a.hmin, a.hmin),
                      (a.bopt, 1, a.hopt, a.hopt),
                      (a.bmax, 1, a.hmax, a.hmax))
    cfg.add_optimization_profile(p)

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        sys.exit("engine build returned None")
    dt = time.perf_counter() - t0
    out = os.path.join(ENG, tag + ".plan")
    with open(out, "wb") as fh:
        fh.write(plan)

    exp = json.load(open(os.path.join(ENG, tag + ".export.json")))
    import torch
    props = torch.cuda.get_device_properties(0)
    meta = dict(exp)
    meta.update(precision=precision, strongly_typed=strongly_typed,
                hmin=a.hmin, hopt=a.hopt, hmax=a.hmax,
                bmin=a.bmin, bopt=a.bopt, bmax=a.bmax,
                trt=trt.__version__, built_on=torch.cuda.get_device_name(0),
                sm="%d%d" % (props.major, props.minor), portable=portable,
                build_seconds=round(dt, 1), bytes=os.path.getsize(out),
                full_graph=True, plan=os.path.basename(out))
    json.dump(meta, open(out + ".json", "w"), indent=2)
    print("  built  %-14s %6.2f MB  in %5.1f s  |  %s  |  batch %d-%d  hw %d-%d  |  %s"
          % (tag + ".plan", os.path.getsize(out) / 1e6, dt, precision,
             a.bmin, a.bmax, a.hmin, a.hmax,
             "PORTABLE sm_80+" if portable else "sm_%s ONLY" % meta["sm"]))
    return tag


# ---------------------------------------------------------------- verify
def cmd_verify(a, tag=None):
    """Deserialise each plan and compare against the exported torch reference."""
    import numpy as np, tensorrt as trt
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart

    def chk(r):
        e = r[0] if isinstance(r, tuple) else r
        if int(e) != 0:
            raise RuntimeError("CUDA error %s" % e)
        return r[1] if isinstance(r, tuple) and len(r) > 1 else None

    tags = [tag] if tag else sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ENG, "*.plan")))
    chk(cudart.cudaFree(0))
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    ok = True
    for t in tags:
        x = np.ascontiguousarray(np.load(os.path.join(ENG, t + ".ref_in.npy")))
        ref = np.load(os.path.join(ENG, t + ".ref_out.npy"))
        eng = rt.deserialize_cuda_engine(open(os.path.join(ENG, t + ".plan"), "rb").read())
        if eng is None:
            print("  %-14s DESERIALISE FAILED" % t); ok = False; continue
        ctx = eng.create_execution_context()
        y = np.empty_like(ref)
        di = chk(cudart.cudaMalloc(x.nbytes)); do = chk(cudart.cudaMalloc(y.nbytes))
        s = chk(cudart.cudaStreamCreate())
        chk(cudart.cudaMemcpyAsync(di, x.ctypes.data, x.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
        ctx.set_input_shape("lr", x.shape)
        ctx.set_tensor_address("lr", int(di)); ctx.set_tensor_address("sr", int(do))
        ctx.execute_async_v3(s)
        chk(cudart.cudaMemcpyAsync(y.ctypes.data, do, y.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, s))
        chk(cudart.cudaStreamSynchronize(s))
        d = float(np.abs(y - ref).max())
        print("  %-14s max|TRT-torch| = %.3e   %s" % (t, d, "OK" if d < 2e-3 else "HIGH"))
        ok &= d < 2e-3
        chk(cudart.cudaFree(di)); chk(cudart.cudaFree(do))
    return ok


# ---------------------------------------------------------------- index
def write_index():
    rows = []
    for j in sorted(glob.glob(os.path.join(ENG, "*.plan.json"))):
        rows.append(json.load(open(j)))
    rows.sort(key=lambda r: r["depth"])
    idx = dict(arch=rows[0]["arch"] if rows else None,
               nb_full=rows[0]["nb_full"] if rows else None,
               weights_sha1=rows[0]["weights_sha1"] if rows else None,
               engines=rows)
    json.dump(idx, open(os.path.join(ENG, "index.json"), "w"), indent=2)
    print("\n  index.json: %d engine(s) at depths %s"
          % (len(rows), [r["depth"] for r in rows]))


def cmd_all(a):
    depths = [int(x) for x in a.depths.split(",") if x.strip()]
    for d in depths:
        a.depth = d
        tag = cmd_export(a)
        cmd_build(a, tag)
    write_index()
    print("\n  verifying against torch references:")
    cmd_verify(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="c", required=True)

    def shared(q):
        q.add_argument("--weights", default=None)
        q.add_argument("--depth", type=int, default=0)
        q.add_argument("--hw", type=int, default=128, help="export dummy size")
        q.add_argument("--opt", type=int, default=32, help="export dummy batch")
        q.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
        q.add_argument("--hmin", type=int, default=32)
        q.add_argument("--hopt", type=int, default=128)
        q.add_argument("--hmax", type=int, default=512)
        q.add_argument("--bmin", type=int, default=1)
        q.add_argument("--bopt", type=int, default=32)
        q.add_argument("--bmax", type=int, default=64)
        q.add_argument("--workspace", type=int, default=8)
        q.add_argument("--portable", action="store_true")
        q.add_argument("--device", default="cuda")
        return q

    shared(sub.add_parser("export")).set_defaults(fn=cmd_export)
    shared(sub.add_parser("build")).set_defaults(fn=cmd_build)
    shared(sub.add_parser("verify")).set_defaults(fn=lambda a: cmd_verify(a))
    q = shared(sub.add_parser("all")); q.add_argument("--depths", default="4,8,12,15,16")
    q.set_defaults(fn=cmd_all)
    a = ap.parse_args(); a.fn(a)
