#!/usr/bin/env python3
"""PRIORITY 1: what is this machine, and which optimisations are even available?

Records every version the optimisation report must cite, and probes each backend
BEFORE we spend time on it. Prints a capability verdict per optimisation path.
"""
import os, platform, subprocess, sys, time

print("="*72); print("ENVIRONMENT"); print("="*72)
print(f"  python        {sys.version.split()[0]}  ({platform.machine()})")
print(f"  OS            {platform.system()} {platform.release()} {platform.version()}")

t0=time.perf_counter(); import torch; t_imp=time.perf_counter()-t0
import numpy as np
print(f"  torch         {torch.__version__}     (import took {t_imp:.3f} s)")
print(f"  numpy         {np.__version__}")
print(f"  torch CUDA    {torch.version.cuda}")
print(f"  cuDNN         {torch.backends.cudnn.version()}")
try:
    import torchvision; print(f"  torchvision   {torchvision.__version__}")
except Exception: print("  torchvision   not installed")

print(f"\n  cuda available        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print(f"  GPU                   {p.name}")
    print(f"  compute capability    sm_{p.major}{p.minor}")
    print(f"  total VRAM            {p.total_memory/1024**3:.2f} GB")
    print(f"  SMs                   {p.multi_processor_count}")
    try:
        d=subprocess.run(["nvidia-smi","--query-gpu=driver_version,power.limit,clocks.max.sm",
                          "--format=csv,noheader"],capture_output=True,text=True,timeout=10)
        print(f"  driver / power / clk  {d.stdout.strip()}")
    except Exception as e: print(f"  nvidia-smi            unavailable ({type(e).__name__})")
    print(f"  TF32 matmul allowed   {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  TF32 cudnn allowed    {torch.backends.cudnn.allow_tf32}")
    print(f"  cudnn.benchmark       {torch.backends.cudnn.benchmark}  <- run.py default")
    # bf16 / fp8 support
    print(f"  bf16 supported        {torch.cuda.is_bf16_supported()}")
    fp8 = hasattr(torch,'float8_e4m3fn')
    print(f"  torch has float8      {fp8}"
          f"{'  (sm_89+ has FP8 tensor cores; needs TRT or a kernel lib to exploit)' if fp8 else ''}")

print("\n"+"="*72); print("BACKEND AVAILABILITY"); print("="*72)
def probe(name, fn):
    try:
        ok, note = fn()
    except Exception as e:
        ok, note = False, f"{type(e).__name__}: {str(e)[:90]}"
    print(f"  {name:<22}{'AVAILABLE' if ok else 'NOT AVAILABLE':<15}{note}")
    return ok

def _triton():
    import triton; return True, f"triton {triton.__version__}"
def _compile():
    if not hasattr(torch,"compile"): return False,"torch.compile missing"
    if platform.system()=="Windows":
        try:
            import triton  # noqa
            return True,"torch.compile + triton present on Windows"
        except Exception:
            return False,("no triton -- on Windows the inductor GPU backend needs it; "
                          "torch.compile will fall back or fail")
    return True,"posix"
def _trt():
    import tensorrt as trt; return True, f"tensorrt {trt.__version__}"
def _torchtrt():
    import torch_tensorrt as t; return True, f"torch_tensorrt {t.__version__}"
def _onnx():
    import onnxruntime as o; return True, f"onnxruntime {o.__version__}  providers={o.get_available_providers()}"

has_triton=probe("triton", _triton)
has_compile=probe("torch.compile", _compile)
has_trt=probe("tensorrt", _trt)
has_ttrt=probe("torch_tensorrt", _torchtrt)
has_ort=probe("onnxruntime", _onnx)

print("\n"+"="*72); print("VERDICT -- which priorities can actually run here"); print("="*72)
V=[("P2 torch.compile", has_compile and has_triton,
    "needs triton; without it inductor cannot emit GPU kernels"),
   ("P3 TensorRT FP16/FP8", has_trt or has_ttrt,
    "FP8 additionally needs sm_89+ AND TensorRT; torch alone cannot use FP8 tensor cores"),
   ("P3 pure FP16 weights", True, "model.half() -- always available, halves weight traffic"),
   ("P3 BF16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "supported on Ampere+; on this memory-bound model expect parity with FP16"),
   ("P4 batch sweep", torch.cuda.is_available(), "limited by VRAM on this card"),
   ("P5 cudnn.benchmark", torch.cuda.is_available(), "free to test; costs autotune at startup"),
   ("P6 startup / IO", True, "always available")]
for n,ok,note in V:
    print(f"  {'RUN ' if ok else 'SKIP'}  {n:<24}{note}")
print()
