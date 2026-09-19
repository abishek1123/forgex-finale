"""Warm batch-one timings at four sizes and five depths; no silent fallback.

GPU event latency excludes transfers. Restore wall time includes transfers,
validation and final clamp. Throughput is measured as a timed batch-one loop,
not inferred from reciprocal median latency. This is not max batched throughput.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from knob_runtime import DEPTHS, SUPPORTED_SIZES, ForgeXKnob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args()
    assert a.iterations >= 10 and a.warmup >= 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latency.csv").exists():
        raise RuntimeError("Use a new --out to preserve existing measurements")
    metadata = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                    batch=1, warmup=a.warmup, iterations=a.iterations, concurrent_requests=1,
                    pytorch_precision="FP32, TF32 disabled", tensorrt_precision="see build manifest",
                    input="seeded synthetic uniform [0,1.5] at four sizes; no quality claim",
                    timing="warm, CUDA events for GPU; perf_counter + synchronization for wall; no disk IO or loading",
                    command_line=subprocess.list2cmdline(__import__('sys').argv))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    runtime = ForgeXKnob(a.engines)
    gpu_model = runtime.load_model("cuda:0")
    rng = np.random.default_rng(0)
    rows = []
    with torch.inference_mode():
        for knob, depth in enumerate(DEPTHS, 1):
            for size in SUPPORTED_SIZES:
                arr = rng.uniform(0, 1.5, (size, size)).astype(np.float32)
                x = torch.from_numpy(arr)[None, None].cuda()
                for backend in ("pytorch", "tensorrt"):
                    if backend == "tensorrt":
                        engine = runtime.load_engine(depth)
                        fn = lambda: engine(x)
                    else:
                        fn = lambda: gpu_model.forward_depth(x, depth)
                    for _ in range(a.warmup):
                        fn()
                    torch.cuda.synchronize()
                    timings = []
                    start_event = torch.cuda.Event(enable_timing=True)
                    stop_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(a.iterations):
                        start_event.record()
                        fn()
                        stop_event.record()
                        stop_event.synchronize()
                        timings.append(start_event.elapsed_time(stop_event))
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(a.iterations):
                        fn()
                    torch.cuda.synchronize()
                    device_throughput = a.iterations / (time.perf_counter() - t0)
                    # Both application paths include identical host validation/transfers/clamp.
                    runtime.restore(arr, knob, backend=backend)
                    walls = []
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(a.iterations):
                        _, info = runtime.restore(arr, knob, backend=backend, return_info=True)
                        expected = "tensorrt" if backend == "tensorrt" else "pytorch_cuda"
                        if info["backend"] != expected:
                            raise RuntimeError("Fallback detected; will not label fallback as a GPU benchmark")
                        walls.append(info["wall_ms"])
                    wall_throughput = a.iterations / (time.perf_counter() - t0)
                    row = dict(depth=depth, size=size, backend=backend, batch=1,
                               gpu_median_ms=float(np.median(timings)), gpu_p95_ms=float(np.percentile(timings, 95)),
                               app_median_ms=float(np.median(walls)), app_p95_ms=float(np.percentile(walls, 95)),
                               device_loop_images_per_second=device_throughput,
                               app_loop_images_per_second=wall_throughput)
                    rows.append(row)
                    with (out / "latency.csv").open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=rows[0].keys())
                        w.writeheader()
                        w.writerows(rows)
                    print(json.dumps(row), flush=True)
            # Do not keep all five TensorRT contexts allocated during benchmarking.
            runtime.engines.clear()
            engine = None
            fn = None
            gc.collect()
            torch.cuda.empty_cache()
    print("40 benchmark rows complete. These are single-GPU, batch-one results only.", flush=True)


if __name__ == "__main__":
    main()
