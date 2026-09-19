"""Paired quality evaluation on all five exits with separate backend accounting.

No image is resized to fit a benchmark. Natural image sizes are retained. A GT
image must have exactly twice the input's height and width. Metrics use the same
repository functions as score_multiexit.py. Runtime fallback is visible in CSV.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from knob_runtime import DEPTHS, SUPPORTED_SIZES, ForgeXKnob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--repo", default="/workspace/forgex-kla-ps01")
    p.add_argument("--expected", type=int, default=1197)
    p.add_argument("--out", required=True)
    p.add_argument("--backends", nargs="+", choices=("pytorch", "tensorrt", "auto"), default=["pytorch", "auto"])
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve() / "src"))
    from metrics import psnr, ssim, lpips
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gtroot, lrroot = Path(a.data) / "GT", Path(a.data) / "NoisyLR"
    gts = {f.relative_to(gtroot).as_posix(): f for f in gtroot.rglob("*.npy")}
    lrs = {f.relative_to(lrroot).as_posix(): f for f in lrroot.rglob("*.npy")}
    if set(gts) != set(lrs) or len(gts) != a.expected:
        raise RuntimeError(f"Require {a.expected} exact pairs. GT={len(gts)}, LR={len(lrs)}, paired={len(set(gts)&set(lrs))}")
    names = sorted(gts)
    inventory, digest = collections.Counter(), hashlib.sha256()
    for name in names:
        x = np.load(lrs[name], mmap_mode="r")
        gt = np.load(gts[name], mmap_mode="r")
        if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] not in SUPPORTED_SIZES:
            raise RuntimeError(f"Unsupported natural input dimensions {name}: {x.shape}; no silent resize")
        if gt.shape != (2*x.shape[0], 2*x.shape[1]) or not np.isfinite(x).all() or not np.isfinite(gt).all():
            raise RuntimeError(f"Bad GT/input shape or nonfinite values: {name}")
        inventory[str(x.shape)] += 1
        digest.update(name.encode())
        digest.update(hashlib.sha256(lrs[name].read_bytes()).digest())
        digest.update(hashlib.sha256(gts[name].read_bytes()).digest())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "per_image.csv").exists():
        raise RuntimeError("Output already contains evaluation results; use a new --out")
    manifest = json.loads((Path(a.engines) / "manifest.json").read_text())
    metadata = dict(data=str(Path(a.data).resolve()), pairs=len(names), sizes=dict(inventory),
                    data_sha256=digest.hexdigest(), manifest=manifest, backends=a.backends,
                    metric_psnr_ssim_lpips="same repo/src/metrics.py as original scorer",
                    metric_source_sha256=hashlib.sha256((Path(a.repo)/"src"/"metrics.py").read_bytes()).hexdigest(),
                    timing_note="Restore-call wall time excludes disk IO and metric computation; first calls may include loading. Use benchmark.py for warm latency.")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"DATASET VERIFIED: {len(names)} pairs; sizes={dict(inventory)}", flush=True)
    fields = ("image", "group", "requested_backend", "backend", "fallback", "reason", "depth", "size", "psnr", "ssim", "lpips", "wall_ms")
    groups = collections.defaultdict(list)
    with (out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for requested in a.backends:
            runtime = ForgeXKnob(a.engines)
            for knob, depth in enumerate(DEPTHS, 1):
                t0 = time.perf_counter()
                for i, name in enumerate(names, 1):
                    x = np.load(lrs[name]).astype(np.float32)
                    gt = torch.from_numpy(np.load(gts[name]).astype(np.float32))[None, None]
                    y, info = runtime.restore(x, knob, backend=requested, return_info=True)
                    pred = torch.from_numpy(y)[None, None]
                    # Compute metrics on CPU so metric GPU kernels do not overlap inference.
                    with torch.inference_mode():
                        values = dict(psnr=psnr(pred, gt), ssim=ssim(pred, gt), lpips=lpips(pred, gt, device="cpu"))
                    if not all(np.isfinite(v) for v in values.values()):
                        raise RuntimeError(f"Nonfinite metrics on {name}; check LPIPS installation")
                    row = dict(image=name, group=Path(name).parent.as_posix(), requested_backend=requested,
                               backend=info["backend"], fallback=info["fallback"], reason="; ".join(info["reasons"]),
                               depth=depth, size=x.shape[0], wall_ms=info["wall_ms"], **values)
                    writer.writerow(row)
                    groups[(requested, info["backend"], depth, x.shape[0])].append(row)
                    if i % 100 == 0:
                        f.flush()
                        print(f"{requested} depth={depth}: {i}/{len(names)}", flush=True)
                f.flush()
                print(f"DONE {requested} depth={depth}, {time.perf_counter()-t0:.1f}s", flush=True)
            del runtime
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = []
    for (requested, backend, depth, size), rows in groups.items():
        summary.append(dict(requested_backend=requested, backend=backend, depth=depth, size=size,
                            n=len(rows), fallback_count=sum(r["fallback"] for r in rows),
                            **{k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}))
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Quality evaluation complete. Compare paired backend metrics before acceptance; run benchmark separately.", flush=True)


if __name__ == "__main__":
    main()
