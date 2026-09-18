#!/usr/bin/env python3
"""The optimisation ladder. ONE protocol, every configuration, one CSV.

    python tools/optimise.py --data ../semicon_test_data --out docs/optimise.csv

Every row reports, on the SAME fixed image list:
  warm model-only latency (CUDA events, interleaved across configs so thermal
  drift hits every row equally), peak VRAM, PSNR/SSIM/LPIPS, and max|output -
  O0 output| so a quality change is never confused with a speed change.

Cold fresh-process end-to-end is NOT measured here -- it is measured by
tools/timing_sweep.py on the real run.py, because that is the only thing that
can honestly report startup cost.

Stages, each independently skippable:
  O0  eager, fp32 weights + fp16 autocast, channels_last      (current shipping)
  O1  + cudnn.benchmark=True
  O2  fp16 weights (model.half()), no autocast
  O3  bf16 weights
  O4  torch.compile               (compile cost reported separately)
  O5  batch sweep on the winner
The FP32 bicubic + residual addition is preserved in every stage: it lives in
forward() after the trunk, and .half() does not touch `x.float()`.
"""
import argparse, csv, gc, json, os, statistics, sys, time
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import run as R
sys.path.insert(0, os.path.join(HERE, 'tools'))
from validate_out import contract as _contract


def load_model(dtype=None, device="cuda"):
    ck = R.load_checkpoint(R.find_weights(None))
    m = R.Restorer(**(ck.get("config", {}) or {})).eval()
    m.load_state_dict(ck.get("state_dict", ck))
    m = m.to(device)
    if device == "cuda":
        m = m.to(memory_format=torch.channels_last)
    if dtype is not None:
        m = m.to(dtype)
    return m


def infer(m, arr, dev, wdtype, autocast, batch):
    """arr: (N,H,W) float32 -> (N,2H,2W) float32, always clamped, always fp32 out."""
    outs = []
    for i in range(0, len(arr), batch):
        t = torch.from_numpy(arr[i:i + batch])[:, None].to(dev)
        if dev == "cuda":
            t = t.contiguous(memory_format=torch.channels_last)
        if wdtype is not None:
            t = t.to(wdtype)
        with torch.inference_mode():
            if autocast:
                with torch.autocast("cuda", enabled=True):
                    o = m(t).float()
            else:
                o = m(t).float()
        o = torch.nan_to_num(o, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        outs.append(o[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(outs, 0)


def bench(m, dev, wdtype, autocast, batch, hw=128, iters=30, warm=10):
    x = torch.rand(batch, 1, hw, hw, device=dev)
    if dev == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    if wdtype is not None:
        x = x.to(wdtype)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warm):
            m(x) if not autocast else _ac(m, x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            m(x) if not autocast else _ac(m, x)
        e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters / batch
    return ms, torch.cuda.max_memory_allocated() / 1024**3


def _ac(m, x):
    with torch.autocast("cuda", enabled=True):
        return m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with GT/ and NoisyLR/")
    ap.add_argument("--out", default="docs/optimise.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved timing rounds")
    ap.add_argument("--batches", default="1,8,16,32,64,128,256")
    ap.add_argument("--skip", default="", help="comma list of stage ids to skip")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda"
    assert torch.cuda.is_available(), "this ladder needs a GPU"
    sys.path.insert(0, os.path.join(HERE, "src"))
    from metrics import psnr, ssim, lpips as _lp

    gtd, lrd = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
    files = sorted(os.listdir(gtd))[: a.limit or None]
    LR = np.stack([np.load(os.path.join(lrd, f)) for f in files]).astype(np.float32)
    GT = np.stack([np.load(os.path.join(gtd, f)) for f in files]).astype(np.float32)
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.total_memory/1024**3:.1f} GB  "
          f"torch {torch.__version__}  cuDNN {torch.backends.cudnn.version()}")
    print(f"{len(files)} images  batch {a.batch}\n")

    skip = set(a.skip.split(",")) if a.skip else set()
    ref = None
    rows = []

    def quality(out):
        P = [psnr(torch.from_numpy(out[i])[None, None], torch.from_numpy(GT[i])[None, None])
             for i in range(len(out))]
        S = [ssim(torch.from_numpy(out[i])[None, None], torch.from_numpy(GT[i])[None, None])
             for i in range(len(out))]
        try:
            L = [_lp(torch.from_numpy(out[i])[None, None], torch.from_numpy(GT[i])[None, None])
                 for i in range(len(out))]
            L = float(np.nanmean(L))
        except Exception:
            L = float("nan")
        return float(np.mean(P)), float(np.mean(S)), L

    STAGES = [
        ("O0", "eager fp32w + fp16 autocast", dict(wdtype=None, autocast=True,  bench_cudnn=False)),
        ("O1", "+ cudnn.benchmark",           dict(wdtype=None, autocast=True,  bench_cudnn=True)),
        ("O2", "fp16 weights",                dict(wdtype=torch.float16, autocast=False, bench_cudnn=True)),
        ("O3", "bf16 weights",                dict(wdtype=torch.bfloat16, autocast=False, bench_cudnn=True)),
    ]
    for sid, name, cfg in STAGES:
        if sid in skip:
            continue
        torch.backends.cudnn.benchmark = cfg["bench_cudnn"]
        m = load_model(cfg["wdtype"], dev)
        t0 = time.perf_counter()
        ms, vram = bench(m, dev, cfg["wdtype"], cfg["autocast"], a.batch)
        out = infer(m, LR, dev, cfg["wdtype"], cfg["autocast"], a.batch)
        P, S, L = quality(out)
        if ref is None:
            ref = out
        md = float(np.abs(out - ref).max())
        ok, fails = _contract(out)
        if not ok:
            print(f"     !! CONTRACT FAILED: {fails}")
        rows.append(dict(stage=sid, config=name, backend="eager", contract="PASS" if ok else "FAIL",
                         precision=("fp32w+fp16ac" if cfg["wdtype"] is None
                                    else str(cfg["wdtype"]).replace("torch.", "") + "w"),
                         cudnn_benchmark=cfg["bench_cudnn"], batch=a.batch,
                         ms_per_img=round(ms, 4), img_per_s=round(1000 / ms, 1),
                         vram_gb=round(vram, 3), psnr=round(P, 4), ssim=round(S, 5),
                         lpips=round(L, 5), max_diff_vs_O0=round(md, 6),
                         setup_s=round(time.perf_counter() - t0, 1)))
        print(f"  {sid} {name:<30}{ms:>8.4f} ms  {1000/ms:>7.1f} img/s  "
              f"{vram:>6.3f} GB  {P:.4f}/{S:.5f}/{L:.5f}  maxdiff {md:.2e}", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()

    # ---- O4 torch.compile -------------------------------------------------
    if "O4" not in skip and hasattr(torch, "compile"):
        try:
            torch.backends.cudnn.benchmark = True
            m = load_model(None, dev)
            t0 = time.perf_counter()
            cm = torch.compile(m)
            x = torch.rand(a.batch, 1, 128, 128, device=dev).contiguous(
                memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast("cuda", enabled=True):
                cm(x)
            torch.cuda.synchronize()
            comp = time.perf_counter() - t0
            ms, vram = bench(cm, dev, None, True, a.batch)
            out = infer(cm, LR, dev, None, True, a.batch)
            P, S, L = quality(out)
            md = float(np.abs(out - ref).max())
            ok4, f4 = _contract(out)
            if not ok4: print(f"     !! CONTRACT FAILED: {f4}")
            rows.append(dict(stage="O4", config="torch.compile", backend="inductor",
                             contract="PASS" if ok4 else "FAIL",
                             precision="fp32w+fp16ac", cudnn_benchmark=True, batch=a.batch,
                             ms_per_img=round(ms, 4), img_per_s=round(1000 / ms, 1),
                             vram_gb=round(vram, 3), psnr=round(P, 4), ssim=round(S, 5),
                             lpips=round(L, 5), max_diff_vs_O0=round(md, 6),
                             setup_s=round(comp, 1)))
            print(f"  O4 torch.compile                  {ms:>8.4f} ms  {1000/ms:>7.1f} img/s  "
                  f"{vram:>6.3f} GB  {P:.4f}/{S:.5f}/{L:.5f}  maxdiff {md:.2e}")
            print(f"     COMPILE COST {comp:.1f} s  <- paid on every fresh process")
            del m, cm; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  O4 torch.compile FAILED: {type(e).__name__}: {str(e)[:140]}")

    # ---- O5 batch sweep on the best eager config --------------------------
    if "O5" not in skip and rows:
        eager = [r for r in rows if r["backend"] == "eager"]
        best = min(eager, key=lambda r: r["ms_per_img"])
        wd = {"fp32w+fp16ac": None, "float16w": torch.float16,
              "bfloat16w": torch.bfloat16}[best["precision"]]
        ac = wd is None
        torch.backends.cudnn.benchmark = True
        m = load_model(wd, dev)
        print(f"\n  O5 batch sweep on {best['stage']} ({best['config']})")
        print(f"  {'batch':>7}{'ms/img':>10}{'img/s':>10}{'VRAM GB':>10}")
        for b in [int(x) for x in a.batches.split(",")]:
            try:
                ms, vram = bench(m, dev, wd, ac, b)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {b:>7}   OOM"); torch.cuda.empty_cache(); continue
                raise
            rows.append(dict(stage="O5", config=f"{best['stage']} @ batch {b}",
                             backend="eager", contract="n/a", precision=best["precision"],
                             cudnn_benchmark=True, batch=b,
                             ms_per_img=round(ms, 4), img_per_s=round(1000 / ms, 1),
                             vram_gb=round(vram, 3), psnr=None, ssim=None, lpips=None,
                             max_diff_vs_O0=None, setup_s=None))
            print(f"  {b:>7}{ms:>10.4f}{1000/ms:>10.1f}{vram:>10.3f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    with open(op, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
