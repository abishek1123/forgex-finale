#!/usr/bin/env python3
"""The A/B evaluation, exactly as the accept criteria specify.

    python tools/eval_ab.py \
        --ckpt A=runs/A_control/last.pt --ckpt B=runs/B_lap005/last.pt \
        --ckpt shipped=models/model.pt \
        --test ../semicon_test_data --holdout ../semicon_train_data_excluded \
        --out docs/ab_results

For every checkpoint, on BOTH image sets:
  PSNR / SSIM / LPIPS, per image and mean
  HF energy ratio (output/GT above the cut), HF correlation, gradient ratio
  count of images with HF energy ratio > 1.0      <- the fabrication check
  count of images worse than bicubic on SSIM      <- the smoothing failure
  quartiles by GT f90 (texture) and by LR noise severity
  params, FLOPs, peak VRAM, latency at batch 1/32/64 with p50 and p95
Then paired bootstrap CIs for every pairwise delta.

The holdout is the organisers' own 80/20 split -- distribution-matched, NOT OOD.
"""
import argparse, csv, json, os, statistics, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

_R = {}
def _radial(shape):
    if shape not in _R:
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        _R[shape] = np.hypot(y - h / 2, x - w / 2) / (min(h, w) / 2)
    return _R[shape]

def hf_energy(a, cut=0.5):
    F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2
    return float(F[_radial(a.shape) > cut].sum())

def highpass(a, cut=0.5):
    F = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    F[_radial(a.shape) <= cut] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))

def hf_corr(p, g, cut=0.5):
    a, b = highpass(p, cut).ravel(), highpass(g, cut).ravel()
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12: return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

def grad_energy(a):
    return float(np.sqrt(np.diff(a, axis=0)[:, :-1] ** 2 +
                         np.diff(a, axis=1)[:-1, :] ** 2).mean())

def f90(img):
    F = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
    r = _radial(F.shape).ravel(); o = np.argsort(r)
    c = np.cumsum(F.ravel()[o])
    return 0.0 if c[-1] <= 0 else float(r[o][np.searchsorted(c, 0.9 * c[-1])])

def box_down(x, f=2):
    H, W = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def build(path):
    """Shipped Restorer by default; any registry architecture when an arch.txt
    sits beside the checkpoint (that is how the E1 runs record their variant)."""
    import torch, run as R
    full = os.path.join(HERE, path) if not os.path.isabs(path) else path
    ck = R.load_checkpoint(full)
    cfg = ck.get("config", {}) or {}
    at = os.path.join(os.path.dirname(full), "arch.txt")
    arch = open(at).read().strip() if os.path.isfile(at) else ""
    if arch and arch != "forgex":
        for cand in (os.path.join(HERE, "..", "kla2"), os.path.join(HERE, "arch_ext")):
            if os.path.isdir(os.path.join(cand, "arch")):
                sys.path.insert(0, os.path.abspath(cand)); break
        from arch.registry import ARCHS
        m = ARCHS[arch](cfg.get("ch"), cfg.get("nb")).eval()
        print(f"    [{os.path.basename(os.path.dirname(full))}] arch={arch} "
              f"ch={cfg.get('ch')} nb={cfg.get('nb')}")
    else:
        m = R.Restorer(**{k: cfg[k] for k in ("ch","nb","scale","res_scale") if k in cfg}).eval()
    m.load_state_dict(ck.get("state_dict", ck))
    return m.cuda().to(memory_format=torch.channels_last), ck.get("epoch", -1)


def infer_all(m, LR, batch=32):
    import torch
    outs = []
    for i in range(0, len(LR), batch):
        t = torch.from_numpy(LR[i:i + batch])[:, None].cuda().contiguous(
            memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            o = m(t).float()
        o = torch.nan_to_num(o, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        outs.append(o[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(outs, 0)


def bicubic_all(LR):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(LR)[:, None]
    return F.interpolate(t, scale_factor=2, mode="bicubic",
                         align_corners=False).clamp(0, 1)[:, 0].numpy()


def score_set(m, root, tag, nmax=0):
    import torch
    from metrics import psnr, ssim, lpips as _lp
    gtd, lrd = os.path.join(root, "GT"), os.path.join(root, "NoisyLR")
    files = sorted(os.listdir(gtd))[: nmax or None]
    GT = np.stack([np.load(os.path.join(gtd, f)) for f in files]).astype(np.float32)
    LR = np.stack([np.load(os.path.join(lrd, f)) for f in files]).astype(np.float32)
    OUT = infer_all(m, LR)
    BIC = bicubic_all(LR)
    rows = []
    for i, f in enumerate(files):
        o, g, b, lr = OUT[i], GT[i], BIC[i], LR[i]
        eg = hf_energy(g)
        rows.append(dict(
            id=f[:-4], tag=tag,
            psnr=psnr(torch.from_numpy(o)[None, None], torch.from_numpy(g)[None, None]),
            ssim=ssim(torch.from_numpy(o)[None, None], torch.from_numpy(g)[None, None]),
            bic_psnr=psnr(torch.from_numpy(b)[None, None], torch.from_numpy(g)[None, None]),
            bic_ssim=ssim(torch.from_numpy(b)[None, None], torch.from_numpy(g)[None, None]),
            hf_ratio=(hf_energy(o) / eg) if eg > 0 else 0.0,
            hf_corr=hf_corr(o, g),
            grad_ratio=grad_energy(o) / max(grad_energy(g), 1e-12),
            f90=f90(g), noise=float((lr - box_down(g, 2)).var())))
    try:
        for i in range(len(files)):
            rows[i]["lpips"] = _lp(torch.from_numpy(OUT[i])[None, None],
                                   torch.from_numpy(GT[i])[None, None])
    except Exception:
        for r in rows: r["lpips"] = float("nan")
    return rows


def latency(m, batches=(1, 32, 64), hw=128, iters=50, warm=10):
    import torch
    out = {}
    for b in batches:
        x = torch.rand(b, 1, hw, hw, device="cuda").contiguous(
            memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            for _ in range(warm): m(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); m(x); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) / b)
        ts.sort()
        out[b] = dict(p50=round(statistics.median(ts), 4),
                      p95=round(ts[int(0.95 * len(ts)) - 1], 4),
                      img_s=round(1000 / statistics.median(ts), 1),
                      vram_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3))
    return out


def quartiles(rows, key, label):
    x = np.array([r[key] for r in rows]); q = np.quantile(x, [.25, .5, .75])
    edges = [-np.inf, *q, np.inf]; out = []
    for j in range(4):
        sub = [r for r, m in zip(rows, (x > edges[j]) & (x <= edges[j + 1])) if m]
        if not sub: continue
        out.append(dict(quartile=f"Q{j+1}", by=label, n=len(sub),
            lo=round(float(min(r[key] for r in sub)), 5),
            hi=round(float(max(r[key] for r in sub)), 5),
            psnr=round(float(np.mean([r["psnr"] for r in sub])), 4),
            ssim=round(float(np.mean([r["ssim"] for r in sub])), 5),
            gain_db=round(float(np.mean([r["psnr"] - r["bic_psnr"] for r in sub])), 4),
            hf_ratio=round(float(np.mean([r["hf_ratio"] for r in sub])), 4),
            worse_than_bicubic=sum(1 for r in sub if r["ssim"] < r["bic_ssim"])))
    return out


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=round(float(d.mean()), 5), ci_lo=round(float(lo), 5),
                ci_hi=round(float(hi), 5), t=round(float(d.mean() / se), 2) if se else None,
                better=int((d > 0).sum()), n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--test", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--out", default="docs/ab_results")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    outdir = os.path.join(HERE, a.out); os.makedirs(outdir, exist_ok=True)
    import torch

    per, summ = {}, []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m, ep = build(path)
        npar = sum(p.numel() for p in m.parameters())
        lat = latency(m)
        per[name] = {}
        for tag, root in (("test", a.test), ("holdout", a.holdout)):
            if not root: continue
            rows = score_set(m, root, tag, a.limit)
            per[name][tag] = rows
            s = dict(model=name, epoch=ep, params=npar, set=tag, n=len(rows),
                psnr=round(float(np.mean([r["psnr"] for r in rows])), 4),
                ssim=round(float(np.mean([r["ssim"] for r in rows])), 5),
                lpips=round(float(np.nanmean([r["lpips"] for r in rows])), 5),
                hf_ratio=round(float(np.mean([r["hf_ratio"] for r in rows])), 4),
                hf_corr=round(float(np.mean([r["hf_corr"] for r in rows])), 4),
                grad_ratio=round(float(np.mean([r["grad_ratio"] for r in rows])), 4),
                hf_over_1=sum(1 for r in rows if r["hf_ratio"] > 1.0),
                worse_than_bicubic=sum(1 for r in rows if r["ssim"] < r["bic_ssim"]),
                b1_p50=lat[1]["p50"], b1_p95=lat[1]["p95"],
                b32_p50=lat[32]["p50"], b32_p95=lat[32]["p95"],
                b64_p50=lat[64]["p50"], b64_p95=lat[64]["p95"],
                vram_b32=lat[32]["vram_gb"])
            summ.append(s)
            print(f"  {name:<10}{tag:<9} PSNR {s['psnr']:.4f}  SSIM {s['ssim']:.5f}  "
                  f"LPIPS {s['lpips']:.5f}  HF {s['hf_ratio']:.4f}  "
                  f"HFcorr {s['hf_corr']:.4f}  >1.0: {s['hf_over_1']}  "
                  f"<bicubic: {s['worse_than_bicubic']}/{s['n']}", flush=True)
            with open(os.path.join(outdir, f"per_image_{name}_{tag}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        del m; torch.cuda.empty_cache()

    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    names = [s.split("=", 1)[0] for s in a.ckpt]
    paired = []
    for tag in ("test", "holdout"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = names[i], names[j]
                if tag not in per.get(A, {}) or tag not in per.get(B, {}): continue
                ra, rb = per[A][tag], per[B][tag]
                assert [r["id"] for r in ra] == [r["id"] for r in rb]
                for metric, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1),
                                     ("hf_ratio", 1), ("hf_corr", 1)):
                    d = [rb[k][metric] - ra[k][metric] for k in range(len(ra))]
                    if np.isnan(d).any(): continue
                    st = boot(d)
                    st.update(set=tag, comparison=f"{B} - {A}", metric=metric)
                    paired.append(st)
    with open(os.path.join(outdir, "paired.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

    qs = []
    for name in per:
        for tag in per[name]:
            for k, lbl in (("f90", "GT texture f90"), ("noise", "LR noise variance")):
                for row in quartiles(per[name][tag], k, lbl):
                    row.update(model=name, set=tag); qs.append(row)
    with open(os.path.join(outdir, "quartiles.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qs[0])); w.writeheader(); w.writerows(qs)

    print(f"\n  wrote {outdir}/  (summary.csv, paired.csv, quartiles.csv, per_image_*.csv)")
    print("\n  PAIRED DELTAS (95% bootstrap CI, positive = second model better "
          "except LPIPS)")
    for p in paired:
        if p["metric"] in ("psnr", "ssim", "lpips"):
            print(f"    {p['set']:<9}{p['comparison']:<22}{p['metric']:<10}"
                  f"{p['mean']:+.5f}  CI [{p['ci_lo']:+.5f},{p['ci_hi']:+.5f}]  "
                  f"t={p['t']}  better {p['better']}/{p['n']}")


if __name__ == "__main__":
    main()
