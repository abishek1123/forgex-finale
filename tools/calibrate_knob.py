#!/usr/bin/env python3
"""Calibrate the quality/latency knob ON THIS MACHINE, then write the datasheet.

Why this exists: FLOPs do not predict latency here, and we measured that the
hard way -- e1f_gate has FEWER FLOPs than the shipped model and is SLOWER,
because LayerNorm and the gate's chunk/multiply are memory-bound, not
compute-bound. A knob labelled in FLOPs would lie to the operator. So every
setting is timed on the actual target GPU, and --budget-ms in run.py reads
these measured numbers.

Timing is INTERLEAVED across depths (round-robin, several rounds, best and
worst round dropped) so thermal drift and background load hit every setting
equally instead of penalising whichever ran while the chip was hot.

    python tools/calibrate_knob.py --data ../semicon_test_data --rounds 5
    python tools/calibrate_knob.py --no-quality          # timing only, fast

Writes models/knob_datasheet.json.  Inspect with:  python run.py --list-knob
"""
import argparse, hashlib, json, os, statistics, sys, time
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import run as R                                    # the SHIPPED entry point, not a copy

# --------------------------------------------------------------------------
# Architecture-aware model construction. Every scoring tool must build the model
# the way the SHIPPED entry point builds it -- run.build_model() reads the
# checkpoint's own config (and, failing that, sniffs the state_dict) and returns
# the right class. Constructing `Restorer(**config)` directly is how a tool ends
# up silently scoring a DIFFERENT architecture than the one being shipped, or
# crashing on an unexpected kwarg like kind='gate'.
# --------------------------------------------------------------------------
def _build_arch(ck):
    """-> (eval()-mode model on CPU, arch name). Strict load: no silent drift."""
    import os as _os, sys as _sys
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import run as _run
    _st = ck.get("state_dict", ck)
    _m, _arch = _run.build_model(ck.get("config", {}) or {}, _st)
    _m.load_state_dict(_st, strict=True)
    return _m.eval(), _arch



def count_macs(m, hw):
    macs = []
    def hook(mod, i, o):
        if isinstance(mod, nn.Conv2d):
            macs.append(o.numel() * mod.kernel_size[0] * mod.kernel_size[1]
                        * mod.in_channels // mod.groups)
        elif isinstance(mod, nn.Linear):
            macs.append(o.numel() * mod.in_features)
    hs = [mm.register_forward_hook(hook) for mm in m.modules()
          if isinstance(mm, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        m(torch.rand(1, 1, hw, hw))
    for h in hs:
        h.remove()
    return sum(macs)


def time_once(model, x, amp, iters, warm):
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        for _ in range(warm):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters / x.shape[0] * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default="", help="root with GT/ and NoisyLR/ for the quality column")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hw", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="quality on the first N images only")
    ap.add_argument("--no-quality", dest="quality", action="store_false")
    ap.add_argument("--out", default=None)
    ap.set_defaults(quality=True)
    a = ap.parse_args()

    w = R.find_weights(a.weights)
    ck = R.load_checkpoint(w)
    model, _arch = _build_arch(ck)
    print(f"  arch={_arch}  blocks={len(model.body)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    model = model.to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    full = list(model.body)
    nb = len(full)
    depths = [d for d in range(R.MIN_SAFE_DEPTH, nb + 1)]
    sha = hashlib.sha1(open(w, "rb").read()).hexdigest()[:12]
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"weights={os.path.basename(w)} sha1={sha}  nb={nb}  gpu={gpu}  batch={a.batch}\n"
          f"timing {len(depths)} settings x {a.rounds} interleaved rounds", flush=True)

    x = torch.rand(a.batch, 1, a.hw, a.hw, device=dev)
    if dev.type == "cuda":
        x = x.to(memory_format=torch.channels_last)

    ms = {d: [] for d in depths}
    for r in range(a.rounds):
        for d in depths:                              # interleaved, not grouped
            model.body = nn.Sequential(*full[:d])
            ms[d].append(time_once(model, x, amp, a.iters, 10))
        print(f"  round {r+1}/{a.rounds}", flush=True)

    gf = {}
    for d in depths:
        model.body = nn.Sequential(*full[:d])
        gf[d] = round(count_macs(model.cpu(), a.hw) * 2 / 1e9, 3)
        model.to(dev)

    q = {}
    if a.quality and a.data:
        sys.path.insert(0, os.path.join(HERE, "src"))
        from metrics import psnr, ssim
        gtd, lrd = os.path.join(a.data, "GT"), os.path.join(a.data, "NoisyLR")
        files = sorted(os.listdir(gtd))[: a.limit or None]
        print(f"scoring quality on {len(files)} images", flush=True)
        for d in depths:
            model.body = nn.Sequential(*full[:d])
            ps, ss = [], []
            with torch.no_grad():
                for f in files:
                    lr = torch.from_numpy(np.load(os.path.join(lrd, f)))[None, None].to(dev)
                    gt = torch.from_numpy(np.load(os.path.join(gtd, f)))[None, None].to(dev)
                    o = model(lr).clamp(0, 1)
                    ps.append(psnr(o, gt)); ss.append(ssim(o, gt))
            q[d] = (round(float(np.mean(ps)), 4), round(float(np.mean(ss)), 5))
            print(f"  depth {d:>2}  PSNR {q[d][0]:.4f}  SSIM {q[d][1]:.5f}", flush=True)
    model.body = nn.Sequential(*full)

    rows = []
    for d in depths:
        v = sorted(ms[d])[1:-1] or ms[d]          # drop best + worst round
        row = dict(depth=d, ms_per_img=round(statistics.median(v), 4),
                   spread_pct=round((max(v) - min(v)) / statistics.median(v) * 100, 1),
                   gflops=gf[d])
        if d in q:
            row["psnr"], row["ssim"] = q[d]
        rows.append(row)

    out = a.out or os.path.join(HERE, "models", R.DATASHEET)
    sheet = dict(gpu=gpu, weight_sha1=sha, batch=a.batch, dtype="fp16" if amp else "fp32",
                 hw=a.hw, nb_full=nb, rounds=a.rounds,
                 note="ms_per_img measured on this GPU; FLOPs do not predict it",
                 rows=rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(sheet, fh, indent=2)

    base = rows[-1]
    print(f"\n  {'depth':>6}{'ms/img':>10}{'spread':>8}{'img/s':>9}{'GFLOPs':>9}"
          f"{'PSNR':>9}{'dPSNR':>8}{'speedup':>9}")
    for r in rows:
        dp = r.get("psnr", float("nan")) - base.get("psnr", float("nan"))
        print(f"  {r['depth']:>6}{r['ms_per_img']:>10.3f}{r['spread_pct']:>7.1f}%"
              f"{1000/r['ms_per_img']:>9.1f}{r['gflops']:>9.2f}"
              f"{r.get('psnr', float('nan')):>9.4f}{dp:>+8.3f}"
              f"{base['ms_per_img']/r['ms_per_img']:>8.2f}x")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
