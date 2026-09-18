#!/usr/bin/env python3
"""How far down can the model shrink before quality moves?

    python tools\capacity_sweep.py --data <train-root> --p-real 1.0

Going UP is measured and flat (2.7x params: +0.026 dB; 1.8x receptive field:
+0.008 dB, both inside a 0.010 dB noise floor). Going DOWN has never been
tested, and those are different questions -- the curve can be flat above the
current size and fall off a cliff below it.

Trains seven configurations spanning 78x of parameter count at a matched
budget, then scores them all on the same held-out split AND times a forward
pass for each, because the reason to shrink is latency, not elegance.

Runs SMALLEST FIRST. The small models are the cheap ones and the interesting
ones -- if the night is cut short, we still have the bottom of the curve,
which is where the knee lives. The 1.37M reference trains last.

Screening runs are short on purpose: round-1 evidence says 40 epochs reaches
~96% of the final number, which is enough to locate the knee. Confirm the
winner at 120 after.

Skips any run already complete, so it is safe to re-run after an interruption.
"""
import argparse, csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY   = sys.executable

CONFIGS = [   # (ch, nb) -- descending parameter count; this is the REPORT order
    (64, 16),   # 1,368,705   current model, the reference point
    (32, 16),   #   343,361
    (32,  8),   #   195,393
    (24,  8),   #   110,257
    (16,  8),   #    49,313
    (16,  4),   #    30,753   TL Sanjivani's weight class
    (12,  4),   #    17,449   smaller than theirs
]
REF = (64, 16)


def rows(run):
    p = os.path.join(ROOT, "runs", run, "log.csv")
    return list(csv.DictReader(open(p))) if os.path.isfile(p) else []


def latency_ms(ch, nb, device, reps=30, batch=32, hw=128):
    """Median per-image forward time, fp16 autocast, the way run.py does it."""
    import torch
    sys.path.insert(0, ROOT)
    from src.model import Restorer
    m = Restorer(ch=ch, nb=nb).to(device).eval()
    x = torch.rand(batch, 1, hw, hw, device=device)
    amp = (device == "cuda")
    ts = []
    with torch.no_grad():
        for i in range(reps + 5):
            if device == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.amp.autocast(device_type=device, enabled=amp):
                m(x)
            if device == "cuda": torch.cuda.synchronize()
            if i >= 5: ts.append((time.perf_counter() - t) * 1000 / batch)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--p-real", type=float, default=1.0)
    ap.add_argument("--order", choices=["small-first", "large-first"], default="small-first")
    ap.add_argument("--report-only", action="store_true", help="skip training, just read the curve")
    a = ap.parse_args()

    order = list(CONFIGS) if a.order == "large-first" else list(reversed(CONFIGS))

    if not a.report_only:
        print(f"capacity sweep | {len(order)} configs | {a.epochs} epochs each | "
              f"p_real={a.p_real} | {a.order}\n", flush=True)
        for ch, nb in order:
            run = f"cap-ch{ch}nb{nb}"
            if len(rows(run)) >= a.epochs:
                print(f"  {run:<16} already complete, skipping", flush=True); continue
            print(f"  {run:<16} training...", flush=True)
            t = time.perf_counter()
            r = subprocess.run([PY, os.path.join(ROOT, "src", "train.py"),
                                "--data", a.data, "--out", os.path.join("runs", run),
                                "--epochs", str(a.epochs), "--ch", str(ch), "--nb", str(nb),
                                "--p-real", str(a.p_real), "--seed", "0", "--split-seed", "0",
                                "--amp"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"     FAILED rc={r.returncode}\n{(r.stderr or '')[-800:]}", flush=True)
            else:
                f5 = rows(run)[-5:]
                ps = sum(float(x['val_psnr']) for x in f5) / len(f5)
                print(f"     done in {(time.perf_counter()-t)/60:.1f} min   "
                      f"final-5 PSNR {ps:.4f}", flush=True)

    # ---- read the curve ----
    import statistics as st, torch
    sys.path.insert(0, ROOT)
    from src.model import Restorer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'config':<12}{'params':>11}{'x ref':>8}{'ms/img':>9}{'x ref':>8}"
          f"{'PSNR':>10}{'dPSNR':>9}{'SSIM':>9}{'dSSIM':>9}")
    print("-" * 85)
    ref_ps = ref_ss = ref_p = ref_ms = None
    out = []
    for ch, nb in CONFIGS:
        rs = rows(f"cap-ch{ch}nb{nb}")
        if not rs: continue
        p  = sum(q.numel() for q in Restorer(ch=ch, nb=nb).parameters())
        ps = st.mean([float(x["val_psnr"]) for x in rs[-5:]])
        ss = st.mean([float(x["val_ssim"]) for x in rs[-5:]])
        ms = latency_ms(ch, nb, dev)
        if (ch, nb) == REF:
            ref_ps, ref_ss, ref_p, ref_ms = ps, ss, p, ms
        out.append((ch, nb, p, ms, ps, ss, len(rs)))

    for ch, nb, p, ms, ps, ss, n in out:
        xp = f"{p/ref_p:.3f}x" if ref_p else "--"
        xm = f"{ms/ref_ms:.2f}x" if ref_ms else "--"
        dp = f"{ps-ref_ps:+.4f}" if ref_ps is not None else "--"
        ds = f"{ss-ref_ss:+.4f}" if ref_ss is not None else "--"
        print(f"ch{ch}nb{nb:<7}{p:>11,}{xp:>8}{ms:>9.2f}{xm:>8}{ps:>10.4f}{dp:>9}{ss:>9.4f}{ds:>9}")

    with open(os.path.join(ROOT, "capacity_sweep.csv"), "w", newline="") as f:
        f.write("ch,nb,params,ms_per_img,final5_psnr,final5_ssim,epochs\n")
        for r_ in out: f.write(",".join(str(v) for v in r_) + "\n")
    print(f"\ndevice={dev}   noise floor is 0.010 dB -- anything inside that is NOT a real difference")
    if ref_ps is None:
        print("NOTE: the ch64 nb16 reference has not finished yet, so the delta columns are blank.")
    print(f"written to {os.path.join(ROOT,'capacity_sweep.csv')}")


if __name__ == "__main__":
    main()
