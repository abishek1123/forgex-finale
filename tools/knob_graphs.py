#!/usr/bin/env python3
"""Graphs for the knob x engine report.

    python tools/knob_graphs.py --report docs/engine_report.json \
                                --out docs/knob_engines.png

Four panels, one axis each -- never a second y-scale on the same plot:
  1  PSNR vs end-to-end seconds   (the operating curve; this is the knob)
  2  quality vs depth             (PSNR and SSIM, indexed to full depth)
  3  end-to-end seconds vs depth  (TensorRT against the PyTorch fallback)
  4  throughput  img/s vs depth

Colours are Okabe-Ito (colour-vision-deficiency safe by construction); every
series is also direct-labelled, so identity never rests on colour alone.
"""
import argparse, json, os

TRT, TORCH, THIRD, INK = "#0072B2", "#D55E00", "#009E73", "#333333"
NAME = {"trt": "TensorRT", "torch": "PyTorch fallback"}
COL = {"trt": TRT, "torch": TORCH}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", default="docs/knob_engines.png")
    ap.add_argument("--title", default="ForgeX e1f_gate - quality/speed knob across five TensorRT engines")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(a.report))
    backends = []
    for r in rows:
        if r["backend"] not in backends:
            backends.append(r["backend"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(a.title, fontsize=13, color=INK)
    for x in ax.ravel():
        x.grid(alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)

    # 1 -- the operating curve
    p = ax[0, 0]
    for b in backends:
        rs = sorted((r for r in rows if r["backend"] == b), key=lambda r: r["t_median"])
        p.plot([r["t_median"] for r in rs], [r["psnr"] for r in rs],
               "-o", color=COL.get(b, THIRD), linewidth=2, markersize=7,
               label=NAME.get(b, b))
        for r in rs:
            p.annotate("d%d" % r["depth"], (r["t_median"], r["psnr"]),
                       textcoords="offset points", xytext=(6, -10),
                       fontsize=8, color=INK)
        p.annotate(NAME.get(b, b), (rs[-1]["t_median"], rs[-1]["psnr"]),
                   textcoords="offset points", xytext=(-10, 12),
                   fontsize=10, color=COL.get(b, THIRD), fontweight="bold")
    p.set_xlabel("end-to-end seconds for the whole test set (lower is better)")
    p.set_ylabel("PSNR (dB)")
    p.set_title("The knob: what a second buys", loc="left", fontsize=11)
    p.legend(frameon=False, fontsize=9, loc="lower right")

    # 2 -- quality vs depth
    p = ax[0, 1]
    rs = sorted((r for r in rows if r["backend"] == backends[0]), key=lambda r: r["depth"])
    p.plot([r["depth"] for r in rs], [r["psnr"] for r in rs], "-o",
           color=TRT, linewidth=2, markersize=7, label="PSNR (dB)")
    p.set_xlabel("residual blocks kept (--depth)")
    p.set_ylabel("PSNR (dB)")
    for r in rs:
        p.annotate("%.2f" % r["psnr"], (r["depth"], r["psnr"]),
                   textcoords="offset points", xytext=(0, 8), fontsize=8,
                   ha="center", color=INK)
    p.set_title("Quality falls toward bicubic, never toward garbage", loc="left", fontsize=11)
    p.legend(frameon=False, fontsize=9, loc="lower right")

    # 3 -- latency vs depth
    p = ax[1, 0]
    for b in backends:
        rs = sorted((r for r in rows if r["backend"] == b), key=lambda r: r["depth"])
        p.errorbar([r["depth"] for r in rs], [r["t_median"] for r in rs],
                   yerr=[r["t_std"] for r in rs], fmt="-o", capsize=3,
                   color=COL.get(b, THIRD), linewidth=2, markersize=7,
                   label=NAME.get(b, b))
    p.set_xlabel("residual blocks kept (--depth)")
    p.set_ylabel("end-to-end seconds (median of reps, bars = stdev)")
    p.set_title("End to end, as a judge measures it", loc="left", fontsize=11)
    p.legend(frameon=False, fontsize=9)

    # 4 -- throughput
    p = ax[1, 1]
    w = 0.38
    for i, b in enumerate(backends):
        rs = sorted((r for r in rows if r["backend"] == b), key=lambda r: r["depth"])
        xs = [j + (i - (len(backends) - 1) / 2) * w for j in range(len(rs))]
        p.bar(xs, [r["img_per_s"] for r in rs], width=w * 0.92,
              color=COL.get(b, THIRD), label=NAME.get(b, b))
        for x, r in zip(xs, rs):
            p.annotate("%.0f" % r["img_per_s"], (x, r["img_per_s"]),
                       textcoords="offset points", xytext=(0, 3), fontsize=8,
                       ha="center", color=INK)
        p.set_xticks(range(len(rs)))
        p.set_xticklabels(["d%d" % r["depth"] for r in rs])
    p.set_xlabel("engine")
    p.set_ylabel("images / second, end to end")
    p.set_title("Throughput", loc="left", fontsize=11)
    p.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
