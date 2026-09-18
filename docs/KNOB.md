# The quality / latency knob

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> The Grand Finale ships **`e1f_gate-120`** — 1,346,360 params, **23.8297 PSNR / 0.62104 SSIM / 0.18126 LPIPS**, `models/model.pt` sha1 `f95377a21e86…`. Numbers below are kept as the record of how that decision was reached, not as a description of what ships.
>
> Current: [`SHIP_E1F_GATE.md`](SHIP_E1F_GATE.md)


One checkpoint. Fourteen operating points. Zero extra parameters, zero retraining.

## Why it works on this architecture

`run.py`'s `forward` is:

```python
f = self.head(self.stem(x))
f = f + self.body_tail(self.body(f))      # long skip around the whole body
residual = self.tail(self.up(f))          # tail is zero-initialised at train time
return bicubic_up(x) + residual.float()   # global skip, fp32
```

Three properties the network already had make truncating `self.body` a free knob:

1. **The body is a flat `nn.Sequential` at constant width and constant resolution.**
   Block 12's output is the same shape and kind of tensor as block 16's, so one shared
   tail can read any depth. A U-net with encoder/decoder stages could not do this.
2. **`res_scale = 0.1`.** Each block contributes `x + 0.1·branch`, so features evolve
   slowly and additively — depth 8 and depth 16 differ in refinement, not in character.
3. **Zero-init tail + global bicubic skip.** The network predicts only a *correction* to
   bicubic. A shallower run makes a *smaller* correction, so the failure mode is
   "falls back toward bicubic", never "produces garbage".

**This is a safety property, not a performance one:** turning the speed up cannot produce
a catastrophic output, only a more conservative one. Measured — even depth 0 scores
**+2.21 dB over bicubic**.

## Usage

```bash
python run.py <in> <out>                     # unchanged. full depth. bit-identical to before
python run.py <in> <out> --depth 8           # explicit setting
python run.py <in> <out> --budget-ms 3.0     # pick the best quality that fits the budget
python run.py <in> <out> --budget-ms 3.0 --prefer ssim
python run.py --list-knob                    # print the calibrated datasheet
python run.py <in> <out> --timing            # per-stage breakdown, incl. MODEL-ONLY ms/image
```

Calibrate on the target machine first — the datasheet is machine-specific:

```bash
python tools/calibrate_knob.py --data ../semicon_test_data --rounds 5
```

## Measured datasheet — RTX 4050 Laptop, batch 32, fp16, weights `e208d13d62b3`

| depth | ms/img | img/s | speedup | GFLOPs | PSNR | ΔPSNR | SSIM | |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 1.518 | 658.6 | 3.10× | 13.42 | 22.4691 | −1.163 | 0.47505 | |
| 4 | 1.762 | 567.6 | 2.67× | 15.84 | 22.5085 | −1.123 | 0.47910 | |
| 5 | 2.008 | 497.9 | 2.34× | 18.25 | 22.5809 | −1.051 | 0.48877 | |
| 6 | 2.253 | 443.9 | 2.09× | 20.67 | 22.7242 | −0.908 | 0.50423 | |
| 7 | 2.500 | 400.0 | 1.88× | 23.08 | 22.8550 | −0.777 | 0.51414 | |
| 8 | 2.744 | 364.5 | 1.72× | 25.50 | 22.9667 | −0.665 | 0.52333 | |
| 9 | 2.990 | 334.5 | 1.57× | 27.91 | 23.0903 | −0.542 | 0.53420 | |
| 10 | 3.236 | 309.0 | 1.46× | 30.33 | 23.1882 | −0.444 | 0.54251 | |
| 11 | 3.482 | 287.2 | 1.35× | 32.75 | 23.3167 | −0.315 | 0.55601 | |
| 12 | 3.726 | 268.4 | 1.26× | 35.16 | 23.3532 | −0.279 | 0.56853 | |
| 13 | 3.972 | 251.8 | 1.19× | 37.58 | 23.4064 | −0.226 | 0.57670 | |
| 14 | 4.219 | 237.0 | 1.12× | 39.99 | 23.3931 | −0.239 | 0.58433 | **DOM** |
| 15 | 4.462 | 224.1 | 1.06× | 42.41 | 23.5229 | −0.109 | 0.59257 | |
| **16** | **4.708** | **212.4** | **1.00×** | **44.83** | **23.6320** | **+0.000** | **0.60792** | **default** |

Measurement quality: 5 interleaved rounds, best and worst dropped, **spread 0.2–1.1%**.
Latency is linear in depth — fixed cost **0.782 ms**, **0.2454 ms per block**.

## The datasheet is a CEILING, not the shipping number

`tools/calibrate_knob.py` forces `cudnn.benchmark = True` and times synthetic input.
`run.py` leaves `cudnn.benchmark` at its default (`False`), a deliberate choice recorded in
`docs/REPRODUCIBILITY.md`. Measured on the real path with `--timing` at depth 8:

| | ms/image |
|---|---:|
| datasheet (synthetic, benchmark ON) | 2.7436 |
| `run.py` measured, run 1 | 3.9497 |
| `run.py` measured, run 2 | 3.3107 |

So the datasheet runs optimistic, and the real forward time varies ~19% run to run.
`--timing` prints this ratio itself so the datasheet cannot be quoted by accident.

## Measured sweep on the shipping path

`tools/timing_sweep.py`, 297 images, batch 32, fp16, **5 interleaved repetitions per depth**,
medians reported. This is the table to quote.

| depth | fwd ms/img | spread | img/s | fwd s | e2e s | fwd % of e2e | datasheet ms | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2.0906 | 3.7% | 478.3 | 0.621 | 2.829 | 22.0% | 1.5184 | 1.38× |
| 8 | 3.2527 | 4.4% | 307.4 | 0.966 | 3.177 | 30.5% | 2.7436 | 1.19× |
| 13 | 4.6042 | 2.2% | 217.2 | 1.367 | 3.605 | 37.9% | 3.9720 | 1.16× |
| **16** | **5.3062** | 2.0% | 188.5 | 1.576 | 3.762 | **41.9%** | 4.7082 | 1.13× |

```bash
python tools/timing_sweep.py --data ../semicon_test_data/NoisyLR --depths 3,8,13,16 --reps 5
```

* **Knob span in forward time: 0.955 s (2.54×).**
* **Knob span in end-to-end: 0.933 s — 24.8% of the full-depth run.**
* The forward saving passes through to end-to-end at **97.7%** (0.933 / 0.955): nothing else
  absorbs it, because the reader and writer pools already overlap the GPU.
* End-to-end throughput: **79.0 img/s** at full depth, **105.0 img/s** at depth 3.
* `import torch` measures 1.568–1.593 s at every depth — depth-independent, as it must be.
  That constancy is the check that the instrumentation is sound.
* The datasheet's optimism shrinks with depth (1.38× → 1.13×): the missing `cudnn.benchmark`
  autotune is a roughly fixed per-call cost, so it dominates more when there is less work.

## Where the time actually goes (297 images, batch 32, fp16, depth 8)

| stage | seconds | ms/image | % of end-to-end |
|---|---:|---:|---:|
| import torch + numpy | 1.626 | — | 37.9% |
| checkpoint load + model to device | 0.091 | — | 2.1% |
| host → device | 0.017 | 0.0587 | 0.4% |
| **MODEL FORWARD** | **0.983** | **3.3107** | **22.9%** |
| device → host + postprocess | 0.663 | 2.2337 | 15.4% |
| **END-TO-END** | **4.295** | **14.4600** | 100% |

Two corrections to earlier documentation, both from this measurement:

* `docs/DAY_LOG_17SEP.md` states the forward pass is **~1%** of a scored end-to-end run.
  It is **22.9% at depth 8**, and **41.9% at full depth** (measured). The knob therefore moves
  end-to-end time by a measured **24.8%** across its range — it is a real efficiency lever, not
  only a deployment control.
* The same file states `import torch` is **82%** of the run. Warm, it is **37.9%**. The 82%
  figure came from a ~17 s cold-cache run; `import torch` is a roughly fixed ~1.6 s, so its
  share falls as everything else gets faster.

**Read the `MODEL FORWARD` row, not `GPU path total`.** `device → host` is wall-clocked and
`.cpu()` blocks until queued GPU work drains, so that row partially double-counts the very
kernels the CUDA events already measured. The forward number is clean; the D2H number is an
upper bound.

## Two things the calibration found

**Depth 14 is Pareto-dominated on PSNR.** It is slower than depth 13 (4.219 vs 3.972 ms)
*and* lower (23.3931 vs 23.4064 dB). A naive "deepest setting that fits the budget" solver
would hand an operator a strictly worse row. `--budget-ms` therefore maximises the
`--prefer` metric inside the budget rather than maximising depth. Depth 14 is *not*
dominated on SSIM (0.58433 > 0.57670), which is why `--prefer ssim` still selects it.

**Depths 0–2 are non-monotone** (depth 2 scores below depth 0). `MIN_SAFE_DEPTH = 3`
refuses them and clamps with a warning.

## Safety

* Default (`--depth 0`) is full depth and is **bit-identical** to the pre-knob `run.py` —
  verified on all 297 test images, max abs diff **0.0**.
* `tools/stress_run.py`: **13/13 PASS**.
* Missing or unreadable datasheet → `--budget-ms` warns and falls back to **full depth**.
* Budget below the fastest setting → warns and uses the fastest.
* The datasheet records GPU name, weight sha1, batch and dtype, so a datasheet from the
  wrong machine or the wrong weights is visibly wrong.

## Honest limitation

Round 2 scored **end-to-end** wall clock, and in an end-to-end run the forward pass is
**~1% of total time** (`import torch` alone is ~82%). Against that baseline a 3.10× forward
speedup moves total wall clock by well under 1%. The knob is a deployment control with a
published quality contract — **not a way to win a time score that is measured end-to-end.**
Whether it moves the score depends entirely on whether round 3 times the forward pass or
the whole process.
