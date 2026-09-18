# Engineering log — KLA PS01, Round 2

Every experiment we ran on the round-2 data, including the ones that failed and
the conclusions we had to withdraw. Team ForgeX · VIT Vellore.

---

## 0. Method: how we decide that something is real

Nothing below is reported unless it clears a measured noise floor.

**The floor.** Two training runs differing *only* in random seed — same data,
same split, same schedule — land **0.010 dB** apart when scored as a
final-5-epoch mean. Scored by best epoch instead, the same pair appears
**0.055 dB** apart, because best-epoch selection is a max over ~120 noisy draws
and therefore rewards luck.

Consequences we hold to throughout:

1. Every number is a **final-5-epoch mean**, never a best epoch.
2. Anything under 0.010 dB is reported as **null**, regardless of sign.
3. The validation split is fixed by `--split-seed`, independent of `--seed`.
   Before this separation existed, a single `--seed` drove both, so two "seed
   variance" runs would have been scored on two *different* validation sets and
   the experiment would have measured nothing.

---

## 1. Round 1 → Round 2: what actually changed

We re-derived every data property rather than assuming it carried over. Almost
none of it did.

| Property | Round 1 | Round 2 | How measured |
|---|---|---|---|
| Content | natural photographs | SEM micrographs | contact sheet, visual |
| Pairs | 3,200 | 4,785 | inventory |
| Sizes | mixed | all 256→128, no 512×512 | shape histogram over all pairs |
| f90 | 0.0945 | 0.2913 | radial power spectrum |
| Normalisation | — | per-image min–max, exact `[0,1]` | 200/200 spans exactly `[0,1]` |
| Downsampling | 2×2 box average | 2×2 box average | fit vs bilinear/bicubic/area/blur-decimate |
| Bicubic ×2 | 23.23 dB | 20.22 dB | direct measurement |

**Dataset ids are content-ordered.** Scoring by contiguous id block gives f90
0.3677 (ids 478–955) monotonically down to 0.1233 (ids 2868–3345), while
σ_mul stays flat at ≈0.19 throughout. So content varies along the id axis and
noise does not — which makes id blocks usable as true out-of-distribution
splits (`--holdout-range`), and means a random split is *easier* than the hidden
test set will be.

---

## 2. The degradation model, refitted

### 2.1 The fit

Regressing residual variance on the ground-truth block statistics:

```
var(residual) = σ_add² + σ_mul²·m²  +  c·v          R² = 0.948
```

`m` = 2×2 block mean (the noise-free low-res reference), `v` = within-block
variance (how much detail the block destroyed).

| Parameter | Observed range | Training range (deliberately wider) |
|---|---|---|
| σ_add | 0.000 – 0.065 | 0.000 – 0.094 |
| σ_mul | 0.155 – 0.258 | 0.116 – 0.335 |
| c | 0.026 – 0.234 | 0.016 – 0.316 |

### 2.2 Three corrections to the round-1 generator

| Change | Evidence |
|---|---|
| **Added `c·v`** | detail share of variance: 4.4% on SEM vs 1.8% on photographs; `c/b` ratio 3.40 vs 2.22 |
| **Removed blur** | measured *harmful*: residual error 0.01074 → 0.01235 with blur present |
| **Gaussian → Gamma(k=14)** | residual is spatially white (lag-1 −0.055 / −0.053 / +0.009) but skew +0.43, excess kurtosis +0.62 |
| **Removed operation-order randomisation** | round 1 randomised order on the brief's "do not read into the order"; on round-2 data the order is identifiable and fixed |

### 2.3 Calibration check

Refitting our own synthetic output with the same estimator used on KLA's data:

| | σ_add | σ_mul |
|---|---|---|
| KLA's real pairs | 0.0414 | 0.1863 |
| Our rebuilt generator | 0.0415 | 0.1986 |

Reproduce: `python tools/calibrate.py --data <data>`.

---

## 3. The experiment ledger

All runs: 120 epochs unless noted, 500 iterations/epoch, batch 32, crop 64,
ch 64, nb 16, seed 0, split-seed 0, AMP on. Scores are final-5-epoch means on
the same held-out 200.

| Run | What it tested | PSNR | SSIM | Δ vs reference | Verdict |
|---|---|---|---|---|---|
| `r2-preal1` | 100% real pairs | **23.2672** | **0.6124** | — (reference) | **shipped** |
| `r2-newgen` | 70% rebuilt synthetic | 23.2178 | 0.5925 | −0.0494 | rejected |
| `r2-long120` | 70% old synthetic | 23.1342 | 0.5930 | −0.1330 | rejected |
| `r2-degfix` | degradation recalibration only | 23.1180 | 0.5932 | −0.1492 | rejected |
| `r2-full-s0` | 40-epoch baseline, seed 0 | 23.1038 | 0.5839 | — | screening |
| `r2-full-s1` | identical but seed 1 | ≈23.09 | — | 0.010 | **the noise floor** |
| `r2-sub500` | 500 training images only | 23.0893 | 0.5838 | +0.006 vs full | null |
| `r2-ctx127` | 1.8× receptive field, params matched | — | — | +0.008 | null |
| `v1` (round 1 model) | zero-shot on round-2 data | 23.1721 | 0.5596 | −0.095 | see §4.2 |

### 3.1 The headline result

**Question, pre-registered before the run:** does a correctly-calibrated
generator beat training on real pairs alone?

**Answer: no, and the rebuild still improved.** `r2-newgen` beat the old
generator by **+0.084 dB** (8× the noise floor) — the recalibration genuinely
worked. It still lost to real pairs by **0.049 dB and 0.020 SSIM**.

We shipped `p_real = 1.0` on the strength of this, and the round-1 conclusion —
synthesise unlimited pairs, keep 30% real as insurance — looked like it did not
survive on round-2 data.

> **This conclusion was later reversed. See § 7.** The measurement above is
> correct and stands; the inference drawn from it does not. σ ≈ 0.19 — where
> this comparison was made — is the one operating point at which all three
> models agree, so the comparison was resolving a 0.13 dB difference at the
> place where there is nothing to see. The shipped configuration is
> 50% real / 25% wide synthetic / 25% narrow synthetic.

### 3.2 The null-result table

| Change | Effect | Verdict |
|---|---|---|
| 9× training data (500 → 4,585) | +0.006 dB | null |
| 2.7× parameters (1.37 M → 3.74 M) | +0.026 dB | null |
| 1.8× receptive field, params matched | +0.008 dB | null |
| Random seed | 0.010 dB | the floor |
| Degradation recalibration alone | −0.016 dB | null |
| 3× training length (40 → 120 epochs) | +0.030 dB | marginal |
| Real pairs instead of synthetic | **+0.133 dB** | **real** |

One in seven cleared the floor. This table is the most useful artefact we own:
it tells us where *not* to spend the remaining time.

### 3.3 Content dominates everything we control

| Block | f90 | PSNR |
|---|---|---|
| Coarsest (ids 2868–3345) | 0.1233 | 26.664 dB |
| Finest (ids 478–955) | 0.3677 | 21.274 dB |
| **Spread** | | **5.39 dB** |

40× larger than any architectural change measured. Which images the hidden test
set contains will move the score more than anything left to tune.

### 3.4 Capacity frontier — complete

Seven configurations, 40 epochs each, `p_real 1.0`, identical split
(`docs/capacity_sweep.csv`). Latency is measured, not estimated:

| Config | Parameters | PSNR | ms / image |
|---|---|---|---|
| ch12 nb4 | 17,449 | 22.7959 | 0.53 |
| ch16 nb4 | 30,753 | 22.8300 | 0.49 |
| ch16 nb8 | 49,313 | 22.8738 | 0.73 |
| ch24 nb8 | 110,257 | 22.9056 | 1.27 |
| ch32 nb8 | 195,393 | 22.9410 | 1.67 |
| ch32 nb16 | 343,361 | 22.9790 | 2.85 |
| **ch64 nb16 (shipped)** | **1,368,705** | **23.0865** | **6.50** |

Regressing PSNR on log2(parameters): **0.0448 dB per doubling, R² = 0.992**.
Across the full 78.4× span the total gain is **0.291 dB**, and there is **no
knee anywhere** — the fit is a straight line to three decimal places.

A model starved of capacity climbs steeply and then flattens. This one does
neither, which is the diagnostic that we are not capacity-limited: the
bottleneck is information the 2×2 downsample destroyed, and parameters do not
recover destroyed information.

Two consequences we act on. First, "make it bigger" is closed as a line of
attack — 2.7× parameters bought +0.026 dB elsewhere, inside the floor, and this
curve says why. Second, 1.37 M is now a *decision* rather than a default: a
49 k model gives up 0.21 dB, is 28× smaller and 9× faster, which is the trade to
make if deployment ever demands it.

Two smaller observations. **ch16 nb4 is faster than ch12 nb4** (0.49 ms vs
0.53 ms) despite 1.8× the parameters — below ~30 k, kernel-launch overhead
dominates and parameter count stops predicting latency. And a rival team's
20.9 k-parameter model, scored on our split, lands on this curve within 0.02 dB,
which suggests the curve is a property of the problem rather than of our
architecture.

### 3.5 Out-of-distribution holdout — complete

Ids 2868–3345 — structure 3× coarser than the rest — withheld from training
entirely, then scored on that block:

| model | trained on the block? | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| `r2-preal1` (real-only) | **yes** | 24.5622 | 0.6638 | 0.3385 |
| `ood-w50` (wide, block withheld) | no | 24.4261 | 0.64389 | — |
| `loss-lp05-120` | no | 24.2752 | 0.6378 | 0.1885 |
| **shipped** (`pr50-w50-lp05-120`) | **no** | **24.3021** | **0.6416** | **0.1816** |

The generalisation cost is **0.136 dB** for the wide model that never saw the
block, against a model that trained on it — far smaller than the 0.4 dB we had
budgeted, and 40× smaller than the 5.39 dB spread content alone produces.
(`ood-w50` ran the full 120 epochs on the laptop; the figure above is its
final-5 mean, 24.4261.)

That run also gave the cleanest confirmation of § 0's methodology we have.
Its **best epoch reads 24.4767 against a final-5 mean of 24.4261 — an inflation
of 0.051 dB**, against the 0.055 dB we measured for best-epoch selection at the
start of the project. Its final-5 spread is 0.0126 dB, against a 0.010 dB seed
floor. Both numbers landed where the method said they would, on a run that had
nothing to do with establishing them.

The shipped model gives up a further 0.14 dB of PSNR and returns **44% better
LPIPS** on the same unseen content. That is the § 3 trade again, now measured
on content the model has never seen rather than on the training distribution.

The earlier projection in this section — "a generalisation cost near 0.05 dB",
extrapolated from epoch 42 — was too optimistic by 3×. The extrapolation
assumed the holdout run would gain at the same rate as the reference run over
epochs 42→119. It gained more slowly. Recorded because the projection is the
kind of thing that gets quoted before the run finishes.

---

## 4. Corrections — claims we had to withdraw

Kept deliberately. Each is a check that caught something, and each changed how
we verify afterwards.

### 4.1 A CPU test used to certify GPU behaviour

We claimed an optimised `run.py` was bit-identical to its predecessor on the
strength of a CPU comparison. On GPU it differed by 2.4e-4 — the fp16 quantum.
The CPU check had no autocast, no `cudnn.benchmark`, and batch 1, so it
exercised none of the code paths that could differ. **This is the same failure
as round 1's `.half()`-without-autocast verification.** Equivalence is now
checked on the device that will run it, in the configuration that will run.

### 4.2 Two predictions that were simply wrong

* Predicted degradation recalibration would recover ≈1.2 dB. It recovered
  **zero** (`r2-degfix` 23.1180 vs `r2-long120` 23.1342).
* Predicted retraining on-domain would recover ≈1.2 dB PSNR. It recovered
  **0 PSNR** (though +0.023 SSIM and −0.10 LPIPS).

Both errors share a cause: treating "gain over bicubic" as a fixed quantity when
the achievable range had itself shrunk. The round-2 bicubic floor is 3.0 dB
lower than round 1's, so equal *skill* shows up as a lower *score*.

### 4.3 Optimisations that were slower

An "optimised" inference script ran **3.8× slower** on GPU (17.0 s vs 4.5 s).
Cause: `cudnn.benchmark=True` autotuning and per-batch `pin_memory()` — both
added by reasoning, neither measured. Removing them gave 3.8 s. Nothing enters
`run.py` now on the strength of an argument alone.

### 4.4 A biased benchmark, caught before it reached a slide

`tools/noise_sweep.py` builds its test inputs with *our own* generator, which
structurally favours a synthetic-trained model. It was nearly used to justify a
model-selection decision. `tools/real_noise_bins.py` replaces it: real data
only, binned by measured σ or by f90.

### 4.5 A seed that controlled two things

`--seed` originally drove both the train/validation split and training
randomness, which would have made the entire seed-variance experiment —
and therefore the noise floor — meaningless. `--split-seed` now separates them.

---

## 5. Latency

End-to-end, warm, 200 images, RTX 4050 Laptop:

| Stage | Time | Share |
|---|---|---|
| `import torch` | 1.62 s | 35% |
| CUDA context init | 1.67 s | 37% |
| Compute + disk I/O | 1.17 s | 26% |
| **Total** | **4.58 s** | |

Cold, `import torch` alone is 11.07 s (75.8% of the run). **Roughly 72% of
wall-clock is fixed process startup, not the network** — so shrinking the model
attacks at most the last quarter of the budget, and startup amortises over the
size of the test set. `run.py` therefore overlaps what it can: CUDA
initialisation on a background thread, threaded input reads, memory-mapped
weights, a writer pool.

---

## 6. Open questions

Superseded by § 15 at the end of this log.

---

## 7. The reversal: how a settled conclusion came undone

This is the most important entry in this log, because for about a week we were
confidently wrong in a way that no in-distribution test could have caught.

### 7.1 What we believed

§ 3.1 established, correctly, that 100% real training pairs beat every synthetic
mixture on the held-out split: `r2-preal1` 23.267 against `r2-newgen` 23.218 and
`r2-long120` 23.134. We shipped `p_real = 1.0` and wrote it up as a finding that
*reversed* round 1's conclusion. The measurement was sound. The inference from
it was not.

### 7.2 What broke it

`tools/noise_sweep.py` regenerates the same held-out images at nine noise levels
and scores every model on identical inputs. Gain over bicubic, in dB:

| σ_mul | 0.05 | 0.10 | 0.15 | **0.19** | 0.25 | 0.30 | 0.35 | 0.40 | **0.45** |
|---|---|---|---|---|---|---|---|---|---|
| r2-preal1 | +0.86 | +1.47 | +2.04 | **+2.40** | +2.45 | +2.15 | +1.89 | +1.79 | **+1.88** |
| r2-newgen | +0.98 | +1.52 | +2.17 | **+2.70** | +3.59 | +4.30 | +4.74 | +4.22 | **+2.65** |
| r2-long120 | +0.82 | +1.33 | +1.99 | **+2.45** | +2.96 | +3.63 | +4.49 | +5.24 | **+5.71** |

σ ≈ 0.19 is where KLA's data sits. All three models are within 0.30 dB there,
which is precisely why a single-point comparison could not distinguish them —
we had been resolving a 0.13 dB difference at the one place the models agree.

Three steps out, `r2-preal1` is **3.83 dB behind**. Worse, it *turns over*: its
gain peaks at σ = 0.25 and then falls. A model that had learned the inverse
problem would keep gaining as noise rises, because there is more noise to
remove. Turning over is the signature of a model that has learned a noise
amplitude rather than a noise process.

### 7.3 The objection, and why the result survives it

§ 4.4 of this log records our own objection to `noise_sweep.py`: it builds test
inputs with our generator, which structurally favours a synthetic-trained model.
We nearly discarded the sweep for that reason. Two things rescue it.

1. **The bias does not appear where it should.** At σ = 0.05–0.19, `r2-preal1`
   (which never saw our generator) *beats* `r2-long120` (which trained on it).
   A test that merely rewarded generator-fit could not produce that ordering.
2. **The conclusion reproduces on inputs the generator never touched** — the ten
   NFFA morphologies (§ 8) and the withheld content block (§ 3.5), both scored
   against real ground truth.

The sweep is where the effect was *found*. It is not where it is certified.

### 7.4 What we changed

`degrade_wide()` samples a family deliberately wider than the observed one —
σ_add to 0.180, σ_mul 0.060–0.500, c to 0.450, Gamma k 4–60, optical blur on
30% of samples, a Gaussian-weighted downsampling kernel on 25%. Blur is applied
to the **clean ground truth before noise is sampled**: optics act on the image,
not on the sensor, and blurring the degraded output would spatially correlate
noise that we measured to be white (lag-1 −0.055 / −0.053).

The shipped mixture is 50% real / 25% wide / 25% narrow. (`wide_p` is the
fraction of the SYNTHETIC draws taken from the wide family, not of all draws:
`--p-real 0.5 --wide-p 0.5` leaves 50% synthetic, split half and half. The runs
tagged `pr30-*` in § 12 used `--p-real 0.3`, which is 30/35/35; we stated that
as 30/50/20 in an earlier revision, which was wrong.)

### 7.5 The lesson, stated so we do not repeat it

**A single-point evaluation cannot distinguish a model that solved the problem
from a model that memorised the operating point.** Every comparison in this
repo now sweeps at least one axis. The cost of learning this was a week; the
cost of not learning it would have been a model that scored well on our split
and fell over on a test set drawn even slightly differently — which is exactly
what the organisers said their round-2 test set would be.

---

## 8. Generalisation, tested four ways

### 8.1 Ten labelled morphologies (`tools/per_category.py`)

KLA's round-2 images come from NFFA-EUROPE, which ships **its own morphology
labels** — ten named classes. This was the user-facing idea that unlocked the
section: we do not need to *infer* content categories, because the source
dataset already provides them. We took NFFA natives, cropped above the
instrument data bar, applied the observed degradation, and scored per class
(`docs/per_category.csv`):

| category | bicubic | ours | gain | SSIM | LPIPS |
|---|---|---|---|---|---|
| Patterned surface | 20.485 | 25.761 | **+5.276** | 0.571 | 0.199 |
| Tips | 20.197 | 25.024 | +4.827 | 0.464 | 0.232 |
| MEMS devices / electrodes | 20.132 | 24.515 | +4.382 | 0.451 | 0.254 |
| Fibres | 21.667 | 25.847 | +4.180 | 0.621 | 0.226 |
| Particles | 23.833 | 27.895 | +4.062 | 0.577 | 0.202 |
| Powder | 19.817 | 23.777 | +3.960 | 0.622 | 0.225 |
| Nanowires | 20.904 | 24.727 | +3.824 | 0.719 | 0.201 |
| Porous sponge | 19.681 | 23.470 | +3.789 | 0.571 | 0.222 |
| Films / coated surface | 19.108 | 22.681 | +3.573 | 0.480 | 0.299 |
| Biological | 19.632 | 21.896 | **+2.264** | 0.446 | 0.277 |
| **mean / spread** | | | **+4.014 / 3.012** | | |

All ten positive. The mean gain exceeds the in-distribution gain (+3.18), and
LPIPS stays inside 0.199–0.299 against 0.193 in distribution — so the
perceptual improvement is not tied to one image source.

The 3.01 dB spread orders morphologies the way the f90 analysis predicts:
regular geometric structure (patterned surfaces, tips) is easiest; soft organic
structure (biological) is hardest, because it has the least regular
high-frequency content for a learned prior to exploit.

**Detail that cost time:** the instrument data bar is burned into the bottom of
every NFFA image and its height *varies per image* (content rows 593–661,
median 623). `tools/make_512.py` detects it per image by scanning for the first
dark, low-variance row below 55% height rather than assuming a fixed crop.
Cropping at a constant height would have put black bar pixels into the ground
truth and inflated PSNR on a region containing no image.

### 8.2 Withheld content block

See § 3.5. Generalisation cost 0.14 dB; LPIPS 44% better than the real-only
model on the same unseen content.

### 8.3 Nine-axis OOD suite (`tools/ood_suite.py`)

Three noise levels, three blur levels, one non-box downsampling kernel, two
content blocks — all scored as gain over bicubic on identical inputs, reporting
**mean and worst axis**. The worst-axis figure is the one that decided the
shipped configuration: a model that is excellent on eight axes and negative on
one is not robust, and the mean hides that.

### 8.4 256 → 512 with no training at that scale

778 held-out 512×512 pairs built from NFFA natives, scored with the **shipped**
weights:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| ours | 25.392 | 0.576 | 0.236 |
| bicubic ×2 | 20.877 | 0.410 | 0.538 |
| gain | **+4.515** | +0.166 | **56% lower** |

This row has been wrong twice and both corrections are worth keeping.

**First**, we published **+4.914 dB** from `r2-newgen`, a checkpoint from the
scale study rather than the shipped model. It scores 25.913 PSNR but **0.394
LPIPS against the shipped model's 0.236** — the same trade as everywhere else,
appearing unprompted at a scale neither model saw in training.

**Second**, the replacement figure (25.458 / bicubic 20.999) was scored on a
**200-image subset** of the 778 pairs, because `validate.py` defaults to
`--n-val 200`. The bicubic column gives it away: 20.999 there against 20.877 on
the full set. By our own § 11.2 rule those two files cannot be compared, and the
subset number is withdrawn. The table above scores all 778.

The reason this works at all is structural, not lucky: the network is fully
convolutional and operates at low resolution, so a 256×256 input is just more
patches; and the degradation is *local*, depending only on a 2×2 block, so it is
identical at every scale.

---

## 9. The noise model, settled (`tools/noise_physics.py`)

The `σ_mul²·m²` term in § 2.1 says the noise grows with the *square* of local
brightness. Poisson shot noise — the obvious physical candidate in an
electron-counting instrument — grows *linearly*. The two call for different
variance-stabilising transforms, so we stopped assuming and fitted both.

Four models, identical bins, 6.55 M pixels:

| Model | form | adj R² | BIC |
|---|---|---|---|
| **M1 multiplicative** | `a + b·m² + c·v` | **0.9987** | **−1814.5** |
| M2 Poisson | `a + p·m + c·v` | 0.9370 | −696.2 |
| M3 both | `a + p·m + b·m² + c·v` | 0.9989 | −1849.5 |
| M4 no brightness term | `a + c·v` | 0.1808 | +37.7 |

288 usable bins, 6,553,600 pixels. Raw output committed at
`docs/tool_output/noise_physics.txt`. An earlier revision of this table gave
M4's form as `a` and its adjusted R² as 0.1837 — that was the raw R² of a
different form. Corrected against the tool's own output.

**ΔBIC(M1, M2) = 1118.3.** Conventionally, ΔBIC > 10 is "very strong"; this is
two orders of magnitude past that. M3 fits marginally better than M1 but drives
the linear coefficient to **−0.00196** — negative — which has no physical meaning
for a shot-noise term. It is absorbing curvature, not detecting photons. We keep
M1. (The tool's own "lowest BIC" line names M3; we override it on that physical
grounds, and say so here rather than quietly reporting M1 as the winner.)

Consequences: the generator multiplies rather than drawing Poisson counts, and
the network's stem *offers* `√x` (the Anscombe transform, which stabilises
Poisson) as one of three views rather than being built on it. Had we assumed
Poisson, the natural design would have been to apply Anscombe unconditionally —
stabilising the wrong thing.

---

## 10. Things we tried that did nothing

Each is a real experiment with a measured null, kept because the absence of an
effect is a result.

| Intervention | Measured | Verdict |
|---|---|---|
| 8× test-time augmentation | +0.018 dB, 7.1× slower (12.2 → 87.0 ms) | rejected |
| Pinned staging buffers + CUDA streams | +0.04 s on a 1.94 s run-to-run spread; output bit-identical | rejected |
| `channels_last` layout fix | −0.06 s on a 0.73 s spread; output bit-identical | rejected |
| Content-balanced sampling (p ∝ f90^w) | −0.090 dB | rejected |
| Gaussian-weighted downsampling in the wide family (`soft_p`) | +0.01 dB (+2.99 vs +2.98 OOD mean) | null; **kept only because the shipped weights were trained with it** |
| 3.74 M-parameter model | +0.026 dB | rejected |
| 1.8× receptive field, parameters matched | +0.008 dB | rejected |

A note on reading `docs/results.csv`: the row tagged `SHIPPED-tta+tta`
(23.3023) was measured while `models/model.pt` still held the *previous*
checkpoint, `r2-preal1` (23.2842) — that pairing is where the +0.018 dB TTA
figure comes from. The rows tagged `SHIPPED` (23.0899) are the current weights.
The tag is misleading and is left as written rather than edited after the fact.

**The `channels_last` story is worth the extra line.** Reading `run.py` to answer a
question about inference cost, we found that line 154 converts the input to
`channels_last` and the forward call then does `model(v.contiguous())` — and
`.contiguous()` with no argument converts a channels_last tensor straight back
to NCHW. The layout conversion was being paid for and thrown away every batch.
It looked like free speed. Benchmarked: **−0.06 s median against a 0.73 s
run-to-run spread**, bit-identical output. cuDNN was evidently already picking
the right kernel from the weight layout. A real bug in the code that costs
nothing measurable is still a null result, and we report it as one.

One thing the fix *did* change: its run-to-run standard deviation is 0.29 s
against 1.17 s, and its worst case 6.37 s against 9.13 s. It is steadier, not
faster. We did not ship it, because the median is what the scoring will see and
the median did not move.

On `soft_p` specifically: the honest thing would be to delete it. We are not
deleting it, because `models/model.pt` was trained with it in the sampler, and
removing it from `src/degrade.py` would mean the shipped configuration in the
README no longer reproduces the shipped weights. It is documented as dead weight
instead. If this code outlives the hackathon, that is the first line to cut.

---

## 11. What we got wrong, second batch

### 11.1 A category stride that was pattern-matching

We claimed the content-ordered ids revealed KLA's own category boundaries at a
fixed stride of 478.5 images. Testing the hypothesis properly — comparing the
f90 series' agreement with that grid against random offsets — put the best
offset at the **58th percentile**, i.e. indistinguishable from chance. The claim
is withdrawn. The *ordering* is real and the blocks remain usable as holdouts;
the *stride* was us seeing a pattern in seven numbers.

The replacement is better than the thing we were trying to fake: NFFA publishes
the labels (§ 8.1), so we stopped inferring categories and used the real ones.

### 11.2 A cross-run comparison on different test sets

Three noise sweeps were compared against each other when they had been produced
by different versions of the script against different test sets — the bicubic
column read 24.15 in the old CSVs and 22.74 in the new one. Caught because PSNR
and SSIM rankings disagreed, which is nearly always a broken test rather than an
interesting model difference. Both older runs were re-scored with the current
script before any conclusion was drawn.

**Rule adopted:** a baseline column that differs between two result files means
the two files cannot be compared, full stop. The baseline is the checksum.

### 11.3 A selection rule that ignored two of three metrics

`tools/scorecard.py` prints a `FINAL` column and a `SHIP:` line. That column
averages **PSNR only**. It duly recommended shipping the model with the *worst*
perceptual score of the nine — the model it ranked last on LPIPS was 47% better
there. The recommendation was ignored and the nine configurations were re-ranked
under eight different weightings of PSNR / SSIM / LPIPS / worst-OOD-axis;
`lp05-120` wins under all eight. The `SHIP:` line is left in the tool as a
warning and should not be trusted.

### 11.4 A hallucination metric that flagged the one method that cannot hallucinate

The first version of `tools/hf_energy.py` reported the 95th percentile of
*per-image* HF-energy ratios. On smooth images the denominator approaches zero
and the ratio explodes, so the statistic labelled **bicubic interpolation** —
which cannot synthesise frequency content, by construction — as "HALLUCINATING".
Replaced with an aggregate ratio (sum of numerators over sum of denominators).
Under the corrected statistic the shipped model sits at 0.391 against bicubic's
0.382.

**Rule adopted:** when a metric flags something that is impossible, fix the
metric.

### 11.5 A four-fold error in a time estimate that cost us runs

We forecast 48 s per epoch on the rented GPU and sized the experiment queue at
40 epochs to fit the budget. The actual figure was 14 s. All seven configurations
could have run to 120 epochs in the same wall-clock. Caught by the team, not by
us. The queue was re-run at 120 epochs for the top configurations, but the first
pass was wasted.

**Rule adopted:** measure one epoch before sizing a queue from an estimate.

### 11.6 A superseded number repeated after it was superseded

We stated more than once that `r2-preal1` was *worse than bicubic* at σ = 0.05.
That figure came from the invalid cross-run comparison in § 11.2. The correct
value is **+0.86 dB** — its weakest point, but above bicubic everywhere. The
error survived several retellings because it made a tidier story. Corrected in
the published artifacts as well as here.

---

## 12. The final selection: eleven configurations, one scoring pass

Eleven 120-epoch runs, all on the same RTX 4090 pod, all scored in a **single
pass by the same code against the same inputs** — the shared bicubic column is
the checksum (§ 11.2). Everything below comes from `docs/queue/`:
`score.log`, `queue_results.csv`, and eleven `noise_sweep_*.csv`.

### 12.1 The whole field

In-distribution is the 200-image held-out split. Sweep columns are the nine-level
noise sweep, 60 images per level. OOD is the nine-axis suite, 40 images per axis.

| config | PSNR | SSIM | LPIPS | sweep mean | sweep LPIPS | OOD mean | OOD worst |
|---|---|---|---|---|---|---|---|
| `pr30-w00` | **23.223** | 0.592 | 0.388 | 2.93 | 0.464 | 3.00 | **1.19** |
| `pr50-w50` | 23.220 | 0.599 | 0.372 | 3.43 | 0.481 | 3.06 | 1.14 |
| `pr30-w50` | 23.212 | 0.591 | 0.387 | **3.49** | 0.477 | **3.12** | 1.17 |
| `pr30-w100` | 23.210 | 0.591 | 0.393 | — | — | 3.09 | 1.14 |
| `pr50-w00` | 23.208 | **0.599** | 0.370 | — | — | 2.96 | 1.15 |
| `lp02` | 23.165 | 0.587 | 0.233 | 3.43 | 0.312 | 3.06 | 1.13 |
| `pr30-w50-hard` | 23.121 | 0.592 | 0.355 | — | — | 3.02 | 1.08 |
| `pr50-w50-lp05` | 23.104 | 0.588 | 0.200 | 3.34 | 0.290 | 2.96 | 1.06 |
| `loss-lp05` | 23.090 | 0.584 | 0.206 | 3.35 | 0.288 | 2.99 | 1.07 |
| `loss-ss30` | 23.097 | 0.594 | 0.390 | — | — | 3.04 | 1.13 |
| `loss-lp15` | 22.979 | 0.576 | **0.193** | 3.23 | **0.271** | 2.86 | 1.00 |

Three structures are visible and all three are the story of this round.

**The whole field spans 0.244 dB in-distribution** — 24× the noise floor across
eleven configurations that differ in training mixture, LPIPS weight and SSIM
weight. In-distribution PSNR barely discriminates between them. This is § 7.2
again: a single operating point is where these models agree.

**PSNR order is LPIPS-weight order, inverted.** The five models with no
perceptual term take the top five slots; then `lp02` (weight 0.02), then the two
at 0.05, then `lp15` (0.15) last. The trade is visible in the data rather than
asserted.

**`pr30-w00` is the model that collapses.** Best in-distribution PSNR of all
eleven, and last on the noise sweep at 2.93 mean — +2.36 dB at σ = 0.45 where
every wide-family model holds above +5.9. It is the best model on the test we
would have run in week one and the worst on the test the organisers described.

### 12.2 The selection rule, and a claim we had to withdraw

**We previously wrote that `loss-lp05-120` "wins under all eight metric
weightings, mean 0.629 against `pr30-w50`'s 0.609". That claim does not
reproduce and is withdrawn.**

Rebuilding the analysis from the single clean pass — min-max normalising PSNR,
SSIM, inverted LPIPS, OOD mean and OOD worst across the field, then scoring
under eight weightings from PSNR-only to robustness-heavy — puts `loss-lp05` at
**6th of 9** on the original field and **8th of 11** on the full one, winning
**0 of 8**. The figures 0.629 and 0.609 do not appear anywhere in the data.

This is § 11.3 repeating: a confident selection line, produced by a scoring
helper, that was not checked against the underlying numbers before being
published. It was written *after* § 11.3 was written.

Two things about the aggregate that are worth stating even though it embarrassed
us. It is **field-dependent** — min-max normalisation means adding a model
changes everyone's score — and it gives each metric equal *range* regardless of
whether that range is meaningful, so PSNR's 0.244 dB spread counts for exactly
as much as LPIPS's 0.20. We report it because we published a number from it, not
because we think a single aggregate should decide this.

### 12.3 What survives, and what the trade actually is

The aggregate collapsing does not touch the two decisions the model rests on,
because both are visible directly in the table.

**The wide family.** `pr30-w00` at 2.93 sweep-mean against `pr30-w50` at 3.49,
and +2.36 vs +6.08 dB at σ = 0.45. Confirmed on this clean pass, larger than we
previously reported.

**The perceptual term.** `pr50-w50` (no LPIPS) gets +0.116 dB over
`pr50-w50-lp05` and pays **86% worse LPIPS** (0.372 vs 0.200) for it. Across the
sweep the gap is 0.481 vs 0.290. If LPIPS is scored at all, that is not a close
call; if it is not scored, we gave up 0.5% of PSNR.

**What does not survive is the specific choice of `loss-lp05` within the
perceptual camp.** `pr50-w50-lp05` is better on PSNR, SSIM *and* LPIPS
simultaneously, and is behind only on the two OOD-suite numbers by 0.03 and
0.01 dB — which the 60-image noise sweep contradicts, putting them level at 3.34
vs 3.35. More real anchoring (`p_real` 0.5 vs 0.3) bought in-distribution
accuracy at no measurable robustness cost.

There is no dominant model. `lp02` is the most robust of the perceptual group
(sweep 3.43, OOD worst 1.13) at 13% worse LPIPS; `pr50-w50-lp05` is the most
accurate at 0.09 dB less sweep gain; `lp15` is the best perceptually and the
weakest everywhere else. That is a frontier, not a winner, and saying so is more
honest than manufacturing an aggregate that picks one.

### 12.4 Two late runs, and the rule written before the numbers arrived

Two further 120-epoch configurations were queued after `lp05-120` had already
been deployed as `models/model.pt`:

* `lp02-120` — `--p-real 0.3 --wide-p 0.5 --w-lpips 0.02` (half the LPIPS weight)
* `pr50-w50-lp05-120` — `--p-real 0.5 --wide-p 0.5 --w-lpips 0.05` (more real anchoring)

Swapping the shipped checkpoint is not free. Every generalisation number in this
repo — the ten-category table, the hallucination ratio, the withheld-block
score, the in-distribution triple, and the committed `outputs/` — was measured
against the *current* weights and would all have to be re-measured. With the
deadline where it is, that is a real cost, which makes it exactly the situation
in which one rationalises a marginal number into a win.

**So the rule is written here, before the results were read.** Replace
`lp05-120` only if a challenger:

1. wins on at least **three of the four** decision axes — in-distribution PSNR,
   in-distribution LPIPS, mean OOD gain, worst OOD axis; **and**
2. loses no axis by more than the measured noise floor (0.010 dB on PSNR); **and**
3. still wins under the same eight metric weightings used in § 12.2.

Anything short of all three and `lp05-120` ships, because a difference that
cannot clear that bar is not worth invalidating four measured tables for.

**What happened when the numbers arrived — including to the rule itself.**

`lp02-120` wins three of the four axes (PSNR +0.075 dB, OOD mean +0.07, OOD
worst +0.06) and loses LPIPS by 0.027, which is 13%. It satisfies condition 1
and fails condition 2 — except that condition 2 defines the tolerance as
"0.010 dB on PSNR" and **we never measured a seed-variance floor for LPIPS**, so
the rule cannot actually adjudicate its own decisive axis. That is a defect in
the rule, not a verdict.

`pr50-w50-lp05-120` wins two of four (PSNR, LPIPS) and loses two (OOD mean by
0.03, OOD worst by 0.01). It fails condition 1 — but the rule counts OOD twice,
since the worst axis is one of the nine axes inside the mean, so a model
marginally behind on the suite automatically loses two of four. And the
60-image noise sweep contradicts the 40-image suite, putting it level with
`lp05` at 3.34 vs 3.35.

Condition 3 was unsatisfiable from the start: it required winning under the
eight weightings of § 12.2, and § 12.2 is the claim we withdrew.

So the rule, written specifically to stop us rationalising, turned out to be
under-specified on one axis, double-counting on another, and anchored to a
result that did not survive. We are recording that rather than presenting a
repaired rule as though it had been the rule all along. The decision was made on
the § 12.3 reading instead: the frontier is real, no model dominates, and the
choice within the perceptual group is a judgement call made with the numbers
visible.


---

## 13. The OOD suite on the organisers' own test set

Every OOD number above this section was measured by re-degrading *our* data.
When the organisers released a 297-image paired test set we re-ran the suite
against theirs (`docs/tool_output/ood_testset.txt`), comparing the shipped
weights against `r2-preal1` — the 100%-real-pairs model that has the best
in-distribution score of anything we trained.

| axis | bicubic | shipped | 100% real pairs |
|---|---|---|---|
| noise 0.05 | 23.09 | +1.07 | +1.01 |
| noise 0.19 | 19.79 | +2.91 | +2.66 |
| **noise 0.40** | 15.54 | **+5.60** | **+2.09** |
| blur 0.4 | 19.82 | +2.86 | +2.64 |
| blur 0.8 | 19.70 | +2.70 | +2.57 |
| blur 1.2 | 19.44 | +2.65 | +2.56 |
| soft (non-box) kernel | 19.74 | +2.92 | +2.66 |
| **mean / worst** | | **+2.96 / +1.07** | +2.31 / +1.01 |

Better on **every one of the seven axes**, and by **3.51 dB** at high noise.
`r2-preal1` beats us by 0.234 dB on the undamaged test set and loses on all
seven axes the moment the degradation moves. That is the § 7 trade, measured
end-to-end on the organisers' own images rather than on our synthetic sweep.

### 13.1 A bug the run exposed

The suite also defines two **content** axes by training-set id range —
1850–2134 and 0–1534. The test set has ids 0–296. `pick(lo, hi)` returned an
empty list for the coarse block, `np.mean` of an empty slice returned `NaN`,
and the `NaN` propagated into the MEAN row for every model:

```
content coarse  bicubic    nan   models +nan   r2-preal1 +nan
MEAN gain                        +nan          +nan
```

The failure was loud, which is the only reason it was caught immediately. It
would have been quiet — and wrong — if the range had merely been *partly*
present, returning a mean over three images.

`tools/ood_suite.py` now skips a content block when fewer than `n/4` images fall
inside its range, prints which blocks it skipped and why, and computes the mean
over the axes that actually ran. The seven axes above are the ones meaningful on
a dataset with its own id numbering.

**Rule adopted:** an axis defined by a property of one dataset must check that
the property exists before scoring, and say so when it does not.

---

## 14. Shipping `pr50-w50-lp05-120`

`tools/reship.py` deploys a checkpoint and re-measures, in one pass, every
number that depends on which weights are shipped. Ten of eleven steps passed;
the eleventh is § 14.3.

### 14.1 What changed against `loss-lp05-120`

| measurement | `loss-lp05` | **shipped** |
|---|---|---|
| organisers' test set | 23.608 / 0.6037 / 0.1975 | **23.632 / 0.6079 / 0.1929** |
| our 200-image split | 23.090 / 0.5838 / 0.2060 | **23.104 / 0.5884 / 0.2002** |
| withheld content block | 24.275 / 0.6378 / 0.1885 | **24.302 / 0.6416 / 0.1816** |
| ten NFFA morphologies, mean | +4.059 | +4.014 |
| 256 → 512 | — | 25.392 / 0.576 / 0.236 |
| noise sweep, mean / at σ 0.45 | 3.35 / +5.91 | 3.343 / +5.908 |
| hallucination aggregate | 0.402 | 0.421 (0 images over 1.0) |

Better on all three metrics on the organisers' test set, on our own split, and
on unseen content. Marginally lower on the NFFA mean and at 512 — those are the
two places it gives ground, and both are inside a tenth of a dB.

### 14.2 Cross-machine reproducibility, restated

The pod scored these weights at **23.632 / 0.60791 / 0.19288** on the
organisers' test set. The laptop, a different GPU on a different machine,
scored **23.632 / 0.60791 / 0.19288**. Identical to five decimals.

Our previous claim was 22.7468 against 22.74 — a two-decimal agreement on a
different configuration. This is stronger, and it matters because the eleven-run
selection queue was split across the two machines. Had they disagreed, the whole
comparison would have been invalid in exactly the way § 11.2 describes.

### 14.3 The step that failed

`tools/package_check.py` crashed with `ModuleNotFoundError: No module named
'degrade'`. The script put the repo root on `sys.path`, but `src/dataset.py`
imports its siblings by bare name (`from degrade import ...`), so `src/` itself
has to be on the path. A one-line fix.

It is recorded because of *when* it failed: the packaging verification is the
last thing between us and submission, and it had never been run against the
new weights. A harness that only runs at the end is a harness that fails at the
end.

---

## 15. Open questions

1. **We cannot verify the 512×512 path against KLA's own 512 data**, because the
   round-2 training set contains none. Our 512 evidence is built from NFFA
   natives — the same corpus, not the same acquisition.
2. **The wide family's bounds are a judgement, not a measurement.** We know
   breadth helps three σ-steps out; we do not know these bounds bracket whatever
   the round-2 test set contains.
3. **Metric weighting.** The organisers have said they do not know how PSNR,
   SSIM and LPIPS will be combined. We optimise all three jointly and selected
   the shipped model under eight different weightings rather than betting on one.
4. **No genuinely different microscope has been tested.** Every image we have
   seen, KLA's and NFFA's alike, traces to the same corpus.
5. **ONNX export is untested** — the only remaining lever on the 72% of
   wall-clock that is process startup.
6. **Per-bin ceiling.** § 3.3 gives per-content PSNR but not per-content
   *ceiling*; the two together would say whether the fine-structure blocks are
   genuinely harder or merely further from convergence.



