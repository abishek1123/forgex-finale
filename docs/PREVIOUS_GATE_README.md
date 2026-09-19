# Historical gate-release README

This is the previous release documentation. For the current trained shared-head model and matching H100 engines, use the root README.

# AI-Based Restoration of Degraded Images — KLA PS01

Restores degraded semiconductor inspection images: removes signal-dependent
noise and upscales 2× (128×128 → 256×256, or 256×256 → 512×512).

**Team ForgeX** — Abishek SR · Anmol BA · Hardik — VIT Vellore
SEMICON India Hackathon 2026 · Problem Statement PS01 · **Grand Finale**

Shipped model `e1f_gate`, 1,346,360 parameters, `models/model.pt` sha1
`f95377a21e86…` — **23.8297 PSNR / 0.62104 SSIM / 0.18126 LPIPS** on the
organisers' 297-image test set.

---

## Quick start — evaluators

```bash
pip install -r requirements-inference.txt     # torch + numpy, nothing else
python run.py <input-dir> <output-dir>
```

That is the whole submission path. `run.py` imports only torch and numpy,
inlines the network definition, and resolves its weights to `models/model.pt`.
It runs on CPU if no GPU is present — use `requirements-cpu.txt` instead if the
CUDA 12.6 wheels above do not resolve on your machine. CPU and GPU agree to
0.0001 dB; the committed `outputs/` were produced on GPU under fp16 autocast.

**Contract.** In: `.npy`, `(H,W)` float32, any range. Out: `.npy` float32
`(2H,2W)`, same filename, finite, clamped to `[0,1]`, trailing singleton axis
preserved if the input had one. If you point it at a dataset directory holding
`GT/` and `NoisyLR/`, it finds the inputs one level down and says so.

**Check you have what we shipped:**

```bash
python tools/verify_shipped.py                # hash, re-run, compare to outputs/
```

It compares `models/model.pt` by sha1 — never by filename and never by size —
then re-runs the model and diffs against the 297 restorations committed in
`outputs/`. Anything other than `CORRECT` means stop.

---

## The quality / speed knob

The same weights serve fourteen operating points. `--depth N` runs only the
first N of the 16 residual blocks; nothing is retrained, reloaded or distilled.

```bash
python run.py <in> <out> --depth 10
```

That is the whole feature, and it behaves identically on any machine — it
changes the graph, not a timing decision. `params=` in the banner drops with it
(1.35M at full depth, 0.89M at depth 10, 0.36M at depth 3), so you can see it
took effect.

**Why a shallow run is safe.** The network does not predict the image. It
predicts a *correction* to a bicubic upsample, through a tail initialised at
zero, added back in FP32. Remove blocks and the correction gets smaller, so the
output slides back toward the bicubic baseline rather than toward noise —
measured, even depth 0 is still +2.21 dB over bicubic.

**`--depth 1` will not obey you.** Depths 0–2 are not monotone (22.6648,
22.6564, 22.6501 — adding a block there makes the output slightly worse), so
`MIN_SAFE_DEPTH` is 3 and anything below it warns and clamps. A control that
silently honours a setting we know to be bad is not a control.

### The menu

```bash
python run.py --list-knob
```

Prints all 14 settings from `models/knob_datasheet.json`. **The PSNR and SSIM
columns are hardware-independent** — they are the real 297-image scores and
will not change on your machine. **The ms/image column is not.** The datasheet
records which GPU produced it; the one shipped here says `NVIDIA H100 NVL`.

### Choosing by latency instead of by depth

```bash
python run.py <in> <out> --budget-ms 1.0
python run.py <in> <out> --budget-ms 1.0 --prefer ssim
```

**How the budget is resolved — read this before relying on it.** Nothing is
timed during the run. `--budget-ms` is a *lookup*: it filters the datasheet to
rows whose recorded `ms_per_img` fits your budget and returns the best-quality
survivor (maximising PSNR, or SSIM with `--prefer ssim`). It deliberately does
**not** take the deepest setting that fits — a quality curve can be
non-monotone, and "deepest that fits" would then hand you a setting that is
strictly worse.

Those milliseconds were measured offline by `tools/calibrate_knob.py`: 14
depths × 5 interleaved round-robin rounds × 30 iterations at batch 32 and
128×128, best and worst round dropped, median taken. Measuring at run time
would cost more than the knob saves on a 297-image job.

**So on hardware other than the datasheet's, the millisecond figures do not
transfer, and the selected setting may not meet the budget you asked for.**
`run.py` compares the datasheet's recorded GPU against the current device and
warns when they differ. To make the budget meaningful on your machine:

```bash
python tools/calibrate_knob.py --data <organisers-test-set> --rounds 5
```

A few minutes on a laptop — it is roughly 90,000 timed forward passes. The
quality columns come out identical; only the timings change.

**If you want one number you can trust without calibrating, use `--depth`.**
The quality of every depth is fixed and published; only the speed is yours to
measure.

## TensorRT, with a PyTorch fallback

`run.py` chooses its own runtime. There is no second entry point and no flag to
remember — the same command a judge already typed uses TensorRT whenever
TensorRT can serve the request, and PyTorch whenever it cannot.

```bash
python run.py <in> <out>              # TensorRT if it can, PyTorch if not
python run.py <in> <out> --no-trt     # force PyTorch
```

**Why the probe is at the top of the file.** `import torch` is 54% of a scored
run — 1.445 s of 2.686 s on an H100 — while the model's own arithmetic is 4.7%.
A runtime chosen *after* importing torch would already have spent the entire
saving. So the first thing `run.py` does, above its own torch import, is read
the `.npy` **headers** (never the pixels) to learn the shapes, and then
`os.execv` into the TensorRT runner — which replaces the process, so nothing is
imported twice. The probe costs ~0.12 s. When it dispatches it says so:

```
[run.py] TensorRT: gate_d16.plan (depth 16, fp32(tf32))  probe 0.144s
```

Measured on an H100 NVL over the 297 images: **2.046 s against PyTorch's 2.586 s
at full depth (1.26×), 1.513 s against 2.364 s at depth 3 (1.56×)**, for a PSNR
difference of 0.0001 dB. Warm steady state is 1.61 s, 184 images/s.

**It falls through to the PyTorch path** — unchanged, 12/12 stress — on any of:
`--no-trt`; `--tta`, `--half`, `--no-fp16`, `--profile`, `--list-knob`;
`--device cpu`; no engine index; no engine at the requested depth (the depth is
honoured exactly, never substituted); `models/model.pt` not matching the
engine's recorded sha1; an input outside the engine's shape profile; TensorRT
not installed; or a TensorRT failure at run time before any output has been
written. Being wrong costs a tenth of a second, never a result.

**The engines are not in this repository.** A `.plan` is compiled machine code
welded to one GPU family, one TensorRT version and one checkpoint hash — 99 MB
of build artefact. A fresh clone therefore runs correctly on PyTorch and says
`engine file missing`. To build them (five engines, ~145 s, needs TensorRT 11.3
and cuda-python):

```bash
python tools/trt_native.py all --depths 3,6,10,13,16 --portable
python tools/bless_engines.py                # proves engine <-> checkpoint identity
```

`run_fast.py` is kept as a forwarding shim so older notes still work; it simply
execs `run.py`.

## Reproducing the numbers

```bash
python tools/package_check.py --data <organisers-test-set>   # the submission, end to end
python tools/calibrate_knob.py --data <test-set> --rounds 5  # the knob datasheet
python tools/engine_report.py --test <test-set> --reps 9 --warmup 2
python tools/engine_report.py --from-json docs/engine_report.json   # no GPU needed
python train_submitted.py --data <dataset>                   # retrain from scratch
```

Datasets are **not** in the repo — see [`docs/DATA.md`](docs/DATA.md).

**[`docs/SHIP_E1F_GATE.md`](docs/SHIP_E1F_GATE.md)** is the technical summary:
the model, the 297 scores, the robustness sweep against the model it replaces,
all five engines, the dispatcher, and a closing section recording every claim we
withdrew and the measurement that overturned it.

---

## Team setup

```bash
git clone https://github.com/abishek1123/forgex-finale.git
cd forgex-finale

python -m venv .venv
.venv\Scripts\activate                       # Windows
# source .venv/bin/activate                   # Linux / macOS

pip install -r requirements.txt               # GPU, CUDA 12.6
# pip install -r requirements-cpu.txt         # no NVIDIA GPU

python tools/verify_shipped.py
```

Before you push, demo, or hand over: **`python tools/verify_shipped.py`**. The
wrong model has shipped three times on this project; twice it scored *higher*
PSNR than the correct one, so only a hash check catches it.
See [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Everything below this line is
*why* — the degradation model, the experiments, and the reasoning behind each
decision. Read it when you need it, not to get started.

---

**Input:** `.npy`, grayscale, shape `(H, W)` or `(H, W, 1)`, float32. Values may
fall outside `[0, 1]` — expected for this degradation, and handled.

**Output:** `.npy`, float32, shape `(2H, 2W)` — or `(2H, 2W, 1)` if the input
carried a trailing channel axis. Guaranteed finite and clipped to `[0, 1]`.

**Requirements:** `run.py` imports `torch` and `numpy` only. No internet, no API
keys, no model downloads, no user interaction. Weights ship in
`models/model.pt`.

| Flag | Effect |
|---|---|
| `--weights PATH` | use a different checkpoint (default `models/model.pt`) |
| `--device cpu` | force CPU |
| `--no-fp16` | disable half precision on CUDA |
| `--tta` | 8× self-ensemble. Measured **+0.018 dB for 7× the time** — off by default, and we recommend leaving it off. |
| `--batch N` | images per forward pass (default 32 on GPU, 1 on CPU) |
| `--workers N` | threads for disk read/write (default 8) |
| `--profile` | print a stage-by-stage wall-clock breakdown |

---

## Results

The organisers' 297-image paired test set, zero overlap with training (verified
by SHA-1 over the raw arrays). Scores are **final-5-epoch means**, not best-epoch — see
*How we decide what is real* below for why that distinction matters.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 (no denoising) | 20.455 | 0.5099 | 0.4655 |
| `pr50-w50-lp05-120` — 1,368,705 params (Round 2, superseded) | 23.632 | 0.60792 | 0.19287 |
| **`e1f_gate-120` — 1,346,360 params (SHIPPED)** | **23.8296** | **0.62104** | **0.18126** |
| gain over bicubic | **+3.374 dB** | **+0.111** | **−61%** |

Both model rows are scored by the same code on the same inputs on the same GPU
(`kla2/results/final.csv`). Paired per image across all 297, the shipped model
beats the Round 2 architecture by **+0.15181 dB** (t = 10.34), **+0.02724 SSIM**
(297/297 images) and **−0.03171 LPIPS** (t = 13.95), and **0 of 297** images
fall below bicubic. Full detail: [`docs/SHIP_E1F_GATE.md`](docs/SHIP_E1F_GATE.md).

> Everything below this heading was written for the Round 2 model and is kept as
> the record of how the design was arrived at. Where it says 1.37 M parameters or
> 23.632 dB it means `pr50-w50-lp05-120`, not what ships. The shipped model's
> numbers are the bold row above and in `docs/SHIP_E1F_GATE.md`.

The shipped configuration is `--p-real 0.5 --wide-p 0.5 --w-lpips 0.05`,
120 epochs (`train_submitted.py` reproduces it with no flags), chosen from **eleven** 120-epoch configurations scored in a single
pass on three axes: in-distribution accuracy, a nine-level noise sweep, and a
nine-axis out-of-distribution suite (`docs/queue/`).

It does **not** have the best in-distribution PSNR of the eleven — it is ninth,
and the whole field spans only 0.244 dB, so that ranking carries less
information than it appears to. What it has is the second-best perceptual
distance while staying level with the field on robustness. The five models
above it on PSNR all score **0.37–0.39 LPIPS against our 0.206**, because none
of them carries a perceptual loss term.

We are not claiming a single winner. § 12 of the engineering log lays out the
frontier honestly, including a selection claim we published and had to
withdraw.

Reproduce with
`python src/validate.py --data <data> --ckpt models/model.pt --baseline`.

**Round 1 for context:** 28.43 dB / 0.764 SSIM, +5.19 dB over bicubic. The gap
between rounds is the data, not the model — round-2 content is 3.1× finer, so
3.1× more of the signal lives above the Nyquist limit of the downsampled input.

---

## The four things the problem statement asks for, and where each is answered

| Ask | Where | Result |
|---|---|---|
| Recovery of fine spatial detail lost during downsampling | § 4 *Architecture*, *Results* | **+3.18 dB** over bicubic on the organisers' test set |
| Suppression of speckle noise across diverse image types | § 2 *The degradation*, § 3 *What we train on* | noise model settled by BIC (ΔBIC = 1118 for multiplicative over Poisson); positive gain across a 9× span of noise level |
| Broad generalization across multiple image distributions | *Generalisation* | positive on all ten labelled NFFA morphologies, mean **+4.01 dB**; +3.65 dB on a withheld content block; better than a real-only model on **all seven** OOD axes of the organisers' own set |
| Reduction in spatial resolution (the ×2 path) | *Quick start*, § 4 *Architecture* | native ×2 via PixelShuffle at the last block; 256→512 works with no retraining |

---

## Approach

### 1. What we measured about the data

Round 2 is a **different problem** from round 1, not a harder version of it. We
re-derived every property rather than assuming any carried over:

| | Round 1 | Round 2 |
|---|---|---|
| Content | natural photographs | SEM micrographs |
| Training pairs | 3,200 | 4,785 |
| Sizes present | mixed | all 256→128; **no 512×512 anywhere** |
| f90 — structure fineness | 0.0945 | **0.2913** (3.1× finer) |
| Normalisation | — | per-image min–max, exact `[0,1]` |
| Downsampling operator | exact 2×2 box average | exact 2×2 box average |
| Bicubic ×2 baseline | 23.23 dB | 20.22 dB |

`f90` is the radial frequency below which 90% of non-DC energy sits. Structure
3.1× finer means 3.1× more of the signal lives above the Nyquist limit of the
downsampled image — information genuinely destroyed, not merely obscured.

Dataset ids are **content-ordered**: `f90` runs from 0.3677 (ids 478–955) down
to 0.1233 (ids 2868–3345). That makes contiguous id blocks usable as genuine
out-of-distribution splits, which `--holdout-range` exploits.

> **A correction we owe the reader.** We initially claimed the id ordering
> revealed a fixed category stride of 478.5 images. It does not. Testing that
> hypothesis against the f90 series put the best offset at the 58th percentile
> of random offsets — indistinguishable from chance. The *ordering* is real and
> the blocks are usable as holdouts; the *stride* was pattern-matching on our
> part and is withdrawn. See `docs/ENGINEERING_LOG.md` § 11.1.

### 2. The degradation, reverse-engineered

Fitting the residual variance of KLA's own pairs against the ground truth gives
a three-term law — **adjusted R² = 0.9987** on 288 binned levels over 6.55 M
pixels — where `m` is the 2×2 block mean and `v` the within-block variance:

```
var(residual)  =  σ_add²  +  σ_mul²·m²  +  c·v

σ_add ∈ [0.000, 0.065]   additive floor
σ_mul ∈ [0.155, 0.258]   speckle — multiplicative, grows with brightness
c     ∈ [0.026, 0.234]   detail-dependent term
```

**Is the brightness dependence speckle or shot noise?** This matters: Poisson
shot noise has variance linear in the mean, multiplicative speckle quadratic,
and they call for different variance-stabilising transforms. We fitted four
models to identical bins over 6.55 M pixels (`tools/noise_physics.py`):

| Model | form | adj R² | BIC |
|---|---|---|---|
| **M1 multiplicative** | `a + b·m² + c·v` | **0.9987** | **−1814.5** |
| M2 Poisson | `a + p·m + c·v` | 0.9370 | −696.2 |
| M3 both terms | `a + p·m + b·m² + c·v` | 0.9989 | −1849.5 |
| M4 no brightness term | `a + c·v` | 0.1808 | +37.7 |

288 usable bins over 6,553,600 low-resolution pixels. Raw output:
`docs/tool_output/noise_physics.txt`.

ΔBIC between M1 and M2 is **1118.3** — decisive for multiplicative (ΔBIC > 10 is
conventionally "very strong"). M3 fits marginally better but drives the linear
coefficient to **−0.00196**, i.e. negative, which is physically meaningless for a
shot-noise term — it is absorbing curvature, not counting photons. We keep M1. This is why the
generator multiplies rather than draws Poisson counts, and why the network's
stem offers `√x` (Anscombe-style) as one option rather than being built on it.

Three further changes from the round-1 generator, each driven by a measurement:

* **Added the `c·v` term.** Detail-dependent noise is 4.4% of total variance on
  SEM against 1.8% on photographs — the noise tracks edges here in a way it did
  not before.
* **Removed the blur** from the *narrow* family. Measured harmful: residual
  error rose 0.01074 → 0.01235. (It returns, deliberately, in the wide family —
  see § 3.)
* **Gaussian → Gamma (k = 14).** The real residual is spatially white (lag-1
  correlation −0.055 / −0.053) but decidedly not Gaussian: skew +0.43, excess
  kurtosis +0.62.

Calibration check — refitting our own synthetic output with the same estimator:

| | σ_add | σ_mul |
|---|---|---|
| KLA's real pairs | 0.0414  [0.000–0.085] | 0.1863  [0.139–0.238] |
| Our generator, **narrow** family | 0.0415  [0.000–0.080] | 0.1986  [0.146–0.251] |
| Our generator, **training** mixture | 0.0507  [0.000–0.097] | 0.2192  [0.117–0.342] |

The narrow family brackets reality — that is the calibration test. The training
mixture is **visibly wider on purpose**, and that width is what § 3 shows buys
out-of-distribution robustness. `tools/calibrate.py`; raw output in
`docs/tool_output/calibrate.txt`.

### 3. What we train on — and the result that reversed our answer

**First answer (wrong, and we shipped it for a week).** On the held-out split,
100% real pairs beat every synthetic mixture:

| Run | Training data | PSNR | SSIM |
|---|---|---|---|
| **r2-preal1** | **100% real pairs** | **23.267** | **0.612** |
| r2-newgen | 70% rebuilt synthetic | 23.218 | 0.593 |
| r2-long120 | 70% old (round-1) synthetic | 23.134 | 0.593 |

The rebuild worked as a rebuild — +0.084 dB over the old generator, eight times
our noise floor — and still lost to real data. So we set `p_real = 1.0`.

**Second answer.** Then we swept the noise level instead of holding it fixed.
`tools/noise_sweep.py` regenerates the *same* held-out images at nine
multiplicative-noise levels and scores every model on identical inputs. Gain
over bicubic, in dB (`docs/noise_sweep_*.csv`):

| σ_mul | 0.05 | 0.10 | 0.15 | **0.19** | 0.25 | 0.30 | 0.35 | 0.40 | **0.45** |
|---|---|---|---|---|---|---|---|---|---|
| r2-preal1 (100% real) | +0.86 | +1.47 | +2.04 | **+2.40** | +2.45 | +2.15 | +1.89 | +1.79 | **+1.88** |
| r2-newgen (70% synth) | +0.98 | +1.52 | +2.17 | **+2.70** | +3.59 | +4.30 | +4.74 | +4.22 | **+2.65** |
| r2-long120 (70% synth) | +0.82 | +1.33 | +1.99 | **+2.45** | +2.96 | +3.63 | +4.49 | +5.24 | **+5.71** |

σ ≈ 0.19 is where KLA's own data sits. The three models are within 0.3 dB of
each other there — which is exactly why the first comparison could not see the
difference. Three steps out, the real-only model is **3.8 dB worse** than a
synthetic-trained one. It has learned the noise level it was shown, not the
inverse problem.

The real-only model does not merely stop improving; it **turns over** at
σ = 0.25 and declines. A model that generalised would keep gaining, because
there is more noise to remove. That turnover is the signature of a model that
has memorised a noise amplitude.

**The objection we had to answer first.** `tools/noise_sweep.py` builds its test
inputs with *our own* generator, which structurally favours a synthetic-trained
model — we flagged this ourselves in `docs/ENGINEERING_LOG.md` § 4.4 and nearly
threw the sweep out for it. Two things rescue it. First, the real-only model is
*not* penalised at the levels where the generator's bias would bite hardest: at
σ = 0.05–0.19 it beats `r2-long120`, which was trained on that very generator.
A test that merely rewarded generator-fit would not produce that ordering.
Second, and decisively, the conclusion is confirmed on inputs our generator
never touched — the ten NFFA morphologies and the unseen-content block below.
The sweep is where we *found* the effect; it is not where we certify it.

**Third answer, which is what we ship.** If breadth of the training degradation
is what buys robustness, generate breadth on purpose. `degrade_wide()` in
`src/degrade.py` samples a family deliberately *wider* than the observed one:

| parameter | observed in KLA data | wide family |
|---|---|---|
| σ_add | 0.000 – 0.065 | 0.000 – 0.180 |
| σ_mul | 0.155 – 0.258 | 0.060 – 0.500 |
| c | 0.026 – 0.234 | 0.000 – 0.450 |
| Gamma k | 14 | 4 – 60 |
| optical blur | absent | σ 0.3–1.1 on 30% of samples |
| downsampling kernel | exact 2×2 box | box, or Gaussian-weighted on 25% |

Blur is applied to the **clean ground truth before sampling noise**, not to the
degraded output — optics act on the image, not on the sensor noise, and
blurring the output would correlate noise that is measurably white.

`wide_p` is the fraction **of the synthetic draws** taken from the wide family,
not of all draws, so `--p-real 0.5 --wide-p 0.5` gives **50% real / 25% wide /
25% narrow**. Real pairs stay in because they are the only ground truth about
what KLA's sensor actually does; the wide family is what makes the model
robust to a sensor it has never seen.

**What it costs.** On the in-distribution split the shipped model is 0.19 dB
*below* the real-only model. We paid that knowingly. The next section is what
we bought.

### 4. Architecture (`src/model.py`)

A residual CNN that does all its work at low resolution and upsamples only in
the final block via PixelShuffle — 4× cheaper than upsample-first designs, which
matters because inference time is scored.

* **Global bicubic skip** — the network predicts only the *correction* to a
  cheap baseline, so it starts at the bicubic score and converges fast.
* **Variance-stabilising stem** — a parameter-free layer feeding the first
  convolution raw, `√x` and `log(1+x)` views, so it can pick the representation
  in which signal-dependent noise is closest to uniform.
* **No BatchNorm** — well established to hurt super-resolution.

1.37 M parameters, 5.5 MB. `tools/capacity_sweep.py` maps the accuracy/size
frontier across a **78.4× span** of parameter count: the curve is a straight
line at **0.045 dB per doubling (R² = 0.992), with no knee anywhere**
(`docs/capacity_sweep.csv`) — 0.291 dB total across the whole span.
A model starved of capacity climbs steeply then flattens; ours does neither, so
we are not capacity-limited. The bottleneck is information the downsampling
destroyed. That is also the answer to "why not a transformer" — 2.7× the
parameters bought +0.026 dB, inside the noise floor.

### 5. Loss (`src/losses.py`) — and why LPIPS is now on

Charbonnier + SSIM + gradient + **LPIPS at weight 0.05**. No GAN.

We resisted this for most of the project, on a real concern: perceptual and
adversarial losses work by *synthesising plausible texture*, and invented
texture on an inspection image is a fabricated defect. Then we measured both
sides of the trade.

The clean ablation — same mixture, same length, the LPIPS term the only
difference, on the organisers' 297-image test set:

| | LPIPS ↓ | PSNR | high-frequency energy ratio |
|---|---|---|---|
| wide mixture, no LPIPS term (`pr50-w50`) | 0.355 | 23.763 | — |
| **wide mixture + w_lpips 0.05 (shipped)** | **0.193** | 23.632 | **0.421** |
| bicubic ×2 (cannot fabricate, by construction) | 0.466 | 20.455 | 0.375 |

**46% better perceptual distance for 0.131 dB**, and the advantage holds at all
nine levels of the noise sweep, not only at this one. Put the other way, the
same term takes LPIPS from **0.355 to 0.193 — 46% — for 0.131 dB** (`pr50-w50` vs
`pr50-w50-lp05`, identical mixture, on the organisers' test set).

And the hallucination question
is settled by measurement rather than by abstinence: `tools/hf_energy.py`
integrates output energy *above the Nyquist frequency of the input* — the band
where any content is necessarily invented, because the input cannot contain it.
On the organisers' 297-image test set the model sits at **0.421** against
bicubic's **0.375**. Bicubic interpolation is mathematically incapable of
inventing structure, so its ratio is the floor for "no fabrication"; we are 7%
above it and still far below 1.0, the line above which a model puts more energy
into that band than the ground truth contains. **Zero of our outputs cross that
line, against fourteen of bicubic's** — those fourteen are ringing on very
smooth images, which is what the metric does when the denominator is near zero.
Raw output: `docs/tool_output/hf_energy.txt`.

The gradient term sharpens edges only where the ground truth has edges, which
is the mechanism that keeps the perceptual term honest.

---

## Generalisation

Four independent tests, none of which reuses the training distribution.

### Ten labelled morphologies from a different source

KLA's round-2 images are drawn from **NFFA-EUROPE**, a public SEM corpus of
18,577 micrographs — and NFFA ships **its own morphology labels**, ten named
classes. We took the natives (1024×768, cropped above the instrument data bar),
applied the *observed* degradation, and scored per class
(`tools/per_category.py`, `docs/per_category.csv`):

| category | bicubic | ours | gain | SSIM | LPIPS |
|---|---|---|---|---|---|
| Patterned surface | 20.49 | 25.76 | **+5.28** | 0.571 | 0.199 |
| Tips | 20.20 | 25.02 | +4.83 | 0.464 | 0.232 |
| MEMS devices / electrodes | 20.13 | 24.51 | +4.38 | 0.451 | 0.254 |
| Fibres | 21.67 | 25.85 | +4.18 | 0.621 | 0.226 |
| Particles | 23.83 | 27.90 | +4.06 | 0.577 | 0.202 |
| Powder | 19.82 | 23.78 | +3.96 | 0.622 | 0.225 |
| Nanowires | 20.90 | 24.73 | +3.82 | 0.719 | 0.201 |
| Porous sponge | 19.68 | 23.47 | +3.79 | 0.571 | 0.222 |
| Films / coated surface | 19.11 | 22.68 | +3.57 | 0.480 | 0.299 |
| Biological | 19.63 | 21.90 | **+2.26** | 0.446 | 0.277 |
| **mean** | | | **+4.01** | | |

Every class is positive. The mean gain (+4.01 dB) is *higher* than on the
organisers' own test set (+3.18), and LPIPS stays in 0.199–0.299 against 0.193 in
distribution — so the perceptual gain is not an artefact of one image source.
The 3.01 dB spread tracks morphology exactly as the f90 analysis predicts:
regular geometric structure is easiest, soft organic structure hardest.

### Unseen content blocks

A contiguous id block, held out entirely (`--holdout-range`), so the model has
never seen that content class:

| model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic ×2 | 20.652 | 0.521 | 0.458 |
| **shipped** | 24.302 | 0.642 | **0.182** |
| r2-preal1 (real-only) | **24.562** | 0.664 | 0.338 |

**+3.65 dB on content the model has never seen.** The real-only model — which
*was* trained on this block — is 0.26 dB ahead on PSNR and **86% worse** on
perceptual distance. This is the same trade as § 3, measured on content the model was
never trained on.

### The OOD suite, run on the organisers' own test set

`tools/ood_suite.py` re-degrades the same images along several axes at once and
scores gain over bicubic on identical inputs, reporting the mean **and the worst
axis** — a model that is excellent on eight axes and negative on one is not
robust. Run against the organisers' 297 images
(`docs/tool_output/ood_testset.txt`):

| axis | bicubic | shipped | 100% real pairs |
|---|---|---|---|
| noise 0.05 | 23.09 | +1.07 | +1.01 |
| noise 0.19 | 19.79 | +2.89 | +2.66 |
| **noise 0.40** | 15.54 | **+5.59** | **+2.09** |
| blur 0.4 | 19.82 | +2.84 | +2.64 |
| blur 0.8 | 19.70 | +2.68 | +2.57 |
| blur 1.2 | 19.44 | +2.63 | +2.56 |
| soft (non-box) kernel | 19.74 | +2.89 | +2.66 |
| **mean / worst axis** | | **+2.94 / +1.07** | +2.31 / +1.01 |

Better on **every axis**, and by **3.50 dB** at high noise — on the organisers'
own images, not ours. The two models were scored in separate runs, which is
normally forbidden here; the bicubic column is byte-identical across both
(23.09 / 19.79 / 15.54 / 19.82 / 19.70 / 19.44 / 19.74), which is the checksum
that makes the comparison legitimate.

> **A bug this run exposed.** The suite also defines two *content* axes by
> training-set id range (1850–2134 and 0–1534). The test set has ids 0–296, so
> those ranges came back empty and produced a `NaN` row that propagated into the
> MEAN. The tool now skips out-of-range blocks and reports which it skipped,
> rather than printing a mean that is not a number. The seven axes above are the
> ones that are meaningful on this dataset.

### The 256 → 512 path, with zero training at that scale

The round-2 training set contains **no** 512×512 pairs. On 778 held-out
512×512 pairs built from NFFA natives (`tools/make_512.py`), the shipped weights
score:

| 778 held-out 512×512 pairs | PSNR | SSIM | LPIPS |
|---|---|---|---|
| **ours** | **25.392** | **0.576** | **0.236** |
| bicubic ×2 | 20.877 | 0.410 | 0.538 |
| gain | **+4.52 dB** | **+0.166** | **56% lower** |

That is a *larger* gain than we achieve at 256×256 (+3.18 dB), at a scale the
network has never been trained on. All 778 pairs are scored, not a subset.

This is not luck. The network is fully convolutional and operates at low
resolution, so a 256×256 input is simply more patches; and the degradation is
*local*, depending only on a 2×2 block, so it is identical at every scale.
Scale-invariant architecture plus a local degradation model means the 512 path
needs no training data.

An earlier version of this file quoted **+4.91 dB** here. That figure was
measured with the `r2-newgen` checkpoint during the scale study, not with the
shipped weights, and is withdrawn. The shipped model is 0.46 dB lower on PSNR
and **41% better on LPIPS** (0.232 vs 0.394) — the same trade as everywhere
else in this repo, now confirmed at a scale it never saw in training.

---

## How we decide what is real: the measured noise floor

Before trusting any result we measured what *nothing* looks like. Two runs
differing only in random seed land **0.010 dB** apart on a final-5-epoch mean.
Selecting the best epoch instead inflates that apparent spread to 0.055 dB —
which is why every number in this repo is a final-5 mean, and why anything under
0.010 dB is reported as null.

| Change tested | Effect on PSNR | Verdict |
|---|---|---|
| 9× more training data (500 → 4,585 images) | +0.006 dB | null |
| 2.7× parameters (1.37 M → 3.74 M) | +0.026 dB | null |
| 1.8× receptive field, parameters matched | +0.008 dB | null |
| Random seed alone | 0.010 dB | *the floor itself* |
| Degradation recalibration on its own | −0.016 dB | null |
| Content-balanced sampling (p ∝ f90) | −0.090 dB | null |
| Gaussian-weighted downsampling in the wide family | +0.01 dB | null |
| 8× test-time augmentation | +0.018 dB, 7.1× slower | null, and expensive |
| Pinned staging buffers + CUDA streams | +0.04 s on a 1.94 s spread | null, bit-identical output |
| `channels_last` layout fix | −0.06 s on a 0.73 s spread | null, bit-identical output |
| 3× training length (40 → 120 epochs) | +0.030 dB | marginal |
| **Real pairs instead of synthetic (in-distribution)** | **+0.133 dB** | **real effect** |
| **Wide degradation family (3 σ-steps out)** | **+3.8 dB** | **the effect that decided the model** |

Ten interventions in twelve failed to clear the floor. The two that cleared it
were both data decisions, and they point in opposite directions — which is the
whole story of this round.

**What actually dominates the score is content**, not anything we control:

| Held-out content block | f90 | PSNR |
|---|---|---|
| Coarsest structure (ids 2868–3345) | 0.1233 | 26.664 dB |
| Finest structure (ids 478–955) | 0.3677 | 21.274 dB |
| **Spread from content alone** | | **5.39 dB** |

That spread is 40× larger than any architectural change we measured, which is
why robustness across content — not peak score — is what this pipeline targets.

Full experimental record, including the failures:
[`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md).

---

## Repository layout

```
run.py                  ← 1. EVALUATION SCRIPT: python run.py <in-dir> <out-dir>
train_submitted.py      ← 2. TRAINING SCRIPT: reproduces models/model.pt, no flags
outputs/                ← 3. DENOISED TEST OUTPUTS: all 297, plus preview.png
requirements.txt        ← 4. ENVIRONMENT SPEC: complete pip freeze
models/model.pt         ← trained weights, 1.35 M params, 5.4 MB (e1f_gate)
requirements-inference.txt ← the two packages run.py actually imports
src/
  degrade.py            narrow (calibrated) + wide (OOD) degradation families
  dataset.py            pair loading, real/wide/narrow mixing, random + block splits
  model.py              the network
  train.py              training loop
  losses.py             Charbonnier / SSIM / gradient / LPIPS
  metrics.py            PSNR, SSIM, LPIPS
  validate.py           score a checkpoint, append a row to results.csv
tools/
  check_data.py         verify dataset layout before training
  calibrate.py          does our synthetic damage match KLA's real damage?
  noise_physics.py      speckle vs Poisson, settled by BIC
  capacity_sweep.py     accuracy vs parameter count vs latency frontier
  noise_sweep.py        robustness across synthetic noise levels
  ood_suite.py          nine-axis out-of-distribution scorecard
  per_category.py       score per NFFA morphology label
  hf_energy.py          hallucination guard: energy above the input's Nyquist
  real_noise_bins.py    score checkpoints on REAL data binned by noise or f90
  fetch_nffa.py         stream NFFA-EUROPE categories without the 84 MB download
  make_512.py           build 512×512 pairs from NFFA natives
  categories.py         per-image f90 / noise statistics
  bench.py              end-to-end timing, alternating reps
  scorecard.py          batch-score a directory of checkpoints
  package_check.py      end-to-end packaging + output-contract verification
  reship.py             deploy a checkpoint and re-measure everything it affects
  tonight.py            unattended multi-experiment driver
  preview.py            before/after figure
cloud/
  setup.sh go.sh        one command from a fresh GPU pod to a finished queue
  run_experiments.sh full120.sh
docs/
  ForgeX_KLA_PS01_Round2.pdf / .pptx   the submitted deck
  ENGINEERING_LOG.md    every experiment, including the ones that failed
  tool_output/          raw output of calibrate, noise_physics, hf_energy, ood_suite
  queue/                all eleven checkpoints scored on both test sets
  superseded/           result files scored against a different test set — do not quote
  results.csv           one row per scored checkpoint
  per_category.csv      the ten-morphology generalisation table
  capacity_sweep.csv    the 78× capacity frontier
  noise_sweep_*.csv     the sweeps behind the § 3 reversal
  noise_sweep_queue.csv the three finalist configurations, identical inputs
  bench_results.txt     timing runs
outputs/                restored test-set images
```

---

## Reproducing training

Expected data layout:

```
<data>/GT/000000.npy        256×256 float32 in [0,1]
<data>/NoisyLR/000000.npy   128×128 float32, may fall outside [0,1]
```

```bash
python tools/check_data.py --data <data>       # verify layout
python tools/calibrate.py  --data <data>       # verify degradation model
python tools/noise_physics.py --data <data>    # speckle vs Poisson
python src/train.py --data <data> --smoke      # ~5 s, CPU, checks plumbing

# the SHIPPED configuration
python src/train.py --data <data> --out runs/pr50-w50-lp05-120 --amp \
                    --epochs 120 --iters 500 --batch 32 \
                    --ch 64 --nb 16 \
                    --p-real 0.5 --wide-p 0.5 --w-lpips 0.05 \
                    --seed 0 --split-seed 0

python src/validate.py --data <data> --ckpt runs/pr50-w50-lp05-120/last.pt --baseline
python tools/noise_sweep.py --data <data> --ckpt runs/pr50-w50-lp05-120/last.pt
python tools/ood_suite.py   --data <data> --ckpt runs/pr50-w50-lp05-120/last.pt
python tools/hf_energy.py   --data <data> --ckpt runs/pr50-w50-lp05-120/last.pt
```

`--seed` and `--split-seed` are deliberately separate: the validation split must
stay fixed while training randomness varies, or a seed-variance experiment
compares two models on two different validation sets and measures nothing.

Interrupted run? `--resume runs/<name>/last.pt` picks up exactly where it
stopped, optimiser and schedule included.

**Hardware:** NVIDIA RTX 4050 Laptop, 6 GB (75 W), Windows 11, and an RTX 4090
pod for the nine-run selection queue · **Training time:** 4.3 h on the 4050 /
28 min on the 4090 (120 epochs, 60,000 steps) · **Peak VRAM:** 0.95 GB

Cross-machine reproducibility was checked, not assumed: the shipped weights
score **23.632 / 0.60791 / 0.19288** (the Round 2 checkpoint) on the RTX 4090 pod and **23.632 / 0.60791 /
0.19288** on the RTX 4050 laptop — identical to five decimals.

---

## Notes for reviewers

* `run.py` is **fully self-contained** — the network definition is inlined, so
  it imports only `torch` and `numpy` and depends on no other file in this repo
  except the weights. It downloads nothing at runtime.
* It resolves `models/model.pt` relative to its own location, so it runs from
  any working directory.
* Outputs pass through `nan_to_num` and are clamped to `[0,1]`, so the
  finite-value and range guarantees hold regardless of input.
* Nothing hardcodes an input resolution; 256×256 → 512×512 works unchanged.
* `run.py` overlaps its fixed costs deliberately: CUDA context initialisation
  runs on a background thread, inputs are read on a thread pool, weights are
  memory-mapped, and outputs are written by a writer pool. `--profile` prints
  the breakdown. On a warm machine roughly 72% of wall-clock is process startup
  (`import torch` plus CUDA init) rather than the network, so startup cost
  amortises over the size of the test set. We tried to beat it a second time
  with pinned staging buffers and CUDA streams, and a third time with a
  `channels_last` layout fix; both differences fell inside the run-to-run
  spread and both produced bit-identical output, so neither shipped. The
  measurements are in `docs/bench_results.txt`.

## What we still do not know

Stated before anyone has to ask.

* **We cannot verify the 512×512 path against KLA's own 512 data**, because the
  round-2 training set contains none. Our 512 evidence is built from NFFA
  natives, which is the same corpus but not the same acquisition.
* **The wide degradation family is a guess about breadth, not a measurement.**
  We know it helps three σ-steps out; we do not know that its bounds match
  whatever the round-2 test set actually contains.
* **`soft_p` (Gaussian-weighted downsampling) contributes nothing measurable**
  — +0.01 dB, inside the floor. It stays in the code because the shipped weights
  were trained with it and removing it would break reproducibility of
  `models/model.pt`, not because it earns its place.
* **Biological morphology is our weakest class** at +2.30 dB against a +4.06
  mean. Soft organic structure has the least regular high-frequency content, so
  there is least for a prior to latch onto.
* **We have not tested on a genuinely different microscope.** Every image we
  have seen, KLA's and NFFA's alike, comes from the same corpus.
* **ONNX export is untested.** It is the only remaining lever on the 72%
  startup cost and we ran out of time to validate numerical equivalence.

## References

1. B. Lim et al., *Enhanced Deep Residual Networks for Single Image
   Super-Resolution*, CVPRW 2017. (residual SR backbone, no BatchNorm)
2. W. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
   Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016. (PixelShuffle)
3. W.-S. Lai et al., *Deep Laplacian Pyramid Networks for Fast and Accurate
   Super-Resolution*, CVPR 2017. (Charbonnier loss)
4. H. Zhao et al., *Loss Functions for Image Restoration with Neural Networks*,
   IEEE Trans. Computational Imaging, 2017. (L1 + MS-SSIM loss design)
5. R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric*, CVPR 2018. (LPIPS)
6. F. J. Anscombe, *The Transformation of Poisson, Binomial and
   Negative-Binomial Data*, Biometrika, 1948. (variance stabilisation)
7. G. Schwarz, *Estimating the Dimension of a Model*, Annals of Statistics,
   1978. (BIC, used to settle the noise model)
8. R. Aversa et al., *The First Annotated Set of Scanning Electron Microscopy
   Images for Nanoscience*, Scientific Data 5, 180172, 2018. (NFFA-EUROPE)
