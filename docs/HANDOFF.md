# HANDOFF — everything you need to know about this project

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> This is a historical record. The current shared-head release is documented in [RELEASE_STATUS.md](RELEASE_STATUS.md).
>
> Current: [RELEASE_STATUS.md](RELEASE_STATUS.md)


**Read this first.** If you are a new teammate, a reviewer, or an AI assistant
being brought onto ForgeX, this single file is the context transfer. Every number
here was verified against the CSVs in `docs/` on 16 Sep 2026. Where a number is
uncertain, it says so.

Do not quote a figure about this project that is not in this file or in a CSV
under `docs/`. That rule exists because it has already been broken twice.

---

## 0. The one-paragraph version

Blind restoration of degraded SEM images for semiconductor inspection
(KLA PS01, SEMICON India 2026). Input is a 128x128 noisy low-resolution float32
array in [0,1] as `.npy`; output is 256x256 — so the task is joint denoising and
2x super-resolution, with the degradation parameters unknown at test time. Our
model is a 1.37 M-parameter EDSR-style residual CNN with a parameter-free
variance-stabilising stem and a global bicubic skip. It scores **23.632 dB PSNR /
0.60791 SSIM / 0.19288 LPIPS** on the organisers' 297-image test set at
**2.168 ms/image**. Team ForgeX: Abishek SR, Anmol BA, Hardik (VIT Vellore).

---

## 1. The problem, precisely

| | |
|---|---|
| Input | `.npy`, float32, [0,1], 128x128, single channel |
| Output | `.npy`, float32, [0,1], 256x256 |
| Task | denoise + 2x upscale, jointly |
| Blind? | yes — degradation parameters are not given at test time |
| Test set | 297 images, provided by the organisers |
| Entry point | `run.py <input_dir> <output_dir>` |

---

## 2. The degradation model — this is the actual contribution

We did not guess the noise. We fitted it.

Take the residual between a real noisy-LR image and the clean image downsampled.
Bin pixels by the 2x2 block mean `m` and the within-block variance `v`, then fit
the residual variance. On **400 images, 6,553,600 low-res pixels, 288 usable bins**:

| model | what it assumes | adj R² | BIC |
|---|---|---|---|
| M1 `a + b·m² + c·v` | **speckle / multiplicative** | 0.9987 | **−1814.5** |
| M2 `a + p·m + c·v` | Poisson / shot noise | 0.9370 | −696.2 |
| M3 `a + p·m + b·m² + c·v` | both | 0.9989 | −1849.5 |
| M4 `a + c·v` | no signal dependence | 0.1808 | +37.7 |

**Speckle beats Poisson by ΔBIC = 1118.3.** A BIC gap above 10 is decisive; this
is a hundred times that. The noise on these images is **multiplicative**, not shot
noise — which is the opposite of what most people assume about electron microscopy
and is the single most defensible claim we make.

**The M3 subtlety — know this before a judge finds it.** M3 (both terms) has a
slightly better BIC than M1, by 35. But look at its fitted coefficients:
`[a=0.00043, p=−0.00196, b=0.03844, c=0.46318]`. The Poisson coefficient `p` is
**negative**. Negative shot noise is physically meaningless. M3 buys its 35 BIC
points by fitting noise in the fit, not physics. We ship M1 because M1 is the
model whose parameters are all physically admissible. Say exactly that.

Fitted M1 coefficients: `σ_add² = 0.00004`, `σ_mul² = 0.03649`, `c = 0.45944`.

**The `c·v` term proves the degradation ordering.** `v` is the within-block
variance of the *high-resolution* image. It can only influence the residual if the
noise was applied **before** downsampling. So the pipeline is
noise → downsample (pipeline A), not downsample → noise (pipeline B). We did not
have to ask the organisers; the data told us. Box-averaging four iid samples gives
σ²/4, which is why the residual variance at the LR scale is what it is.

We model the multiplicative field as a **shifted Gamma with k = 14**.

---

## 3. The architecture

```
input 128x128
  -> variance-stabilising stem: concat(x, sqrt(x), log1p(x))   [PARAMETER-FREE]
  -> conv to 64 channels
  -> 16 x EDSR residual blocks (res_scale = 0.1)
  -> PixelShuffle x2
  -> conv to 1 channel, ZERO-INITIALISED tail
  -> + global bicubic upsample of the input, computed in fp32
output 256x256
```

`Restorer(ch=64, nb=16, scale=2, res_scale=0.1)` — **1,368,705 parameters**
(1.369 M as reported everywhere). Roughly `params ≈ 18·nb·ch²`.

Three choices are load-bearing and each has a reason:

- **The stem is parameter-free.** Because the noise is multiplicative, its variance
  scales with the signal. `sqrt` and `log1p` are the classical variance-stabilising
  transforms for exactly that case. Giving the network all three views costs zero
  parameters and hands it the physics instead of making it learn it.
- **The zero-init tail plus bicubic skip** means the network starts as *exactly*
  bicubic interpolation and can only learn the residual. This is a range/null-space
  decomposition: bicubic is a valid right-inverse `A⁺` of box-averaging (pixel
  replication), so the network only has to supply the null-space component.
  Training never has to rediscover interpolation.
- **`res_scale = 0.1`** is the EDSR stability trick; without it deep residual stacks
  diverge at higher learning rates.

---

## 4. The numbers — VERIFIED 16 Sep 2026

### 4.1 Shipped model

`pr50-w50-lp05-120`, sha1 **`e208d13d62b3ded7b19954c273e80355e718a5f4`**

| metric | value | where |
|---|---|---|
| PSNR | **23.632 dB** | `docs/queue/testset_results.csv` |
| SSIM | **0.60791** | same |
| LPIPS | **0.19288** | same |
| ms/image | **2.168** | same |
| parameters | 1.369 M | same |

Trained with `--p-real 0.5 --wide-p 0.5 --w-lpips 0.05`, 120 epochs.

### 4.2 The full checkpoint field (organisers' 297-image test set)

| tag | PSNR | SSIM | LPIPS |
|---|---|---|---|
| pr50-w50-120 | **23.7626** | 0.61915 | 0.35518 |
| pr50-w00-120 | 23.7546 | **0.61954** | 0.35333 |
| pr30-w00-120 | 23.7567 | 0.61343 | 0.36773 |
| pr30-w50-120 | 23.7435 | 0.61221 | 0.36763 |
| pr30-w100-120 | 23.7403 | 0.61074 | 0.37485 |
| lp02-120 | 23.6846 | 0.60700 | 0.22007 |
| pr30-w50-hard-120 | 23.6386 | 0.61129 | 0.33984 |
| loss-ss30-120 | 23.6339 | 0.61563 | 0.36949 |
| **pr50-w50-lp05-120  (SHIPPED)** | **23.6320** | **0.60791** | **0.19288** |
| loss-lp05-120 | 23.6080 | 0.60369 | 0.19750 |
| loss-lp15-120 | 23.4786 | 0.59467 | **0.18668** |

**The shipped model is 9th of 11 on PSNR and 8th on SSIM.** It ships because it is
2nd on LPIPS. Whether that is the right call depends entirely on the organisers'
metric weighting, **which we do not know**. It wins only if LPIPS carries roughly
35% or more of the score.

The whole field spans **0.284 dB** on PSNR (23.4786 to 23.7626). That is the
number to quote for "spread across the checkpoint field" — do not confuse it with
the capacity-sweep span in 4.3.

### 4.3 Capacity scaling — CORRECTED

Two files measure this on two different sets. Both are legitimate; quote the right
one for the right claim.

| span 17,449 → 1,368,705 params (78.4x) | ΔPSNR | per doubling | source |
|---|---|---|---|
| held-out val split, 40 epochs | 0.29 dB | 0.046 dB | `docs/capacity_sweep.csv` |
| organisers' 297 test set | 0.39 dB | 0.061 dB | `docs/timesweep.csv` |

**Earlier drafts quoted "0.284 dB across 78.4x". That was wrong** — it was the
checkpoint-field spread (4.2) accidentally attached to the capacity claim. The
correct capacity figures are above. The conclusion is unchanged and if anything
stronger: **a 78-fold increase in parameters buys less than 0.4 dB.** The model is
not the bottleneck.

Full capacity ladder on the test set (`timesweep.csv`, phase A):

| params | PSNR |
|---|---|
| 17,449 | 23.2667 |
| 30,753 | 23.3002 |
| 49,313 | 23.3615 |
| 110,257 | 23.3944 |
| 195,393 | 23.4607 |
| 343,361 | 23.5077 |
| 1,368,705 | 23.6527 |

Monotone, no knee — still inside the power-law regime, not saturated. (A *knee*
would indicate saturation. A straight line does not.)

### 4.4 Robustness — the biggest effect we found

Nine-level noise sweep, gain over bicubic at σ_mul = 0.45
(`docs/noise_sweep_queue.csv`; bicubic = 14.53 dB there):

| config | PSNR @ σ0.45 | gain over bicubic |
|---|---|---|
| `--wide-p 0.5` (wide degradation family) | 20.61 | **+6.08 dB** |
| `--wide-p 0.5 --w-lpips 0.05` | 20.44 | +5.91 dB |
| `--wide-p 0.0` (narrow family) | 16.89 | **+2.36 dB** |

**A 3.7 dB difference, from one training flag.** Against 0.39 dB from a 78x
capacity change. This is the headline: *the degradation distribution you train on
matters an order of magnitude more than the size of the network.*

Critically, **robustness splits on `--wide-p`, not on `--p-real`.** It is the
*width* of the synthetic degradation family that buys generalisation, not the
proportion of real data.

### 4.5 Generalisation, eight axes (organisers' test set)

`docs/tool_output/reship_ood_testset.txt` — mean gain **+2.93 dB**, worst axis
**+1.07 dB**. No axis where we lose to bicubic.

| axis | bicubic | our gain |
|---|---|---|
| noise 0.05 | 23.09 | +1.07 |
| noise 0.19 | 19.79 | +2.89 |
| noise 0.40 | 15.54 | **+5.59** |
| blur 0.4 | 19.82 | +2.84 |
| blur 0.8 | 19.70 | +2.68 |
| blur 1.2 | 19.44 | +2.63 |
| soft kernel | 19.74 | +2.89 |
| content fine | 19.79 | +2.89 |

The model helps *most* where the input is worst. That is the right shape for an
inspection tool.

### 4.6 Per-morphology (ten labelled NFFA categories)

`docs/per_category.csv` — gain over bicubic ranges **+2.26 dB (Biological)** to
**+5.28 dB (Patterned surface)**. Semiconductor-relevant categories
(Patterned surface +5.28, MEMS +4.38, Tips +4.83, Fibres +4.18) are at the top.
Biological is at the bottom and is the least relevant to inspection.

### 4.7 Inference time

`docs/timesweep.csv`, 297 images, 5 reps, median seconds end-to-end:

| configuration | median s | vs shipped |
|---|---|---|
| **SHIPPED** | **4.692** | — |
| + `--batch 64` | 4.737 | +0.045 |
| + `--batch 128` | 4.803 | +0.111 |
| + `CUDA_MODULE_LOADING=LAZY` | 4.810 | +0.118 |
| + `cudnn.benchmark = True` | 7.661 | **+2.969** |
| all three levers | 7.904 | +3.212 |
| smallest model (49,313 params) + all levers | 6.637 | +1.945 |

**Every startup lever is a null or a loss.** `cudnn.benchmark` costs three seconds
of autotuning it can never earn back on a single fixed 128x128 input shape, and it
perturbs the output (LPIPS moves 0.19287 → 0.19288). It stays off.

The model is roughly a third of end-to-end time; the rest is I/O and process
startup. Shrinking the network does not shrink the wall clock proportionally.

---

## 5. Traps — read before touching anything

### 5.1 The 28.43 dB number is poison

`docs/results.csv` line 5 reads `v1, runs\v1\best.pt, 76, 28.4272, 0.74673,
0.30857`. **This is a ROUND 1 number on a DIFFERENT test set.** The exact same
checkpoint, scored on the round-2 test set, gets **23.1721 dB**
(`results.csv`, tag `v1-round1-model`). A 5.25 dB illusion.

This number has leaked into two research briefs already. The `3.74 M` parameter
count that travels with it is from `v2`, a capacity ablation — not from any shipped
model. If anyone says "28.43", stop them.

### 5.2 Four files, 5,493,743 bytes each, three of them wrong

Identical size, similar names, different models. Only the sha1 distinguishes them.
See `docs/CHECKPOINTS.md`. Always run:

```
python tools/verify_shipped.py
```

Known hashes:

| sha1 | model | verdict |
|---|---|---|
| `e208d13d…` | pr50-w50-lp05-120 | **CORRECT** |
| `92f45544…` | r2-preal1 | *** WRONG *** |
| `8dc5b0a9…` | loss-lp05-120 | *** WRONG *** |
| `a27b4d96…` | v1, round 1 | *** WRONG *** |

### 5.3 `r2-preal1` is a trap that looks like a win

It scores **23.8657 dB** on the test set — higher than anything in the field, and
+0.234 dB over what we ship. It also **loses 4.03 dB at σ_mul 0.45**. It was
trained on real data only, so it never saw hard degradations and falls apart on
them. `swap.py` refuses it without `--force`. Leave that guard alone.

### 5.4 `docs/tool_output/reship_summary.txt` says FAILED

Line 11 of that file reads:

```
packaging : outputs/ for the organisers' test set + clean-directory run   FAILED rc=1
10 of 11 steps OK
```

**This file is in the repo and a judge can read it.** Either re-run
`tools/package_check.py` and regenerate the summary green, or add a one-line note
next to it explaining what failed and that it is fixed. Do not leave a file in the
submission that says FAILED with no explanation.

### 5.5 `models/model.pt` is touched only through `swap.py`

Not `cp`. Not drag-and-drop. `swap.py` hash-checks, backs up, re-scores and
auto-restores on mismatch. See `CONTRIBUTING.md`.

---

## 6. Decisions, and why

| decision | reason | evidence |
|---|---|---|
| residual CNN, not a transformer | sample efficiency at our data scale; capacity barely matters so architecture probably doesn't either | §4.3 |
| effort into degradation modelling, not architecture | 3.7 dB from one training flag vs 0.39 dB from 78x parameters | §4.4 vs §4.3 |
| speckle (M1), not speckle+Poisson (M3) | M3's Poisson coefficient is negative — unphysical | §2 |
| no full determinism | measured cross-GPU gap is 22.7468 (4090) vs 22.74 (4050); ~100x smaller than the effects we study, and deterministic kernels cost 15–25% throughput | `docs/REPRODUCIBILITY.md` |
| `cudnn.benchmark` off | +2.969 s, changes outputs, buys nothing at fixed input shape | §4.7 |
| ship the LPIPS variant | 2nd on perceptual quality; hedge against a perceptually-weighted metric | §4.2 — **conditional, see 7.1** |

### 6.1 The honest version of "why EDSR"

**No architecture comparison was run before the backbone was selected.** A
known-good residual SR backbone was adopted early because it trains on a 6 GB laptop
GPU inside a week, and the remaining time went to the degradation model. The
justification in §4.3 is real and measured, but it was found *after* the choice, not
before — it is a defence of the decision, not the reason it was taken.

Anyone defending this should be able to reconstruct §4.3 from the CSVs themselves
rather than reciting it. The measurement is the argument; the sentence is not.

The weakest part of our position: SEM images of semiconductor structures contain
**periodic repeating structure** (line arrays, contact grids, regular pitch), which
is exactly what non-local attention is good at and what a 16-block CNN with a
~33-pixel receptive field cannot see as a pattern. We have no evidence that
long-range modelling doesn't help. We only have evidence that capacity doesn't.

---

## 7. Open questions

### 7.1 The metric weighting (unanswered, asked of the organisers)

We do not know how PSNR, SSIM and LPIPS are weighted in the final score. This
single unknown determines which checkpoint we should ship:

- **PSNR/SSIM-weighted** → `pr50-w50-120` wins (+0.13 dB PSNR, +0.011 SSIM, and
  better robustness)
- **LPIPS ≥ ~35% of the score** → the current shipped model wins

`pr50-w50-120` was trained, scored, and **the weights were not kept.** Retraining
it takes 120 epochs. If there is any chance of learning the weighting on site, that
retrain should be running.

### 7.2 The architecture comparison (`kla2/`, built, not run)

Ten architectures at matched ~1.37 M parameters: `forgex`, `edsr_base`,
`abl_nostem`, `abl_noskip`, `unet`, `nafnet`, `nafnet_ours`, `restormer`,
`restormer_ours`, `swinir`. The `_ours` variants carry our stem and skip, so a win
can be attributed to blocks versus front-end. Stage 0 (smoke, 2 epochs, all ten)
passed. The full run never started.

Predicted outcome, recorded in advance so it can be checked: NAFNet within
±0.25 dB of ForgeX; `nafnet_ours` beats plain `nafnet` by more than the gap between
architectures; SwinIR underperforms (data-starved); Restormer slowest for no gain.

### 7.3 Not yet done

- Correlation diagnostic: variance-vs-mean and autocorrelation of real versus
  synthetic residuals, side by side. Would test whether our synthetic noise has the
  right *spatial* structure, not just the right variance.
- Regenerate `reship_summary.txt` green (see 5.4).

---

## 8. Repo map

```
forgex-kla-ps01/
  run.py                    inference entry point; OOM-safe batch halving
  train_submitted.py        the training script as submitted
  models/model.pt           SHIPPED WEIGHTS — touch only via swap.py
  src/
    model.py                Restorer, VarianceStabilisingStem, ResBlock
    degrade.py              the synthetic degradation pipeline
    dataset.py              splits; make_split, make_block_split
    train.py                training loop; main() is the injection point
    validate.py             scoring; writes git_sha + weight_sha1 per row
    losses.py  metrics.py
  tools/                    verify_shipped, package_check, bench, scorecard, …
  docs/
    HANDOFF.md              this file
    CHECKPOINTS.md          the four-identical-sizes problem
    DATA.md                 dataset layout and transfer procedure
    REPRODUCIBILITY.md      the determinism decision
    ENGINEERING_LOG.md      42 KB, the full chronological record
    results.csv             every scored run
    queue/testset_results.csv   the 11-checkpoint field
    capacity_sweep.csv  timesweep.csv  per_category.csv  noise_sweep_queue.csv
  CONTRIBUTING.md           the five rules
```

Sibling directories, **not in git**: `kla2/` (architecture study),
`checkpoints/` (renamed, hash-gated weights), `semicon_train_data/`,
`semicon_test_data/`.

---

## 9. If you are an AI assistant reading this

You have just been handed the entire project state. Some guidance:

1. **Do not run `git` commands against this repository.** Hand them to Abishek to
   run. A previous assistant left a stale `.git/index.lock` doing this.
2. **Every figure you state must trace to a CSV in `docs/`.** Two research briefs
   went out with a round-1 baseline because this rule wasn't followed. When you are
   about to quote a number, name the file it came from.
3. **Checkpoint identity is by sha1, never by filename or size.** See 5.2.
4. **Attribution:** commits and repo files carry Abishek's name only. Do not add
   assistant or vendor attribution to anything in this repository.
5. Private keys never appear in chat. `.pub` files only; keys move by USB.
6. The interesting open problems are in §7. The settled ones are in §6 — reopen
   them only with evidence.
