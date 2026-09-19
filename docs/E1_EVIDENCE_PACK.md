# ForgeX — E1 Evidence Pack: image clarity, blur diagnosis, and the e1f_gate decision

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> This is a historical record. The current shared-head release is documented in [RELEASE_STATUS.md](RELEASE_STATUS.md).
>
> Current: [RELEASE_STATUS.md](RELEASE_STATUS.md)


**Date:** 18 September 2026 · **Scope:** files on disk only. Prior notes treated as hypotheses.
Every number below was recomputed from checkpoints, logs, CSVs or source unless a file path is cited.

---

## 1. Exact experiment provenance

### 1.1 Git and checkpoints

| item | value |
|---|---|
| Git HEAD (`forgex-kla-ps01`) | `02e05a6` — *working tree dirty* (`run.py`, `HACKATHON_CONTEXT.md` modified; `tools/stress_run.py`, `tools/calibrate_knob.py`, `tools/demo_knob.py`, `docs/DAY_LOG_17SEP.md` untracked) |
| Protocol rows recorded git sha | `02e05a6+dirty` — **all E1 scoring rows were produced from a dirty tree** |

| run | checkpoint | SHA-1 (12) | bytes | arch.txt | best epoch | `last.pt` |
|---|---|---|---|---|---:|---|
| **shipped ForgeX** | `forgex-kla-ps01/models/model.pt` | `e208d13d62b3` | 5,493,743 | — | *(deployed `last.pt`, ep 119)* | n/a |
| e1a_base | `runs/e1a_base/best.pt` | `2eef1d693797` | 5,495,521 | `e1a_base` | 45 | **ABSENT** |
| e1f_gate | `runs/e1f_gate/best.pt` | `61587e96f554` | 5,419,749 | `e1f_gate` | 89 | **ABSENT** |
| e1g_grad20 | `runs/e1g_grad20/best.pt` | `b843f37bff7f` | 5,495,521 | `e1a_base` | 41 | **ABSENT** |
| e1h_nafdeep | `runs/e1h_nafdeep/best.pt` | `899c9967275c` | 5,873,254 | `nafnet_deep` | 87 | **ABSENT** |
| e1b_se | `runs/e1b_se/best.pt` | `036200ed6db1` | 5,552,325 | `e1b_se` | 43 | **ABSENT** |
| e1c_mdta1 | `runs/e1c_mdta1/best.pt` | `95c068879804` | 5,418,743 | `e1c_mdta1` | 30 | **ABSENT** |

### 1.2 Exact training commands

**Shipped ForgeX** (`README.md` §"the SHIPPED configuration`; corroborated by `docs/REPRODUCIBILITY.md` and `docs/CHECKPOINTS.md`):
```
python src/train.py --data <data> --out runs/pr50-w50-lp05-120 --amp \
    --epochs 120 --iters 500 --batch 32 --ch 64 --nb 16 \
    --p-real 0.5 --wide-p 0.5 --w-lpips 0.05 \
    --seed 0 --split-seed 0
```
`--w-grad` not passed → default **0.05**; `--w-ssim` not passed → default **0.15**; `--loss` default `combo`.

**All E1 rows** (`kla2/tools/e1.sh`), identical `FLAGS` for every row:
```
FLAGS="--seed 0 --split-seed 0 --amp --p-real 0.5 --wide-p 0.5 --w-lpips 0.05 --w-grad 0.05"
DATA=../semicon_train_data/semicon_train_data     EPOCHS=120 (actual; default in file is 40)

python3 tools/train_arch.py --arch <arch> --out runs/<name> --data $DATA \
    --epochs 120 --ch <solved> --workers 3 $FLAGS [row-specific]

e1a_base    : --arch e1a_base   --ch 64
e1f_gate    : --arch e1f_gate   --ch 53
e1g_grad20  : --arch e1a_base   --ch 64   --w-grad 0.20      <- ONLY difference
```

### 1.3 Architecture differences

| | shipped / e1a_base | e1f_gate | e1g_grad20 |
|---|---|---|---|
| Block interior | `x + 0.1·c2(ReLU(c1(x)))` | `x + 0.1·c2(SimpleGate(c1(LayerNorm2d(x))))` | same as e1a_base |
| `c1` output width | ch → ch | **ch → 2·ch** (SimpleGate halves it) | ch → ch |
| Normalisation | none | **LayerNorm2d** per block | none |
| Nonlinearity | ReLU (additive) | **SimpleGate** `a*b` (multiplicative) | ReLU |
| Width `ch` | 64 | **53** (parameter-matched) | 64 |
| Blocks | 16 | 16 | 16 |
| Params | **1,368,705** | **1,346,360** (−1.6%) | 1,368,705 |
| GFLOPs @128² | **44.827** | **44.015** (−1.8%) | 44.827 |
| Unchanged | VS stem · head conv · res_scale 0.1 · long skip · body_tail · PixelShuffle ×2 · zero-init tail · fp32 bicubic skip | | |

### 1.4 Is e1a_base bit-identical to shipped ForgeX?

**Two separate questions; the answer differs.**

| question | verdict | evidence |
|---|---|---|
| Is the **architecture** identical? | **YES, exactly** | Shipped weights load into `ARCHS['e1a_base']` with `strict=True`, 72/72 keys, all shapes match. Forward equivalence on random input: **max\|e1a_base(x) − shipped(x)\| = 0.000e+00**. Both 1,368,705 params |
| Is the **checkpoint** identical? | **NO** | `max\|W_e1a_base − W_shipped\| = 0.5608`. Different SHA-1, different training run, different epoch (45 vs 119) |

`e1.sh`'s header claim *"e1a_base … is bit-identical to src/model.py's Restorer"* is correct **about the architecture** and should not be read as a claim about weights. **This distinction matters for §3 condition 3.**

### 1.5 Held-fixed audit, and every mismatch

| factor | e1a_base | e1f_gate | e1g_grad20 | shipped | match? |
|---|---|---|---|---|---|
| data root | `semicon_train_data` (4,785) | same | same | same | ✅ |
| `--seed` / `--split-seed` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | ✅ |
| degradation mixture | `--p-real 0.5 --wide-p 0.5` | same | same | same | ✅ |
| optimiser / schedule | AdamW, Cosine | same | same | same | ✅ |
| epochs | 120 | 120 | 120 | 120 | ✅ |
| batch / iters / crop | 32 / 500 / 64 | same | same | same | ✅ |
| `--w-ssim` / `--w-lpips` | 0.15 / 0.05 | 0.15 / 0.05 | 0.15 / 0.05 | 0.15 / 0.05 | ✅ |
| `--w-grad` | 0.05 | 0.05 | **0.20** | 0.05 | intended |
| precision | AMP fp16, loss fp32 | same | same | same | ✅ |
| **checkpoint scored** | **`best.pt` ep 45** | **`best.pt` ep 89** | **`best.pt` ep 41** | **`last.pt` ep 119** | ❌ **MISMATCH** |
| training hardware | A100-SXM4-80GB ×3, 2 rows/card | same | same | RTX 4050 / 4090 | ❌ mismatch (no numeric effect established) |
| wall clock | 3.08 h | 3.45 h | 3.03 h | 4.3 h (4050) / 28 min (4090) | informational |

**Mismatches, stated explicitly:**

1. **`best.pt` vs `last.pt` (material).** Every E1 row is scored from a val-PSNR-selected `best.pt`; shipped is the deployed `last.pt`. `last.pt` was **not** retrieved from the pod, so a like-for-like `last.pt` comparison **cannot be run from files currently on disk**. Selecting on the 200-image val split and reporting on the disjoint 297-image test set is methodologically sound, but it is not the same protocol as the shipped number.
2. **Dirty git tree.** All `protocol.csv` / `final.csv` rows record `02e05a6+dirty`. The exact source state is not recoverable from the sha alone.
3. **Documentation defect.** `docs/REPRODUCIBILITY.md` gives the shipped command as `--data data/train`. On this disk `train/` is **round-1 photographs, not SEM** (`docs/DATA.md`: f90 0.287 vs 0.735). The path is a placeholder; a reader following it literally would train on the wrong data.
4. **`docs/per_image_stats.csv` is corrupt by duplication** — 5,876 rows for 4,785 unique ids; **1,091 ids duplicated**, duplicates byte-identical (the generator appends). Consumed by `--hard-w` content-balanced sampling. **`--hard-w` was 0.0 in every command above, so no run here is affected** — latent, not active.

---

## 2. Loss verification

### 2.1 The shipped loss, read from `src/losses.py` and `src/train.py`

$$L=\underbrace{\mathrm{Charb}(\hat y,y)}_{1.00}+0.15\underbrace{(1-\mathrm{SSIM})}_{\text{clamped}}+0.05\,L_{\text{grad}}+0.05\,L_{\text{LPIPS}}$$

| term | weight | exact numerical setting |
|---|---:|---|
| Charbonnier | 1.00 | `sqrt((x−y)² + ε²).mean()`, **ε = 1e-3** |
| SSIM | `--w-ssim` **0.15** | `1 − ssim(pred.clamp(0,1), gt)`; Gaussian **ws=11, σ=1.5**, **C1=0.01², C2=0.03²**, `padding=5` |
| Gradient | `--w-grad` **0.05** | `charbonnier(dx(x),dx(y),ε) + charbonnier(dy(x),dy(y),ε)`, forward differences, ε=1e-3 |
| LPIPS | `--w-lpips` **0.05** | AlexNet, 1→3 channel repeat, `[0,1]→[−1,1]`, frozen (`requires_grad_(False)`) |

Mode is `combo`; `charbonnier` mode exists as a control and was not used.

### 2.2 fp32 / autocast handling — verified in source

| term | autocast | TF32 | verified |
|---|---|---|---|
| Charbonnier | inherits the caller | — | `.float()` on both operands inside `charbonnier()` |
| **SSIM** | **explicitly disabled** — `torch.amp.autocast(..., enabled=False)` in `_ssim_fp32` | **cuDNN TF32 explicitly disabled** and restored in a `finally` | ✅ |
| **Gradient** | inherits; built from `charbonnier()` so operands are `.float()` | — | ✅ |
| **LPIPS** | **explicitly disabled** — same construct | — | ✅ |
| Call site | `src/train.py:189` — `loss, parts = crit(pred.float(), hr_b.float())`, **outside** the autocast block | | ✅ |

Two documented reasons, both in-source: in fp16 the SSIM local variances go slightly negative and the denominator underflows below fp16's smallest normal → Inf → NaN, and NaN is scale-invariant so `GradScaler` skips every step forever with no error raised. Separately, "fp32" convolution on Ampere/Ada runs in TF32 (10-bit mantissa) and `E[x²]−E[x]²` cancels below that precision in flat regions — **measured to under-report SSIM by ≈0.017**.

**Verdict: fp32/autocast handling is correct and defensively written. No defect found.**

### 2.3 What changed in e1g_grad20

Exactly one flag: **`--w-grad 0.05 → 0.20`** (4×). Same network (`arch.txt` = `e1a_base`), same ch 64, same everything else. `e1.sh`'s own comment states the intent: *"the gradient term is the only part of the objective fighting blur … is the blur a loss problem?"*

### 2.4 e1g_grad20 result vs control

| metric | e1a_base | e1g_grad20 | Δ |
|---|---:|---:|---:|
| val last-5 PSNR | 23.1142 | 23.1525 | **+0.0383** |
| val last-5 SSIM | 0.58909 | 0.58858 | **−0.00051** |
| test-297 PSNR | 23.6778 | 23.6901 | +0.0123 |
| test-297 SSIM | 0.59380 | 0.59664 | +0.00284 |
| test-297 LPIPS | 0.21297 | 0.21319 | **+0.00022** (worse) |
| 1,197-set PSNR | 23.4760 | 23.4858 | +0.0098 |
| 1,197-set SSIM | 0.58044 | 0.58207 | +0.00163 |
| 1,197-set LPIPS | 0.21675 | 0.21874 | **+0.00199** (worse) |
| **HF energy ratio** (n=150) | 0.2490 | **0.2411** | **−0.0079 (worse)** |
| **HF correlation** (n=150) | 0.2954 | 0.2993 | +0.0039 |
| grad ratio (n=150) | 0.4913 | **0.4882** | **−0.0031 (worse)** |
| latency b1 / b32 | 9.4887 / 5.0788 ms | 9.4562 / 11.942 ms | within measurement noise |

### 2.5 Ruling on stronger gradient supervision

**REJECTED by the evidence.** Quadrupling the gradient weight produced a PSNR change (+0.038 dB val) far below the 0.10 dB pre-registered threshold, **lost** SSIM on the val split, made **LPIPS worse on both evaluation sets**, and — decisively — **reduced** both high-frequency energy ratio (0.2490 → 0.2411) and gradient ratio (0.4913 → 0.4882). The term meant to fight blur made the output measurably *less* high-frequency. The blur is **not** a loss-weighting problem. Note also that a 4× weight change is a coarse probe; this rejects *this* intervention, not every conceivable edge-supervision design.

---

## 3. e1f_gate versus baseline

### 3.1 Metrics

| metric | e1a_base | e1f_gate | Δ |
|---|---:|---:|---:|
| **val last-5 PSNR** (200-img split) | 23.1142 | 23.2231 | **+0.1089** |
| **val last-5 SSIM** | 0.58909 | 0.60115 | **+0.01206** |
| last-5 epoch spread | 0.030 dB | 0.030 dB | — |
| **test-297 PSNR** | 23.6778 | **23.8296** | **+0.1518** |
| **test-297 SSIM** | 0.59380 | **0.62104** | **+0.02724** |
| **test-297 LPIPS** | 0.21297 | **0.18126** | **−0.03171** |
| **1,197-set PSNR** | 23.4760 | **23.5943** | **+0.1183** |
| **1,197-set SSIM** | 0.58044 | **0.60630** | **+0.02586** |
| **1,197-set LPIPS** | 0.21675 | **0.18499** | **−0.03176** |
| HF energy ratio (n=150) | 0.2490 | **0.2823** | **+0.0333** |
| HF correlation (n=150) | 0.2954 | **0.3261** | **+0.0307** |
| grad ratio (n=150) | 0.4913 | **0.5360** | **+0.0447** |
| Parameters | 1,368,705 | 1,346,360 | −22,345 |
| GFLOPs | 44.827 | 44.015 | −0.812 |
| latency b1 | 15.356 / 9.489 ms *(two runs)* | 24.371 / 23.031 ms | **gate slower** |
| latency b32 | 6.464 / 5.079 ms/img | 15.493 / 29.454 ms/img | **gate slower** |
| throughput b32 | 153.1 / 196.9 img/s | 64.5 / 34.0 img/s | gate lower |
| VRAM b1 / b32 | 0.123 / 3.533 GB | **0.073 / 2.678 GB** | gate lower |
| training wall clock | 3.08 h | 3.45 h | +12% |
| end-to-end timing | **UNKNOWN — not measured for either E1 row** | | |

Latency figures are quoted from both independent runs because they disagree by up to 90% on the same GPU and weights (thermal drift, RTX 4050 laptop). **The ordering is consistent across both runs; the magnitudes are not trustworthy.**

### 3.2 Paired significance (n = 297, same images, same harness)

| metric | mean Δ | sd | t | 95% bootstrap CI | gate better on |
|---|---:|---:|---:|---|---:|
| PSNR (dB) | **+0.15181** | 0.25305 | **+10.3** | [+0.12397, +0.18173] | 240 / 297 |
| SSIM | **+0.02724** | 0.03039 | **+15.4** | [+0.02399, +0.03085] | **297 / 297** |
| LPIPS | **−0.03171** | 0.03917 | **−13.9** | [−0.03620, −0.02740] | **297 / 297** |

Unpaired test-set draw noise for e1f_gate PSNR is sd **0.2097 dB** — the paired comparison is **≈14× tighter**, which is why the paired CI excludes zero comfortably while a naive unpaired comparison would not.

### 3.3 The pre-registered decision rule, applied exactly

From `docs/DAY_LOG_17SEP.md` §"The decision rule, fixed in advance" — swap **only** if a row clears **all** of:

| # | condition | result |
|---|---|---|
| 1 | ≥ 0.10 dB PSNR **or** ≥ 0.010 SSIM over `e1a_base` | **PASS (both, twice).** val last-5: +0.1089 dB **and** +0.01206 SSIM. test-297: +0.1518 dB **and** +0.02724 SSIM |
| 2 | no loss on the "OOD" column | **PASS.** 1,197-set: +0.1183 dB, +0.02586 SSIM, −0.03176 LPIPS — gains on all three, losses on none |
| 3 | `e1a_base` landed near the shipped model's number | **PARTIAL — see below** |

**Condition 3 in detail.** e1a_base scores **23.6778 / 0.59380 / 0.21297** on the 297 set; shipped scores **23.632 / 0.60791 / 0.19288**.
* PSNR: control is **+0.046 dB** from shipped → near. **PASS.**
* SSIM: control is **−0.0141** from shipped → **larger than the rule's own 0.010 SSIM threshold**. **The control did not reproduce the shipped SSIM.**
* LPIPS: control is **+0.0201** worse.

The most likely cause is the `best.pt` (ep 45) vs `last.pt` (ep 119) mismatch from §1.5 — a PSNR-selected checkpoint need not be the SSIM/LPIPS-selected one — but **this is a hypothesis, not established by any file on disk**, because `last.pt` is absent.

**Consequence:** part of e1f_gate's +0.02724 SSIM over the control is recovering ground the control lost relative to shipped. The honest pair of deltas:

| baseline | Δ PSNR | Δ SSIM | Δ LPIPS |
|---|---:|---:|---:|
| vs **e1a_base** (matched protocol) | +0.1518 | +0.02724 | −0.03171 |
| vs **shipped** (mismatched protocol) | **+0.1976** | **+0.01313** | **−0.01162** |

**Both clear condition 1 independently.**

### 3.4 Verdict

**SIGNIFICANT.** e1f_gate clears conditions 1 and 2 outright, on two disjoint image sets, on all three quality metrics, with paired t ≥ 10.3 and bootstrap CIs excluding zero. It wins SSIM and LPIPS on **297/297** images. Condition 3 passes on PSNR and fails on SSIM, which weakens the *magnitude* of the SSIM claim but not its *direction* — the gate still beats the shipped model by +0.0131 SSIM under the same mismatched protocol.

**Not established:** any latency claim (measurements disagree by 90%), any end-to-end claim (never measured), and any claim under a matched `last.pt` protocol (file absent).

---

## 4. Per-image blur / failure analysis

Paired, same 297 images, `kla2/results/per_image/{e1a_base,e1f_gate}.csv`; content statistics recomputed from GT/LR. Full table: `per_image_297.csv`.

### 4.1 What predicts failure

Correlation of **gain over bicubic** with image content (e1a_base):

| predictor | PSNR gain | SSIM gain |
|---|---:|---:|
| GT f90 (fineness) | −0.605 | −0.696 |
| **GT gradient energy** | **−0.724** | **−0.839** |
| GT edge density | +0.075 | +0.239 |
| LR noise variance | +0.087 | −0.081 |

**GT gradient energy is the dominant predictor (r = −0.839 on SSIM). Noise severity is not predictive at all.** The failure is content-driven, not noise-driven.

### 4.2 What predicts SimpleGate's benefit

| predictor | Δ PSNR | Δ SSIM | Δ LPIPS |
|---|---:|---:|---:|
| GT f90 | −0.430 | **+0.513** | −0.307 |
| **GT gradient energy** | −0.418 | **+0.669** | **−0.489** |
| GT edge density | +0.437 | −0.104 | −0.016 |
| LR noise variance | −0.156 | +0.414 | −0.364 |

**The gate's SSIM/LPIPS benefit is largest exactly where the baseline fails worst.** The signs are opposite (−0.839 failure vs +0.669 benefit) against the same predictor — the intervention targets the failure mode rather than lifting everything uniformly.

### 4.3 Quartiles by GT f90 (texture fineness)

| Q | f90 range | n | base gain | gate gain | Δ PSNR | Δ SSIM | base SSIM<bicubic | gate SSIM<bicubic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0391–0.1881 | 75 | +5.784 | +6.065 | +0.2805 | +0.01239 | 0/75 | **0/75** |
| Q2 | 0.1901–0.4866 | 74 | +2.845 | +3.066 | +0.2218 | +0.01558 | 3/74 | **0/74** |
| Q3 | 0.4877–0.8469 | 74 | +2.012 | +2.110 | +0.0976 | +0.02808 | 17/74 | **5/74** |
| Q4 | 0.8499–1.1251 | 74 | +2.214 | +2.220 | +0.0056 | **+0.05314** | **40/74** | **9/74** |

Δ SSIM rises monotonically Q1→Q4 (+0.0124 → +0.0531) while Δ PSNR falls monotonically (+0.2805 → +0.0056). **On the most textured quartile the gate buys essentially no PSNR and the largest SSIM gain** — it is restoring structure, not reducing squared error.

### 4.4 Quartiles by LR noise variance

| Q | noise range | n | base gain | gate gain | Δ PSNR | Δ SSIM | base SSIM<bic | gate SSIM<bic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0024–0.0071 | 75 | +3.172 | +3.395 | +0.2233 | +0.01706 | 3/75 | 0/75 |
| Q2 | 0.0071–0.0096 | 74 | +2.765 | +2.934 | +0.1698 | +0.02079 | 10/74 | 0/74 |
| Q3 | 0.0096–0.0139 | 74 | +3.399 | +3.540 | +0.1411 | +0.02959 | 23/74 | 6/74 |
| Q4 | 0.0139–0.0293 | 74 | +3.555 | +3.627 | +0.0721 | +0.04168 | 24/74 | 8/74 |

**Base gain does not fall with noise** (+3.17 → +3.56). Noise severity is *not* the driver; texture is. The mild trend in SSIM losses across noise quartiles is confounded with texture.

### 4.5 The headline failure count

| model | images scoring **worse than bicubic** on SSIM |
|---|---:|
| e1a_base | **60 / 297** |
| **e1f_gate** | **14 / 297** |

**A 77% reduction in the failure mode.**

### 4.6 Worst-20 baseline SSIM (all 20 rescued or improved by the gate)

| id | f90 | noise | base SSIM | bicubic SSIM | gate SSIM | Δ SSIM | Δ PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000166 | 0.9518 | 0.02870 | 0.16105 | 0.43786 | 0.29158 | +0.13053 | +0.4367 |
| 000167 | 0.9365 | 0.02670 | 0.17174 | 0.46726 | 0.31030 | +0.13856 | +0.4846 |
| 000052 | 1.0237 | 0.02624 | 0.19521 | 0.30170 | 0.33374 | +0.13853 | −0.0442 |
| 000165 | 0.9570 | 0.02345 | 0.19861 | 0.46005 | 0.38778 | +0.18918 | +0.5387 |
| 000040 | 0.8980 | 0.02730 | 0.25264 | 0.44357 | 0.46998 | +0.21734 | +0.0259 |
| 000011 | 1.1247 | 0.01551 | 0.25305 | 0.23942 | 0.30682 | +0.05377 | −0.2844 |
| 000053 | 0.9908 | 0.01732 | 0.26315 | 0.36191 | 0.35986 | +0.09670 | +0.0627 |
| 000017 | 1.1251 | 0.01521 | 0.26608 | 0.23415 | 0.30721 | +0.04113 | −0.3397 |
| 000010 | 1.0952 | 0.01357 | 0.26806 | 0.24525 | 0.30183 | +0.03377 | +0.0284 |
| 000051 | 0.9745 | 0.01718 | 0.28055 | 0.37047 | 0.34214 | +0.06159 | +0.1432 |
| 000159 | 1.0652 | 0.01330 | 0.28080 | 0.32358 | 0.33112 | +0.05032 | +0.0589 |
| 000181 | 0.7554 | 0.01323 | 0.28631 | 0.33706 | 0.35739 | +0.07108 | +0.0775 |
| 000273 | 1.0093 | 0.01824 | 0.28747 | 0.29042 | 0.33393 | +0.04646 | −0.1434 |
| 000160 | 1.0489 | 0.01598 | 0.28800 | 0.29207 | 0.32227 | +0.03427 | −0.0186 |
| 000054 | 1.1172 | 0.00850 | 0.29941 | 0.27471 | 0.34017 | +0.04075 | −0.0376 |
| 000070 | 1.0971 | 0.00942 | 0.29946 | 0.28368 | 0.33483 | +0.03537 | −0.0097 |
| 000161 | 1.0479 | 0.01073 | 0.30641 | 0.33525 | 0.35212 | +0.04571 | +0.0636 |
| 000037 | 1.1041 | 0.00621 | 0.31495 | 0.29614 | 0.35376 | +0.03881 | +0.0684 |
| 000059 | 1.0553 | 0.01332 | 0.31683 | 0.36894 | 0.38506 | +0.06823 | −0.1837 |
| 000026 | 0.8245 | 0.02307 | 0.31978 | 0.46683 | 0.46039 | +0.14062 | +0.1050 |

**Every one of the 20 worst has f90 ≥ 0.755 (dataset range 0.039–1.125).** Failure is confined to the fine-texture tail. Note rows 000166/000167/000165/000040/000026: the gate improves them substantially yet they remain **far below bicubic's SSIM** — improved, not solved.

### 4.7 Where SimpleGate helps most — and least

**Helps most (top 5 by Δ SSIM):** 000040 (+0.21734), 000165 (+0.18918), 000001 (+0.15956), 000026 (+0.14062), 000167 (+0.13856) — all f90 ≥ 0.82, all noise ≥ 0.021.

**Helps least (bottom 5 by Δ SSIM):** 000146 (+0.00056), 000260 (+0.00147), 000047 (+0.00197), 000265 (+0.00222), 000259 (+0.00222).

**There is no "hurts" list. The minimum Δ SSIM across all 297 images is +0.00056 — SimpleGate improves SSIM on every single image.** Δ PSNR is negative on 57/297 and Δ LPIPS is positive (worse) on 0/297.

### 4.8 High-frequency reconstruction (n = 150, stride-8 subset of the 1,197)

`hf_model` = HF energy of output ÷ HF energy of GT above 0.5 of output Nyquist.

| model | HF ratio | HF corr | grad ratio | corr > bicubic | images with HF ratio > 1 |
|---|---:|---:|---:|---:|---:|
| bicubic | **1.1610** | 0.2139 | 0.8210 | — | — |
| e1a_base | 0.2490 | 0.2954 | 0.4913 | 146/150 | **0/150** |
| **e1f_gate** | **0.2823** | **0.3261** | **0.5360** | **150/150** | **0/150** |
| e1g_grad20 | 0.2411 | 0.2993 | 0.4882 | 149/150 | 0/150 |
| e1h_nafdeep | 0.2465 | 0.3020 | 0.5361 | 149/150 | 0/150 |

Paired on the same 150 images: Δ HF ratio **+0.03325** (t = +5.2, better on 104/150), Δ grad ratio **+0.04478** (t = +7.9, 116/150), Δ HF correlation **+0.03070** (t = **+16.4**, **144/150**).

`corr(GT texture, Δ HF ratio) = +0.601` — again, the gate adds the most high-frequency energy exactly on the most textured images.

### 4.9 Diagnosis

**The blur is high-frequency UNDER-RECONSTRUCTION. It is not oversmoothing of correctly-recovered content, not a denoising-strength error, and not a registration error.**

Evidence, in order of strength:

1. **Both models reconstruct only ~25–28% of the GT's high-frequency energy** (0.2490 and 0.2823 against 1.0). The deficit is enormous and one-sided. **0/150 images exceed ratio 1.0 for either model** — there is no image anywhere in the set where the model puts in too much detail.
2. **Bicubic's HF ratio is 1.1610** — i.e. bicubic passes *more* HF energy than GT contains (it passes the input's speckle straight through), yet its HF **correlation** is only 0.2139 against the models' 0.2954/0.3261. So the models place their HF energy far better than bicubic while supplying far less of it. **The deficit is in amount, not in position.** This also rules out registration: a misaligned output would show low HF *correlation*, and the models' correlation is the highest in the table.
3. **Failure is predicted by GT gradient energy (r = −0.839), not by LR noise (r = −0.081).** If the cause were wrong denoising strength, noise severity would predict failure. It does not.
4. **Quadrupling the gradient loss made HF energy *worse*** (0.2490 → 0.2411). A loss-weight lever aimed directly at edges could not move it, which argues the limit is representational/informational rather than a mis-specified objective.
5. **The visual grid confirms it.** On 000166, 000040 and 000052 (f90 0.90–1.02) `e1a_base` collapses to near-flat grey while GT is dense granular texture; `e1f_gate` recovers visible granularity on the same inputs. The error maps are diffuse and texture-shaped, not edge-ringed and not spatially shifted.

**Mechanism consistent with all of the above:** under an L1/Charbonnier-dominant objective the conditional-mean estimate of an unresolvable stochastic texture is its local mean, so the optimal output *is* flat where the texture is not recoverable from the input. That is a property of the degradation's null space, not a training bug. SimpleGate's multiplicative nonlinearity (`a*b`) partially escapes it — measured, not asserted: +0.0333 HF energy, +0.0307 HF correlation, +0.0447 grad ratio, all paired and significant.

**What is NOT established:** that SimpleGate's advantage comes from multiplicativity specifically. e1f_gate changes three things at once — SimpleGate, LayerNorm2d, and width 64→53. **No ablation isolating them exists on disk.**

---

## 5. Decision about the next training change

Candidates ranked by evidence support.

### A. Adopt e1f_gate unchanged — **RANK 1**
* **Problem:** 60/297 images score below bicubic on SSIM.
* **Hypothesis:** already tested. SimpleGate reduces it to 14/297.
* **Design:** no training. Inline `LayerNorm2d`/`SimpleGate`/`GateBlock` into `run.py`, `ch=53`, swap weights, re-verify hash, re-run `stress_run.py`.
* **Cost:** ~1 h engineering, 0 GPU-hours.
* **Fair experiment:** already run — 297 + 1,197 images, paired, matched flags.
* **Success:** met. Clears conditions 1 and 2; t = +10.3 / +15.4 / −13.9.
* **Failure:** condition 3 partial; latency unmeasured and probably worse.

### B. High-frequency / Laplacian-pyramid supervision on e1f_gate — **RANK 2**
* **Problem:** HF ratio 0.2823 against 1.0 — a 72% deficit, the single largest measured gap.
* **Hypothesis:** the current loss has **no term that acts per frequency band**. Charbonnier and SSIM are broadband; the gradient term is a single first-difference. A band-decomposed loss would weight the deficient bands directly.
* **Exact design:** 3-level Laplacian pyramid (`L_k = G_k − up(G_{k+1})`, 5×5 Gaussian, stride 2); add `w_hf · Σ_k λ_k · charbonnier(L_k(ŷ), L_k(y))` with λ = (1.0, 0.5, 0.25) finest→coarsest, `w_hf = 0.05`. All existing terms unchanged. No GAN, no perceptual net beyond the LPIPS already present.
* **Cost:** ~3.5 h on one A100 (pyramid is a few convolutions).
* **Fair experiment:** e1f_gate + the term vs e1f_gate, identical flags/seed/split/data, 120 epochs, `last.pt` **retained**.
* **Success:** HF ratio ≥ 0.32 (+0.04, larger than the +0.033 SimpleGate itself bought) **and** SSIM not worse **and** HF ratio > 1.0 on 0 images.
* **Failure:** HF ratio ≤ 0.29, or any image over 1.0, or SSIM loss > 0.005.
* **Caveat:** §2.5 showed one edge-supervision lever fail. This is a materially different mechanism (per-band vs single-difference), but the prior is not favourable.

### C. Changed crop / patch sampling — **RANK 3**
* **Problem:** failure concentrates in the f90 ≥ 0.85 quartile (40/74 SSIM losses).
* **Hypothesis:** 64×64 uniform crops under-represent fine-texture content relative to its share of the failure.
* **Exact design:** `--hard-w 1.0 --stats-csv docs/per_image_stats.csv` (tilts draws ∝ f90). **Blocked: `per_image_stats.csv` must be de-duplicated first (§1.5 item 4).**
* **Cost:** ~3.5 h + a fix to the stats file.
* **Fair experiment:** e1f_gate + `--hard-w 1.0` vs e1f_gate, everything else fixed.
* **Success:** Q4 SSIM-loss count < 9/74 without Q1–Q2 regressing more than 0.005 SSIM.
* **Failure:** Q4 unchanged, or Q1–Q2 regress.
* **Note:** the mechanism is untested here — the sampler has never been exercised at non-zero weight.

### D. Degradation-conditioned gate — **RANK 4**
* **Problem:** none demonstrated. **LR noise variance correlates −0.081 with SSIM gain.**
* **Verdict:** **REJECT.** The premise is that performance depends on degradation severity. The data says it does not; texture does. No measured failure supports this.

### E. Do nothing further — **RANK 2 (tied with B, on time risk)**
* Defensible. A is significant and shippable; B and C are speculative. With a deadline, A alone is the risk-minimising path.

**Explicitly not recommended:** GANs, transformers, wavelets, attention. The measured failure is an HF *deficit* with **0/150 images showing excess** — the failure mode these methods address (under-sharpening judged perceptually) is real, but they introduce synthesis capability that the measurements give no evidence is needed and that would put the 0/150 property at risk. Attention specifically was tested: e1b_se and e1c_mdta1 both scored below e1f_gate on every metric here.

---

## 6. One recommended next run

**Run B.** But only after A is shipped, and only if the mentor confirms forward-pass timing (if scoring is end-to-end, the forward is ~1% of the run and none of this moves the score).

| field | value |
|---|---|
| **Architecture** | `e1f_gate`, ch **53**, nb 16, scale 2, res_scale 0.1 — **unchanged** |
| **Loss** | `Charb + 0.15·(1−SSIM) + 0.05·grad + 0.05·LPIPS` **+ 0.05·Laplacian-pyramid Charbonnier** (3 levels, λ = 1.0/0.5/0.25), computed in fp32 with autocast explicitly disabled, matching `_ssim_fp32` |
| **Changed flag(s)** | `--w-lap 0.05` (new) — **exactly one** |
| **Fixed** | data `semicon_train_data` (4,785) · `--seed 0 --split-seed 0 --amp` · `--p-real 0.5 --wide-p 0.5` · `--w-ssim 0.15 --w-grad 0.05 --w-lpips 0.05` · 120 epochs · 500 iters · batch 32 · crop 64 · AdamW 2e-4 wd 1e-5 · Cosine η_min 0.02·lr · grad-clip 1.0 |
| **Control** | `e1f_gate` as already trained (`61587e96f554`) — no re-run needed |
| **Compute** | ~3.5 h, one A100-80GB |
| **Metrics** | PSNR/SSIM/LPIPS on 297 and 1,197 · HF energy ratio, HF correlation, grad ratio (n=150) · **per-image HF ratio, to count any image > 1.0** · SSIM-losses-to-bicubic count · f90 quartile table · latency b1/b32 · **`last.pt` AND `best.pt` both retained** |
| **Required improvement** | HF ratio ≥ **0.32** *and* SSIM ≥ 0.62104 *and* **0 images with HF ratio > 1.0** |
| **Abort** | HF ratio ≤ 0.29, or ≥ 1 image over 1.0, or SSIM < 0.616 |

**Why this has the highest information value:** the HF deficit (0.2823 vs 1.0) is the largest measured gap in the system and the direct quantitative statement of "images look smoother than GT". Every other candidate addresses a correlate. It is also the only untested *class* of intervention — §2.5 rejected a broadband first-difference lever, and this tests whether the failure responds to per-band supervision or is genuinely information-limited. **Either outcome is decisive:** success gives a second independent improvement on the same failure mode; failure closes the loss-design branch entirely and directs all remaining effort to data or architecture.

---

## Documentation claims that are incorrect or unsupported

### ❌ 1. Calling the 1,197 excluded images "OOD" — **INCORRECT, and the most important one**

Used in `docs/DAY_LOG_17SEP.md:104` ("OOD (new dataset)"), `:271`, `:310` (the decision rule itself), and throughout `tools/protocol.py` (`--ood`, `-ood.csv`, printed as `OOD n=`).

The repository's own measurements contradict it:
* 4,785 + 1,197 = 5,982 = exactly **80.00% / 20.01%** — an ordinary random split of one corpus.
* Content distribution matches the training pool (f90 KS D = 0.111; ranges 0.039–1.124 vs 0.031–1.121).
* Degradation parameters unchanged.

**It is a held-out in-distribution test set.** Genuine OOD sets exist in the repo and are labelled as such (`data512` = 256→512, a scale never trained on). Calling a random 20% split "OOD" overstates the generalisation claim, and it propagates into a **pre-registered decision rule**, where condition 2 reads as a robustness test but is in fact a second in-distribution test.

**Fix:** rename to `--heldout` / "held-out (1,197)" everywhere, and state that the rule's condition 2 tests *statistical power*, not distribution shift.

### ⚠️ 2. "Zero hallucinations" — **the metric exists and is sound; the phrasing overstates it**

The claim is **better supported than expected**. `tools/hf_energy.py` integrates output energy *above the input's Nyquist* — the band where content is necessarily invented — and `README.md` reports **0.421 for the model vs 0.375 for bicubic, with 0 of our outputs and 14 of bicubic's crossing 1.0**. `docs/ENGINEERING_LOG.md` §11.4 records that an earlier per-image 95th-percentile version was discarded for labelling bicubic a hallucinator, and adopted the rule *"when a metric flags something that is impossible, fix the metric."* That is good practice.

Three precise limitations:

1. **It is a global energy test.** "No image exceeds GT energy above input-Nyquist" does **not** imply "no invented structure" — a model could place the correct *amount* of HF energy in the wrong *places*. The placement evidence is separate: HF correlation 0.3261 vs bicubic 0.2139, better on **150/150**. Both should be quoted together; neither alone supports the claim.
2. **It was measured for the SHIPPED model, not e1f_gate.** My §4.8 numbers use a *different* band (above 0.5 of output Nyquist, from `kla2/tools/edge_metrics.py`), which also gives 0/150 — but the README's headline figure has **not** been recomputed for e1f_gate. **If e1f_gate ships, `tools/hf_energy.py` must be re-run before the claim is repeated.**
3. **Sample size.** 0/150 images (edge metrics) and 0/297 (hf_energy) bound the rate at roughly < 2% and < 1.2% at 95% confidence — not zero.

**Suggested wording:** *"No output in 297 test images placed more energy above the input's Nyquist limit than the ground truth contains (model 0.421 vs bicubic 0.375, 0/297 over 1.0), and high-frequency correlation exceeded bicubic on 150/150 — so what detail is reconstructed is in the right place."*

### ⚠️ 3. "e1a_base is bit-identical to the shipped network" (`kla2/tools/e1.sh` header)
True of the **architecture** (forward max\|diff\| = 0.000e+00) and false of the **weights** (max\|diff\| = 0.5608). Needs one clarifying word.

### ⚠️ 4. `docs/REPRODUCIBILITY.md` shipped command says `--data data/train`
`train/` on this disk is **round-1 photographs, not SEM** (`docs/DATA.md`). Following the documented command literally reproduces the wrong model.

### ⚠️ 5. `docs/per_image_stats.csv` is duplicated
5,876 rows / 4,785 unique ids; 1,091 duplicated, byte-identical. Not consumed by any run here (`--hard-w` = 0.0 throughout), but it blocks candidate C.

### ⚠️ 6. "Epoch-to-epoch validation noise is ~0.25 dB" (`DAY_LOG_17SEP.md`, inside the decision rule)
Measured last-5 spread at convergence is **0.009–0.049 dB** across all six rows — 5–25× smaller. The rule's own tie-band of 0.10 dB is therefore far more conservative than the data requires.

### ✅ Claims that CHECK OUT
* Shipped loss formula and weights — verified in source.
* fp32/TF32 guards on SSIM and LPIPS — verified in source.
* Shipped model scores 23.632 / 0.60791 / 0.19288 — reproduced independently.
* All E1 rows share data, seed, split, mixture, optimiser, schedule, epochs, batch, crop and loss flags — verified in `e1.sh`.
* e1f_gate is parameter-matched to the control (1,346,360 vs 1,368,705, −1.6%).
