# ForgeX / KLA PS01 — work log, 17 September 2026

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> The Grand Finale ships **`e1f_gate-120`** — 1,346,360 params, **23.8297 PSNR / 0.62104 SSIM / 0.18126 LPIPS**, `models/model.pt` sha1 `f95377a21e86…`. Numbers below are kept as the record of how that decision was reached, not as a description of what ships.
>
> Current: [`SHIP_E1F_GATE.md`](SHIP_E1F_GATE.md)


Everything below is measured, with the file or command that produced it. Numbers
without a provenance note were produced today.

---

## 1. Starting position

| | |
|---|---|
| Shipped model | ForgeX, EDSR-style residual CNN, ch 64 / nb 16, **1,368,705 parameters** |
| Checkpoint | `models/model.pt`, sha1 `e208d13d62b3ded7b19954c273e80355e718a5f4`, 5,493,743 bytes |
| Recorded score | **23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS** on the organisers' 297-image test set (`docs/queue/testset_results.csv`) |
| New input today | `finalDataset/semicon_train_data_excluded` — 1,197 GT/NoisyLR pairs |

---

## 2. Submission hardening — all green

The floor was locked before any money was spent.

- **Snapshot** taken at `Desktop/kla/SUBMISSION_SNAPSHOT_17SEP/` (`run.py`, `model.pt`,
  `requirements-inference.txt`). Restores a known-good state in thirty seconds.
- **Hash verified**: `e208d13d62b3ded7b19954c273e80355e718a5f4`, 5,493,743 bytes. Correct
  weights, matching `docs/CHECKPOINTS.md`.
- **New: `tools/stress_run.py`** — 13 hostile input cases against `run.py`. **13/13 PASS**:
  square 128, single image, 64², 256², 512², non-square 180×220, odd 127×129, trailing
  `(H,W,1)` axis, float64, values outside [0,1], NaN/Inf, mixed sizes in one folder, and
  being handed the dataset directory instead of the images directory. Contract held on
  every one — correct 2× shape, float32, finite, inside [0,1], filenames preserved.
- **End-to-end validation**: the real `run.py` on 150 real unseen images — 150 files in,
  150 out, **0 contract violations**, scoring **23.4249 / 0.59811**. That agrees with the
  independent measurement pipeline to **+0.0000 dB / +0.00000 SSIM**, which proves nothing
  is silently broken between the two code paths.

---

## 3. The new dataset, characterised

### 3.1 It is the organisers' own 20% holdout

```
original train set   4,785
new "excluded" set   1,197
total                5,982
split                79.9900% / 20.0100%
```

An exact 80/20 split to within 0.01%, and the folder is literally named
`semicon_train_data_excluded`. **Inference: this is the 20% they held back from round 2.**
If they have released their holdout, the final test set is something else — which makes
training on these 1,197 legitimate, and means the final test data is genuinely unseen.
**Confirm with the mentor.**

### 3.2 Content: the same distribution, no new morphology

Using the repo's own `f90` metric (radial frequency holding 90% of non-DC energy):

| | n | mean | sd | p5 | median | p95 | range |
|---|---:|---:|---:|---:|---:|---:|---|
| original | 5,876 | 0.579 | 0.330 | 0.080 | 0.622 | 1.035 | 0.031–1.121 |
| new | 150 | 0.521 | 0.337 | 0.081 | 0.501 | 1.078 | 0.039–1.124 |

- difference in means = **17.5% of one standard deviation**
- Kolmogorov–Smirnov **D = 0.111**, p = 0.049; Mann-Whitney p = 0.04
- **1 of 150** new images falls outside the original range
- share above the original 75th percentile (the fine-structure end): original 25.0%, new **20.7%**

**The new data contains no morphology the model has not already seen.**

### 3.3 Degradation: unchanged

Fitted on the new set: same law `var = σ_add² + σ_mul²·m² + c·v`, speckle over Poisson by
**ΔBIC 252.3**. M1 coefficients `[-0.00106, 0.04018, 0.49661]` against the original
`[0.00004, 0.03649, 0.45944]` — within a few percent.

### 3.4 Conclusion

**More of the same data, not different data.** It gives statistical power, not
distribution coverage. Scoring well on it demonstrates *reproducibility*, not
*robustness* — a different and weaker claim. Say so precisely.

---

## 4. Model performance on the new data

Three independent measurements, converging:

| sample | PSNR | SSIM | LPIPS | gain over bicubic |
|---|---:|---:|---:|---:|
| 27 images | 23.4691 | 0.59893 | — | +3.3896 ± 0.4759 |
| 150 images | 23.4249 | 0.59811 | — | +3.156 ± 0.177 |
| **all 1,197 (A100)** | **23.4111** | **0.59222** | **0.19702** | **+3.070** |
| *(original 297-image test set, recorded)* | *23.632* | *0.60791* | *0.19288* | — |

The 150-image estimate landed within **0.014 dB** of the full 1,197 measurement.

**And the new set is easier than our own validation split.** Measured in the same run:

| | n | PSNR | SSIM | LPIPS | gain |
|---|---:|---:|---:|---:|---:|
| ID (val split from train) | 200 | 23.1041 | 0.58837 | 0.20016 | +2.880 |
| OOD (new dataset) | 1,197 | 23.4111 | 0.59222 | 0.19702 | +3.070 |

The "drop" is **−0.307 dB** — i.e. it generalises *upward*. **0 of 150 images where
bicubic beats the model on PSNR.** The model never hurts.

---

## 5. The failure mode, quantified

Gain over bicubic correlates negatively with ground-truth fine texture:

| | n = 27 | n = 150 |
|---|---:|---:|
| PSNR gain vs texture | −0.512 | **−0.461** |
| SSIM gain vs texture | −0.520 | **−0.545** |

Split at the median texture level (n = 150):

| | PSNR | SSIM | gain | SSIM losses to bicubic |
|---|---:|---:|---:|---:|
| smooth half (75) | 25.90 | 0.703 | +4.45 dB / +0.1723 | **0 / 75** |
| textured half (75) | 20.95 | 0.494 | +1.86 dB / +0.0233 | **15 / 75** |

On the **30 most textured images, bicubic beats us on SSIM 10 times.** Worst deficit
−0.1879. PSNR is never hurt; SSIM is hurt precisely where fine structure lives.

**Diagnosis:** regression-to-the-mean blur. Under an L1-dominant loss the optimal output
is the conditional average of every plausible fine texture, which is a blur. Smoothing
always helps MSE and can destroy the local covariance SSIM rewards — hence PSNR up, SSIM
down, on exactly the images with the most structure.

This also explains two older results that never sat right: why eight architectures landed
inside 0.12 dB on PSNR, and why 78× more parameters bought only 0.29 dB. Every one of
those runs optimised the same objective and converged to the same estimator.

---

## 6. Two levers measured and killed

| lever | measured | verdict |
|---|---:|---|
| ×8 dihedral TTA | **+0.0463 dB ± 0.0063** (helps 27/27 images) | Real but ~1/20th of the +0.1–0.3 dB predicted. The model trains with dihedral augmentation and carries a bicubic skip, so it is already near-equivariant. Not worth 8× compute for quality; useful only as a latency/quality switch. |
| oracle per-image routing (model vs bicubic) | **+0.0016 SSIM** | Worthless. Routers buy nothing here. Expert *specialisation* might; fallback does not. |

---

## 7. Latency, measured on evaluation-class hardware (A100-SXM4-80GB)

`tools/protocol.py`, fp16, channels_last, cudnn.benchmark, 10 warm-up + 50 timed,
CUDA-synchronised both sides:

| batch | ms/img median | p95 | img/s | peak VRAM |
|---|---:|---:|---:|---:|
| 1 | 2.824 | 2.840 | 354 | 0.07 GB |
| 8 | 0.802 | 0.803 | 1,247 | 0.45 GB |
| 32 | 0.662 | 0.666 | 1,511 | 1.74 GB |
| **64** | **0.634** | 0.639 | **1,577** | 3.45 GB |

Compute: **44.83 GFLOPs/image** (conv + linear MACs ×2).

Three findings:

1. **Batching is worth 4.5×, free and lossless.** 354 → 1,577 img/s, identical output.
2. **The A100 at batch 1 (2.824 ms) is *slower* than an RTX 4050 (2.168 ms, recorded.)**
   At batch 1 a 1.37 M model is pure kernel-launch overhead and the A100's lower clocks
   lose. Bigger GPUs only help with batching.
3. ~~**The forward pass is ~1% of a scored end-to-end run.**~~ **RETRACTED 18 Sep — see
   below.** The original text read: *"297 images at batch 64 is 0.19 s of model compute,
   against ~17 s end-to-end where `import torch` alone is 82% (recorded profile). Round 2
   scored end-to-end; if round 3 does too, model size is almost irrelevant to the time
   score."* Both figures are wrong.

   **Correction (measured 18 Sep with `run.py --timing`, CUDA events + single terminal
   sync, 297 images, batch 32, fp16, depth 8):**

   | stage | seconds | ms/image | % of end-to-end |
   |---|---:|---:|---:|
   | import torch + numpy | 1.626 | — | **37.9%** |
   | checkpoint load + model to device | 0.091 | — | 2.1% |
   | host → device | 0.017 | 0.0587 | 0.4% |
   | **MODEL FORWARD** | **0.983** | **3.3107** | **22.9%** |
   | device → host + postprocess | 0.663 | 2.2337 | 15.4% |
   | **END-TO-END** | **4.295** | **14.4600** | 100% |

   * **The forward pass is 41.9% of an end-to-end run at full depth** (22.0% at depth 3),
     not ~1%. Measured by `tools/timing_sweep.py`, 5 interleaved reps per depth, spread
     2.0–4.4%:

     | depth | fwd ms/img | spread | img/s | fwd s | e2e s | fwd % of e2e | datasheet ms | ratio |
     |---:|---:|---:|---:|---:|---:|---:|---:|---:|
     | 3 | 2.0906 | 3.7% | 478.3 | 0.621 | 2.829 | 22.0% | 1.5184 | 1.38× |
     | 8 | 3.2527 | 4.4% | 307.4 | 0.966 | 3.177 | 30.5% | 2.7436 | 1.19× |
     | 13 | 4.6042 | 2.2% | 217.2 | 1.367 | 3.605 | 37.9% | 3.9720 | 1.16× |
     | **16** | **5.3062** | 2.0% | 188.5 | 1.576 | 3.762 | **41.9%** | 4.7082 | 1.13× |
 Depth 8 does *half* the work of the shipped model, so the shipped
     forward cannot be 0.19 s. The 0.634 ms/img figure behind the original claim implies
     ≈71 TFLOPS sustained on a 75 W RTX 4050, which that part cannot do in dense fp16; it
     was almost certainly timed without a `torch.cuda.synchronize()`, capturing kernel
     *launches* rather than kernel *execution*.
   * **`import torch` is 37.9%, not 82%.** The 82% came from a ~17 s cold-cache run.
     Warm, the import is a roughly fixed ≈1.6 s, so its share falls as the rest speeds up.
   * **Consequence:** model size and forward cost are *not* irrelevant to an end-to-end
     time score. The depth knob spans a measured 24.8% of end-to-end wall clock — 0.933 s of a 3.762 s run (`docs/KNOB.md`).
   * Read `MODEL FORWARD`, not `GPU path total`: `device → host` is wall-clocked and
     `.cpu()` blocks until queued GPU work drains, so it partially double-counts the
     kernels the CUDA events already measured.

---

## 8. The MDTA / attention block, priced

Measured at our trunk width (ch 64, 128×128 feature map):

| block | params | GMAC | vs ResBlock | CPU ms |
|---|---:|---:|---:|---:|
| ResBlock (ours) | 73,856 | 1.208 | 1.00× | 21.6 |
| SE / channel attention | 74,436 | 1.208 | **1.00×** | **21.2** |
| SimpleGate block | 110,912 | 1.812 | 1.50× | 27.3 |
| MDTA only | 18,116 | 0.330 | 0.27× | 45.1 |
| TransformerBlock (MDTA+GDFN) | 54,072 | 0.915 | 0.76× | 83.1 |

Two things this settles:

- **The attention matmul is 33.6 of MDTA's 330 MMAC — 10%.** The other 90% is a 1×1 QKV
  projection and a 3×3 depthwise. And MDTA's attention map is **C×C, over channels, not
  space**: at ch 64 it is a 64×64 matrix re-weighting channels from globally-pooled
  statistics. It does not connect distant pixels.
- **Channel attention is free** — identical FLOPs and wall-clock to a plain ResBlock,
  +580 parameters. The Transformer block is cheaper in FLOPs (0.76×) but **3.85× slower in
  wall clock**, because softmax / transpose / LayerNorm are memory-bound.

This is why the E1 ladder has an SE control: without it, an MDTA win would be reported as
"global attention helps" when the finding may be "any channel re-weighting helps".

---

## 9. A correction to our own reading of the 8-architecture study

The study has been quoted as "the field spans 0.12 dB". That is a **PSNR-only** statement
and it buried the signal. The same table's SSIM and LPIPS columns:

| | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| edsr_base | 22.1021 | 0.52616 | 0.16116 |
| abl_nostem | 22.1095 | 0.52700 | 0.16107 |
| forgex (ours) | 22.1141 | 0.52764 | 0.16077 |
| abl_noskip | 22.0941 | 0.52860 | 0.16081 |
| unet | 22.0872 | 0.53084 | 0.15918 |
| **nafnet** | 22.1406 | **0.54204** | 0.15626 |
| **swinir** | 22.2083 | **0.55323** | 0.15677 |
| **restormer** | 22.1884 | **0.55388** | **0.15355** |

The four EDSR-family rows sit inside **0.0024 SSIM** of each other. NAFNet is **+0.015**
above that cluster; the two attention models are **+0.026** — six and eleven times the
within-family spread, at matched parameters, same data and schedule. LPIPS orders
identically and independently.

That is not noise, and it lands on the exact metric where we measured the failure.

---

## 10. New insight: test-set luck is bigger than the architecture effect

Bootstrap of our own per-image scores (4,000 resamples):

| test set size | PSNR sd from sampling alone | 95% range |
|---|---:|---|
| **297 images** (round 2's size) | **0.212 dB** | **0.84 dB wide** |
| 1,197 images | 0.107 dB | 0.43 dB wide |

Against what we've measured:

- entire matched-parameter architecture field: **0.12 dB**
- 78× more parameters (17k → 1.37 M): **0.29 dB**
- **test-set luck on a 297-image draw: 0.212 dB**

**Which 297 images the organisers draw matters roughly twice as much as which architecture
we pick.** Two teams with identical models could differ by half a decibel from the draw
alone. Consequence: report **"23.41 ± 0.21 dB on a 297-image draw"**, not "23.41 dB".

---

## 11. Training is CPU-bound, not GPU-bound

Observed during E1: all three A100s sustained at **0–3% utilisation**, while epochs take
89–120 s — the same as the recorded 93 s/epoch for this model on a *single RTX PRO 4500
running solo*. The GPUs are idle waiting for batches; the work is 72 dataloader workers
doing per-sample numpy (load, crop, shifted-Gamma draw, box-downsample, dihedral).

**This is a confound in numbers we already relied on.** The recorded study's `s/epoch`
column — forgex 93, nafnet 85, unet 80 — is likely the **dataloader floor** for the cheap
architectures, not a measurement of them. Only the expensive rows (restormer 189, swinir
364) broke through it.

So `s/epoch` was never a valid cost metric for the cheap models. The valid one is
`protocol.py`'s inference ms/image at fixed batch, which is what the decision uses.

---

## 12. Code built today (all on the laptop, in the repo)

| file | what it does |
|---|---|
| `tools/protocol.py` | **The** benchmark. One protocol: quality on a fixed id list + OOD set, latency swept over batch sizes with warm-up and sync, FLOPs by hooks, VRAM, git sha and weight sha1 per row, per-image CSV |
| `tools/stress_run.py` | 13 hostile input cases against `run.py`. Protects the whole submission |
| `tools/pod_bootstrap.sh` | One-command pre-flight: GPUs, cores, deps, registry, all architectures build, data counts, checkpoint hash |
| `tools/gpucheck.py` | MIG-safe device count via `torch.cuda.device_count()` with a real allocation test per device |
| `tools/e1.sh` | The ladder. Round-robin GPU pinning, resumable from `last.pt`, `name=arch` rows, staggered launch |
| `tools/e1_status.sh` | Live per-row progress plus a projected finish time from recent pace |
| `tools/e1_score.sh` | Scores every finished row through `protocol.py` and builds the dashboard |
| `tools/dashboard.py` | Self-contained HTML: table with diverging Δ-vs-control bars, a shaded ±0.10 dB tie band, training curves, no external dependencies |
| `arch/attn.py` | The six E1 variants. **Control is bit-identical to the shipped model** — `strict=True` load, `max|diff| = 0.000e+00` |
| `arch/scaled.py` | `rcan`, `rcan_ours`, `nafnet_deep`, `nafnet_xdeep`, `swinir_deep`, `restormer_deep` + a quadratic-bracket width solver |
| `tools/bench_h100.py` | Random-weight throughput sweep over architecture × size × batch, prices `channels_last`, `torch.compile` and ×8 TTA |
| `tools/register_scaled.py` | Idempotent registry patch. Registry now exposes **25 architectures** |
| `tools/merge_data.py` | Dataset merge with an md5 duplicate check that refuses on overlap |

---

## 13. E1 — running now

Six rows, 120 epochs, matched to ~1,368,705 parameters, identical data / split / seed /
degradation mixture / optimiser / schedule / batch. Three A100-SXM4-80GB, two rows per card.

| row | question it answers |
|---|---|
| `e1a_base` | control — bit-identical to the shipped network |
| `e1b_se` | does **free** channel attention get the SSIM? |
| `e1c_mdta1` | does Restormer's MDTA beat plain channel attention? |
| `e1f_gate` | was NAFNet's edge the SimpleGate rather than the attention? |
| `e1h_nafdeep` | real NAFNet at its proper 32-block shape (the recorded study used 5 blocks — a genuine flaw) |
| `e1g_grad20` | same network, gradient loss 0.05 → 0.20: **is the blur a loss problem?** |

Progress at last check: epochs 13–18 of 120. **ETA ~04:40 IST.** `e1.sh` runs
`e1_score.sh` itself when the last row finishes, so `results/dashboard.html` appears
without intervention.

### The decision rule, fixed in advance

Swap the shipped model **only** if a row clears **all** of:

1. ≥ **0.10 dB** PSNR *or* ≥ **0.010** SSIM over `e1a_base`
2. no loss on the OOD column
3. `e1a_base` itself landed near the shipped model's number — if the control is off, the
   harness is wrong and every delta is meaningless

Epoch-to-epoch validation noise is ~0.25 dB; scoring uses the **mean of the last five
epochs**, and even then a gap under ~0.10 dB is a tie.

---

## 14. Open questions for the mentor

1. **How is the score weighted** across PSNR / SSIM / LPIPS / inference time?
2. **Is the whole process timed, or just the forward pass?** Round 2 was end-to-end.
   *(Updated 18 Sep: the earlier note here said model size is "~irrelevant" under
   end-to-end scoring. That rested on the retracted ~1% figure in §7 and is withdrawn —
   the forward pass is 22.0–41.9% of an end-to-end run, so forward cost matters either way.
   The answer still decides how much it matters, not whether.)*
3. **Batch-1 or folder?** Decides whether 2.824 ms or 0.634 ms is our real number.
4. **Is TTA allowed?**
5. **Is `semicon_train_data_excluded` your held-out 20%, and is the final test separate?**

Question 1 has money in it: `pr50-w50-120` scores **23.763 / 0.61915 / 0.35518** against
the shipped **23.632 / 0.60791 / 0.19288**. We deliberately traded **+0.13 dB and +0.011
SSIM** for a much better LPIPS. If LPIPS is lightly weighted, switching back is a larger
gain than E1 is likely to produce, for zero GPU time. (That checkpoint is not on the
laptop — only the shipped one is.)

---

## 15. Infrastructure notes worth keeping

- **Never `scp -r` thousands of small files.** 9,570 × 256 KB ran at ~200 KB/s because each
  file finishes before TCP/SFTP ramps — you measure the slow-start, repeatedly. One `tar`
  ramps once and holds. Same root cause as the croc failures last week.
- **RunPod's `/workspace` is geesefs (FUSE over S3).** No real ownership — `tar` fails on
  utime/chown — and reading thousands of small files during training would be crushingly
  slow. Use local disk (`/root/work`, on the overlay).
- **Vast's PyTorch images keep torch in `/venv`**, which a non-interactive
  `ssh host "cmd"` does not activate. `source /venv/main/bin/activate` first.
- `nvidia-smi -L | wc -l` **over-counts on MIG hosts** (it prints the parent card too).
  Use `torch.cuda.device_count()` — that's what `CUDA_VISIBLE_DEVICES` indexes into.
- `mkdir -p logs` **before** `nohup ... > logs/x.log` — the shell opens the redirect before
  the script can create its own directory.
