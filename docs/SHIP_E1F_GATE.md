# What ships: e1f_gate + five TensorRT engines

> Historical gate release, superseded by the shared-head model. Its checkpoint, TensorRT environment and performance values are not the current deployment. See [current release status](RELEASE_STATUS.md).

Final state, 18 September 2026. Every number below was measured, not estimated;
where a number replaced an earlier one, the earlier one was wrong and the entry
says so.

## The model

| | |
|---|---|
| architecture | `e1f_gate` -- EDSR trunk with NAFNet SimpleGate blocks and per-pixel LayerNorm2d |
| parameters | 1,346,360 (the superseded model had 1,368,705) |
| `models/model.pt` sha1 | `f95377a21e86b02ad3add4f68c784b439ae36e53` |
| training checkpoint | `checkpoints/e1f_gate-120--CANDIDATE.pt`, sha1 `61587e96f554f32d7538651664b58e4dfe02f2ac` |

The two hashes differ because `swap.py` strips the checkpoint to
`{state_dict, config}` on the way in. The weights are bit-identical -- 104
tensors, verified by `tools/bless_engines.py`, which refuses the re-key if even
one weight differs in the seventh decimal.

## Scores on the organisers' 297-image test set

| | e1f_gate | previous (`pr50-w50-lp05-120`) | delta |
|---|---|---|---|
| PSNR | **23.8297** | 23.632 | **+0.198 dB** |
| SSIM | **0.62103** | 0.60791 | **+0.0131** |
| LPIPS | **0.18126** | 0.19287 | **-0.0116** (lower is better) |

## Robustness -- the reason it is safe to ship

The previous model was chosen partly for not collapsing outside its training
noise range. The candidate had never been tested there, so both were swept on
the same GPU, the same images and the same seeds rather than against a recorded
number from another machine. Gain is model PSNR minus bicubic PSNR.

| sigma_mul | e1f_gate | previous | |
|---|---|---|---|
| 0.15 (the real data's level) | **+2.15** | +2.07 | |
| 0.25 | **+3.53** | +3.40 | |
| 0.35 | **+4.96** | +4.86 | out of training range |
| 0.45 | **+6.21** | +6.12 | out of training range |

Higher at every level, with SSIM higher and LPIPS lower throughout, and the
margin *widening* as noise grows. The failure mode being guarded against would
appear as the candidate crossing below the incumbent somewhere past the training
range. It never does. Raw data: `docs/noise_sweep_e1f_gate.csv`,
`docs/noise_sweep_forgex.csv`.

## The quality/speed knob

`--depth N` keeps the first N of 16 residual blocks. Because the tail is
zero-initialised and the network adds a correction to a global bicubic skip, a
shallower run produces a *smaller correction* -- it degrades toward bicubic,
never toward garbage. `MIN_SAFE_DEPTH = 3`; below that the curve is not monotone
and the flag clamps.

`models/knob_datasheet.json` is measured on the target GPU by
`tools/calibrate_knob.py`; `--budget-ms` reads it and picks the **best quality
inside the budget**, not the deepest setting. On this checkpoint **no row is
Pareto-dominated** -- the superseded model had depth 14 slower *and* worse than
13, a stop an operator must never select. Every stop here is a real trade.

Depth 15 -> 16 gains +0.003 dB PSNR but **+0.029 SSIM**: the last block barely
moves PSNR and matters a lot for structural similarity.

## Five TensorRT engines

`--depth` changes the graph, so one plan cannot serve every depth. Five plans
are published, at depths **3, 6, 10, 13, 16**.

| | |
|---|---|
| precision | FP32 with TF32 tensor cores. TensorRT >= 10.7 networks are strongly typed, so `BuilderFlag.FP16` no longer exists and arithmetic follows the ONNX dtypes. The bicubic skip and residual addition stay FP32, as the spec requires. |
| portability | `HardwareCompatibilityLevel.AMPERE_PLUS` -- deserialises on any sm_80+ GPU, not only the build GPU. Still locked to TensorRT 11.3. |
| shapes | batch 1-64, H/W 32-512, non-square allowed; images are grouped by shape at run time |
| build | 145 s for all five, from `models/model.pt`, via `tools/trt_native.py all --depths 3,6,10,13,16 --portable` |
| size | 7.87 / 13.32 / 20.59 / 26.04 / 31.50 MB |

Engines are **build artefacts, not source**: compiled machine code keyed to one
GPU family, one TensorRT version and one checkpoint sha1. They are gitignored and
shipped in the submission bundle; the metadata (`index.json`, `*.plan.json`) is
committed so the shipped set is described in git.

### Correctness

| check | result |
|---|---|
| ONNX vs PyTorch, 5 engines x 7 shapes (CPU, true FP32) | worst **4.77e-07** |
| TensorRT vs PyTorch reference on the build GPU | 1.06e-04 to 2.78e-04 (TF32, grows with depth) |
| PSNR cost of that deviation, measured on all 297 | **0.0001 dB** |
| output contract (dtype, shape, finite, [0,1], trailing axis) | pass, including mixed-shape folders |

### Speed, end to end over the 297, as the task is scored

H100 NVL, median of 7 reps after 2 warmup reps discarded, interleaved round-robin.

| depth | TensorRT | PyTorch | speedup | dPSNR |
|---|---|---|---|---|
| 3 | 1.513 s | 2.364 s | **1.56x** | 0.0000 |
| 6 | 1.614 | 2.332 | 1.45x | 0.0001 |
| 10 | 1.801 | 2.414 | 1.34x | 0.0000 |
| 13 | 1.916 | 2.469 | 1.29x | 0.0001 |
| 16 | 2.046 | 2.586 | **1.26x** | -0.0001 |

Warm steady-state at depth 16 is **1.61 s (184 img/s)**; the table's 2.046 s is
per-invocation with the engine cold, which is what a judge running one command
sees. Both are real and answer different questions.

The speedup *shrinks* with depth because the win is mostly the fixed startup
floor -- a TensorRT process starts in 0.52 s against PyTorch's 1.71 s -- and
because a deeper plan is a bigger file to deserialise. About 60% of what the knob
saves on the TensorRT path is less compute and 40% is loading a smaller engine, a
mechanism that does not exist on the PyTorch path.

### The knob is only a real control on TensorRT

Difference of medians against the pooled standard error of those medians:

| step | TensorRT | PyTorch |
|---|---|---|
| d3 -> d6 | +0.101 s, **4.6 sigma** | -0.032 s, **INVERTED** |
| d6 -> d10 | +0.187 s, 10.2 sigma | +0.082 s, 2.3 sigma (too close) |
| d10 -> d13 | +0.115 s, 5.7 sigma | +0.055 s, 1.4 sigma (too close) |
| d13 -> d16 | +0.130 s, 5.8 sigma | +0.117 s, 3.2 sigma |

All five TensorRT stops are distinguishable. Three of PyTorch's four steps are
not, and one runs backwards -- there, the three blocks removed are lost inside
`import torch`, which is 54% of the run. Removing the startup floor is what turns
the knob from a plot into a control.

### Batch size

Measured on the TensorRT path over the 297:

| batch | 1 | 4 | 8 | 16 | **32** | 64 |
|---|---|---|---|---|---|---|
| end-to-end | 7.06 s | 2.71 | 2.11 | 1.80 | **1.61** | 2.32 |
| img/s | 42.1 | 109.7 | 141.0 | 165.2 | **184.4** | 127.9 |

32 is optimal on both backends. Batch 1 costs 4.4x -- the per-launch overhead of
serving one image at a time. 64 regresses because the engines are built with
`opt=32`, so TensorRT tuned its kernel tactics at that exact shape.

## How a run is dispatched

`run_fast.py` imports **numpy only**, reads `.npy` headers (never pixels) to
learn the shapes, and `os.execv`s into the right runner -- the decision must be
made before `import torch`, or probing spends the entire saving. The probe costs
about 0.12 s.

It falls back to `run.py` -- unchanged, PyTorch, 12/12 stress -- when TensorRT is
missing, no engine matches the requested depth, `models/model.pt` does not match
the engines' recorded sha1, an input lies outside the shape profile, or a flag
TensorRT cannot honour is present (`--tta`, `--half`, `--device cpu`). If
TensorRT fails at run time before any output is written, `tools/trt_infer.py`
execs `run.py` itself. Being wrong costs a tenth of a second, never a result.
15 dispatcher branches and the runtime fallback are tested; fallback outputs are
bit-identical to a direct `run.py` run.

## Reproducing this

```
python tools/trt_native.py all --depths 3,6,10,13,16 --portable   # 145 s
python tools/bless_engines.py --verify-shipped ../verify_shipped.py
python tools/calibrate_knob.py --data ../semicon_test_data --rounds 5
python tools/engine_report.py --test ../semicon_test_data --reps 9 --warmup 2
python tools/knob_graphs.py --report docs/engine_report.json --out docs/knob_engines.png
```

`tools/engine_report.py --from-json docs/engine_report.json` recomputes every
table and verdict from the stored per-rep times without touching a GPU.

## Corrections made while producing this

Recorded because each one changed a conclusion:

1. Timing spread was first computed with `pstdev` next to a **median**. One I/O
   stall in five reps then reported a knob whose stops are 5-10 sigma apart as
   unresolvable. Fixed to a robust sigma (1.4826 x MAD).
2. Adjacent stops were then compared against a *per-sample* sigma pooled from the
   worst cell, rather than the standard error of the medians being compared.
   That understates separation by about sqrt(n).
3. LPIPS was published as **0.17935**. `tools/engine_report.py` appended one
   LPIPS value per BATCH and took an unweighted mean over batches; at batch 32
   the final batch holds 9 of the 297 images, so weighting it 1/10 instead of
   9/297 pulled the figure down. The true per-image mean is **0.18126**, which
   is what `src/validate.py` (0.18130) and `kla2/results/final.csv` (0.18126)
   independently report. Reproduced exactly from the per-image CSV before being
   corrected. PSNR and SSIM were always per-image and were never affected.
4. `bless_engines.py` first proved the weights match by comparing outputs at
   `1e-5`. The reference was computed on the build GPU under TF32 and re-run on
   CPU under true FP32, so correct weights disagreed at 1e-4 and were refused.
   Replaced with an exact tensor-by-tensor state_dict comparison.
