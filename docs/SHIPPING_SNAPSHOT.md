# ForgeX — KLA PS01 · Final Shipping Model: Technical Snapshot

**Generated:** 18 September 2026
**Scope:** the implementation currently on disk. No proposals, no comparisons, no history.

## 0. Identity of what is being shipped

| item | value |
|---|---|
| Checkpoint | `models/model.pt` |
| **SHA-1 (full)** | `e208d13d62b3ded7b19954c273e80355e718a5f4` |
| Checkpoint mtime | 2026-09-04 16:19 |
| Entry point | `run.py` (self-contained; network inlined) |
| Git HEAD | `02e05a6` — working tree **dirty** (`run.py` modified, untracked tools) |

**Verified statement of fact:** `run.py` contains **0** occurrences of `SimpleGate`, `LayerNorm2d`
or `GateBlock`, and the checkpoint's `config` is `{'ch': 64, 'nb': 16, 'scale': 2,
'res_scale': 0.1}`. The shipping network is the ReLU-ResBlock `Restorer`.

---

## 1. Model Architecture

**Exact architecture:** EDSR-style flat residual CNN for joint denoising + ×2 super-resolution.
Single-scale trunk (no encoder/decoder), one PixelShuffle upsample at the end, global bicubic
skip. Custom class `Restorer`, inlined in `run.py`.

| property | value |
|---|---|
| Parameters (total) | **1,368,705** |
| Trainable parameters | **1,368,705** (100%; no frozen or buffer-only tensors) |
| state_dict entries | 72 tensors, all `torch.float32` |
| Input dimensions | `(B, 1, H, W)`; canonical **128 × 128** |
| Input channels | **1** (grayscale) |
| Output dimensions | `(B, 1, 2H, 2W)`; canonical **256 × 256** |
| Output channels | **1** |
| Kernel sizes | **3 × 3** on every convolution (36 of 36) |
| Strides | **1 × 1** on every convolution |
| Padding | **1 × 1** on every convolution (shape-preserving) |
| Normalization layers | **NONE** — no BatchNorm, LayerNorm, GroupNorm or InstanceNorm anywhere |
| Attention mechanisms | **NONE** |
| Downsampling operations | **NONE** — the trunk never changes resolution |

### Layer-by-layer

```
input x : (B, 1, 128, 128) float32, values in [0, 1]
  │
  ├─────────────────────────────────────────── global bicubic skip ──┐
  │                                                                  │
VarianceStabilisingStem              [custom, 0 parameters]          │
  p = x.clamp_min(0.0)                                               │
  cat([ x , sqrt(p + 1e-6) , log1p(p) ], dim=1)  → (B, 3, 128, 128)  │
  │                                                                  │
head : Conv2d(3 → 64, k3, s1, p1)                     1,728 + 64     │
  │  → (B, 64, 128, 128)                                             │
  ├──────────────────────── long skip (identity) ──────────┐         │
  │                                                        │         │
body : Sequential of 16 × ResBlock          [custom]       │         │
  each:  x + 0.1 * c2(ReLU(c1(x)))                         │         │
         c1 : Conv2d(64 → 64, k3,s1,p1)    36,864 + 64     │         │
         c2 : Conv2d(64 → 64, k3,s1,p1)    36,864 + 64     │         │
  → (B, 64, 128, 128)                                      │         │
  │                                                        │         │
body_tail : Conv2d(64 → 64, k3,s1,p1)      36,864 + 64     │         │
  │                                                        │         │
  └────────────────────── add ◄─────────────────────────────┘        │
  │  f = head_out + body_tail(body(head_out))                        │
  │                                                                  │
up : Sequential                                                      │
  up.0 : Conv2d(64 → 256, k3,s1,p1)       147,456 + 256              │
  up.1 : PixelShuffle(2)   (B,256,128,128) → (B,64,256,256)          │
  │                                                                  │
tail : Conv2d(64 → 1, k3,s1,p1)               576 + 1                │
  │  → residual (B, 1, 256, 256)                                     │
  │                                                                  │
  └────────────── add ◄── F.interpolate(x.float(), scale_factor=2, ──┘
                          mode='bicubic', align_corners=False)
  │
output : (B, 1, 256, 256) float32          [NO clamp inside forward()]
```

### Activations
* **ReLU** — `F.relu`, 16 occurrences (one per ResBlock, between `c1` and `c2`). This is the
  only nonlinearity in the network.
* `sqrt` and `log1p` in the stem are fixed, parameter-free transforms, not learned activations.

### Residual / skip connections (3 distinct kinds)
1. **Per-block residual** × 16 — `x + res_scale * branch`, `res_scale = 0.1` (fixed scalar,
   not learned).
2. **Long skip** × 1 — `f = head_out + body_tail(body(head_out))`, spanning the whole trunk.
3. **Global bicubic skip** × 1 — `bicubic_up(x) + residual`, forced to **float32**.

### Custom modules / operations
| item | nature |
|---|---|
| `VarianceStabilisingStem` | custom `nn.Module`, 0 params; `clamp_min` + `sqrt` + `log1p` + `cat` |
| `ResBlock` | custom `nn.Module` with fixed `res_scale=0.1` |
| `Restorer` | custom top-level `nn.Module`, carries a `config` dict |
| fp32 global skip | `x.float()` and `residual.float()` forced outside autocast. Comment in source: in fp16 the spacing near 0.5 is ~5e-4, so a small early residual would round out of the sum |
| Zero-init tail | `nn.init.zeros_` on `tail.weight`/`tail.bias` — **training-time only**. `run.py`'s inlined `Restorer` omits it, since weights are loaded immediately after construction |

### Depth knob (present in the shipped `run.py`)
`--depth N` truncates `model.body` to its first N blocks; `--budget-ms` selects N from
`models/knob_datasheet.json`. `MIN_SAFE_DEPTH = 3`; lower values are clamped with a warning.
**Default is full depth (16).** Verified: default output is bit-identical to the pre-knob
`run.py` (max abs diff = 0.0). `models/knob_datasheet.json` is **NOT PRESENT**, so
`--budget-ms` currently falls back to full depth with a warning.

---

## 2. Model Checkpoint

| property | value |
|---|---|
| Format | PyTorch `torch.save` ZIP archive (`.pt`) |
| Size | **5,493,743 bytes** (5.24 MiB) |
| Top-level keys | exactly **two**: `state_dict`, `config` |
| `config` | `{'ch': 64, 'nb': 16, 'scale': 2, 'res_scale': 0.1}` |
| Weight datatype | **FP32** — all 72 tensors `torch.float32` |
| Tensor elements | 1,368,705 → 5,474,820 bytes of payload (99.7% of file) |
| Inference-required only? | **YES.** No optimizer state, no scheduler, no EMA, no epoch, no RNG state, no gradients |
| Optimizer/scheduler present | **NO** |

### Exact loading procedure (`run.py`)
1. `find_weights()` — tries `models/model.pt`, then `weights/model.pt`, both relative to
   `run.py`. Exits with an error if neither exists.
2. `load_checkpoint()` — attempts `torch.load(map_location="cpu", ...)` with three kwarg sets
   in order, falling through on exception:
   `{weights_only: True, mmap: True}` → `{weights_only: True}` → `{}`.
   The first (memory-mapped, tensors-only) is the intended path.
3. `state = ck.get("state_dict", ck)` and `cfg = ck.get("config", {}) or {}`
4. `model = Restorer(**cfg).eval()`
5. `model.load_state_dict(state)` — **strict** (default)
6. `model.to(device)`
7. If CUDA: `model.to(memory_format=torch.channels_last)`

No quantization, no fusion, no compilation, no scripting, no ONNX/TensorRT export anywhere
in the shipping path.

---

## 3. Framework and Software Environment

Sourced from `requirements.txt` (documented as the pip freeze of the machine that trained
`models/model.pt`) and `.python-version`.

| item | value | source |
|---|---|---|
| Framework | **PyTorch** | `run.py` |
| Framework version | **torch 2.13.0+cu126** | `requirements.txt` pin |
| torchvision | 0.28.0+cu126 | `requirements.txt` (not imported by `run.py`) |
| Python version | **3.13** | `.python-version` |
| CUDA version | **12.6** (wheel build `+cu126`, index `download.pytorch.org/whl/cu126`) | `requirements.txt` |
| cuDNN version | **UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION** | `docs/REPRODUCIBILITY.md` explicitly notes it is unpinned |
| NVIDIA driver version | **UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION** | same |
| OS | **Windows 11** | `requirements.txt` header, `docs/LAPTOP.md` |
| numpy | 2.4.4 | `requirements.txt` |
| Runtime dependencies of `run.py` | **`torch` and `numpy` only** | `requirements-inference.txt`; `run.py` inlines the network |
| Other pins (not needed at inference) | lpips 0.1.4, scipy 1.18.0, pillow 12.2.0 | `requirements.txt` |

**Determinism setting (explicit choice, documented in `docs/REPRODUCIBILITY.md`):**
`torch.use_deterministic_algorithms` is **not** set; `cudnn.benchmark` is left at its default.
In `run.py` the default is therefore `cudnn.benchmark = False`.

---

## 4. Target Hardware

Two distinct machines appear; they are **not** the same and are separated here.

### 4a. Stated evaluation target
`run.py`'s module docstring: *"Round 2 scores END-TO-END wall clock on an **H100**."*

| item | value |
|---|---|
| GPU model | NVIDIA **H100** (stated target) |
| GPU VRAM | UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION |
| Architecture | Hopper |
| CPU / RAM | UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION |
| Measurements on this GPU | **NONE PRESENT** |

### 4b. Machine all recorded benchmarks were taken on
| item | value |
|---|---|
| GPU model | **NVIDIA GeForce RTX 4050 Laptop GPU** |
| GPU VRAM | **6 GB**, 75 W (`README.md`) |
| Architecture | Ada Lovelace |
| CPU | **13th Gen Intel Core i5-13420H** |
| RAM | UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION |
| OS | Windows 11 |

### 4c. Other machines referenced (no shipping role)
RTX 4090 pod — cross-machine reproducibility check only; A100 SXM — batch-1 latency only.

**`run.py` hard-codes no GPU assumption.** Device is auto-selected; batch defaults to 32 on
CUDA and 1 on CPU; OOM triggers recursive batch halving.

---

## 5. Inference Pipeline

```
Input .npy files in <input-dir>
  ↓
File listing + directory fallback
  ↓
Parallel read/parse (ThreadPoolExecutor, 8 workers)
  ↓
Shape grouping and batching
  ↓
NumPy → Tensor, CPU → GPU, channels_last
  ↓
Model forward (autocast fp16) under inference_mode
  ↓
Mean over TTA variants (1 variant by default)
  ↓
nan_to_num → clamp[0,1] → GPU → CPU → float32 NumPy
  ↓
Parallel write (ThreadPoolExecutor, 8 workers)
  ↓
Output .npy files in <output-dir>
```

| stage | exactly what the code does |
|---|---|
| **Input discovery** | `sorted()` of `*.npy` in `input_dir`. If empty, searches one level down for a subdirectory containing `.npy`, preferring names `noisylr, noisy, lr, input, inputs, degraded, test, images` |
| **Preprocessing** | `load_npy()`: `np.load`; if `(H,W,1)` strips the trailing axis and records `had_channel`; rejects anything not 2-D after that; `np.ascontiguousarray(a, dtype=np.float32)` |
| **Resizing** | **NONE.** No resize, crop, pad or tiling. Native resolution is processed |
| **Normalization** | **NONE.** No mean/std normalization, no scaling, no clipping of the input. Values used as-is; the `[0,1]` assumption is handled inside the model by `clamp_min(0.0)` in the stem |
| **Batching** | Images grouped by exact shape; a group is flushed when it reaches `batch`; leftovers flushed at the end. Default batch **32** (CUDA) / **1** (CPU) |
| **Tensor conversion** | `torch.from_numpy(np.stack(arrays))[:, None]` → `(B,1,H,W)` float32 |
| **CPU → GPU transfer** | `.to(device)` — synchronous, **not** pinned, **not** `non_blocking` |
| **Tensor layout** | `.contiguous(memory_format=torch.channels_last)` on CUDA; model also converted to channels_last |
| **Model execution** | `with torch.inference_mode():` and `with torch.autocast("cuda", enabled=use_amp):` → `model(v.contiguous()).float()` |
| **TTA** | `--tta` off by default. When on: 8 dihedral variants, each inverted, then `torch.stack(outs).mean(0)` |
| **Postprocessing** | `torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)` |
| **GPU → CPU transfer** | `out[:, 0].cpu().numpy().astype(np.float32)` |
| **Image reconstruction** | Split back to a list of `(2H, 2W)` arrays; `y[..., None]` re-added only if the input had a trailing axis |
| **Image saving** | `np.save(os.path.join(output_dir, name), arr)` submitted to the writer pool. Output dtype **float32**, range **[0,1]** guaranteed |
| **Final check** | Counts written files; `sys.exit` with an error if fewer than expected |

---

## 6. Inference Code Behavior

| aspect | current behaviour |
|---|---|
| Model initialization | `Restorer(**cfg)` from the checkpoint's own `config`; class inlined in `run.py` |
| Model loading | `load_state_dict(state)`, strict |
| Device selection | `cuda` if `--device` ∈ {auto, cuda} **and** `torch.cuda.is_available()`, else `cpu` |
| Evaluation mode | `.eval()` called at construction, before `load_state_dict` and `.to(device)` |
| Gradient handling | `torch.inference_mode()` around the forward. **No** `torch.no_grad()`, **no** `requires_grad_(False)` |
| Precision | `torch.autocast("cuda", enabled=use_amp)`, `use_amp = fp16 and device=='cuda'`. Weights stay FP32; autocast casts per-op. Output forced `.float()` |
| Batch size | `--batch`, default 0 → **32** on CUDA, **1** on CPU |
| OOM handling | `restore_safe()` catches `RuntimeError` containing "out of memory", calls `torch.cuda.empty_cache()`, halves the batch, recurses to batch 1 |
| Tensor creation | `torch.from_numpy(np.stack(...))` (zero-copy from the stacked array) |
| Memory transfers | Synchronous `.to(device)` and `.cpu()`. No pinned memory, no CUDA streams, no `non_blocking=True` |
| Synchronization | **Only one explicit `torch.cuda.synchronize()`**, in the background `_init_cuda()` thread. The inference path has none; `.cpu()` synchronizes implicitly |
| **Warm-up** | **NO explicit warm-up.** Stated deliberate: *"there is no separate warm-up pass; the first real batch is the warm-up"* |
| Timing implementation | `_T0 = time.perf_counter()` at the top of the file, before imports. `_mark(name)` appends `(name, elapsed)` to `_STAGES`. Wall-clock only |
| Timing stages recorded | import torch+numpy · CUDA context ready · argparse + list input dir · load checkpoint · join CUDA thread · build model + weights to device · read inputs + inference + queue writes · flush all writes |
| **Model-only timing** | **NOT INSTRUMENTED in `run.py`.** No per-forward timer exists |
| **Preprocessing timing** | Not isolated — folded into the combined read+inference+queue stage |
| **Postprocessing timing** | Not isolated — same stage |
| `--profile` | Prints the cumulative/per-stage/%-of-total table for `_STAGES` |
| Concurrency | Reader pool (8), writer pool (8), CUDA-init thread started at import |

---

## 7. Current Performance

All figures below were measured on the **RTX 4050 Laptop GPU (4b)**. **No H100 measurement
exists in the shipping implementation.**

### End-to-end (the metric `run.py` is built around)
Source: `docs/bench_results.txt`, 297 images, 9 repetitions.

| statistic | value |
|---|---|
| Median | **5.66 s** → **19.06 ms/image** → **52.5 img/s** |
| Mean | 6.04 s |
| Std dev | 1.17 s |
| Best / worst | 5.51 s / 9.13 s |
| Benchmark images | 297 |
| Repetitions | 9 |
| Warm-up runs | **0** |

Recorded stage breakdown: forward pass ≈ **1%** of an end-to-end run; `import torch` ≈ **82%**.
297 images at batch 64 = **0.19 s** of model compute against ≈17 s end-to-end.

### Model-only (synthetic-input microbenchmarks, `tools/protocol.py`)
10 warm-up + 50 timed iterations, `torch.cuda.synchronize()` both sides, fp16.

| batch | ms/image | img/s | peak VRAM | source |
|---:|---:|---:|---:|---|
| 1 | 14.56 | 68.7 | 0.123 GB | `kla2/results/final.csv` (SHIPPED row) |
| 32 | 6.4642 | 154.7 | 3.533 GB | `kla2/results/final.csv` (SHIPPED row) |
| 1 | 2.168 | 461 | — | `docs/DAY_LOG_17SEP.md` (earlier recorded run) |
| 64 | 0.634 | 1,577 | 3.45 GB | `docs/DAY_LOG_17SEP.md` |

**Documented inconsistency:** batch-1 latency is recorded as both 14.56 ms and 2.168 ms on the
same GPU and the same weights. Both values are present in the shipping repository. No
reconciled figure exists. Recorded as-is, without selecting between them.

### Compute
| item | value |
|---|---|
| GFLOPs / image | **44.827** (Conv2d+Linear MACs × 2, batch 1, 128×128; measured by forward hooks) |
| Total MACs | 22,413.3 M |
| Parameters | 1,368,705 |
| Peak training VRAM | 0.95 GB (`README.md`) |
| GPU utilization | **UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION** |

---

## 8. Restoration Quality

Checkpoint `e208d13d62b3…`, organisers' **297-image test set**, all 297 images scored.

| metric | value |
|---|---:|
| **PSNR** | **23.632 dB** |
| **SSIM** | **0.60791** |
| **LPIPS** (AlexNet) | **0.19288** |
| Gain over bicubic ×2 | **+3.177 dB** |

Supporting facts recorded for this exact checkpoint:
* Reproduced independently twice: `kla2/results/final.csv` gives 23.632 / 0.60792 / 0.19287;
  `docs/queue/testset_results.csv` gives 23.632 / 0.60791 / 0.19288.
* Cross-machine: identical to five decimals on the RTX 4090 pod and the RTX 4050 laptop.
* Output contract verified on the 297 outputs: shape `(256,256)`, range `[0.0000, 1.0000]`.
* `tools/stress_run.py`: **13 / 13 cases PASS** (square_128, single_image, size_64, size_256,
  size_512, non_square, odd_dims, trailing_axis, float64, out_of_range, nan_inf, mixed_sizes,
  nested_dir).

No other evaluation metric is recorded for this checkpoint in the shipping implementation.

---

## 9. Computation Graph Characteristics

| characteristic | value |
|---|---|
| Convolution operations | **36** Conv2d, all 3×3 / stride 1 / pad 1 |
| Linear layers | 0 |
| Attention operations | **0** |
| Normalization operations | **0** |
| Downsampling operations | **0** |
| Upsampling operations | **2** — one `PixelShuffle(2)`, one `F.interpolate(mode='bicubic')` |
| Concatenations | **1** — stem, `cat([x, sqrt, log1p], dim=1)`, 1→3 channels |
| Additions | **18** — 16 block residuals + 1 long skip + 1 global bicubic skip |
| Reshapes | **1** — implicit inside `PixelShuffle` |
| Transposes / permutes | **0** in the graph (`channels_last` is a memory format, not an op) |
| Elementwise activations | 16 ReLU |
| Other elementwise ops | 16 `mul` by `res_scale`; stem `clamp_min`, `sqrt`, `add`, `log1p`; output `nan_to_num`, `clamp` |
| Dtype casts | `.float()` on the residual and on the bicubic base (forced fp32 outside autocast) |

### Cost distribution (batch 1, 128×128)

| block | MMAC | % of total |
|---|---:|---:|
| `head` Conv 3→64 | 28.3 | 0.1% |
| **`body` — 16 × ResBlock (32 convs)** | **19,327.4** | **86.2%** |
| `body_tail` Conv 64→64 | 604.0 | 2.7% |
| **`up.0` Conv 64→256** | **2,415.9** | **10.8%** |
| `tail` Conv 64→1 | 37.7 | 0.2% |
| **Total** | **22,413.3** | 100% |

Each of the 33 convolutions operating at 64→64 / 128×128 costs exactly **604.0 MMAC** (2.7%).

### Major tensor dimensions (batch 1)

| tensor | shape | elements |
|---|---|---:|
| input | (1, 1, 128, 128) | 16,384 |
| stem output | (1, 3, 128, 128) | 49,152 |
| trunk feature (33 convs operate here) | (1, 64, 128, 128) | 1,048,576 |
| **`up.0` output — largest activation** | **(1, 256, 128, 128)** | **4,194,304** |
| post-PixelShuffle | (1, 64, 256, 256) | 4,194,304 |
| bicubic base | (1, 1, 256, 256) | 65,536 |
| output | (1, 1, 256, 256) | 65,536 |

Operations with large feature maps: the entire 16-block trunk holds a 64 × 128 × 128 tensor;
`up.0` produces the single largest tensor at 256 × 128 × 128; `tail` is the only convolution
that runs at the 256 × 256 output resolution.

---

## 10. Final Shipping Snapshot

| Category | Final Shipping Model |
|---|---|
| Architecture | EDSR-style flat residual CNN — VS stem → Conv3×3 head → 16 × ResBlock (ReLU, res_scale 0.1) → body_tail → PixelShuffle ×2 → Conv3×3 tail → + bicubic skip. No norm, no attention, no downsampling |
| Parameters | 1,368,705 (all trainable) |
| Input | (B, 1, 128, 128) float32, [0,1], 1 channel |
| Output | (B, 1, 256, 256) float32, [0,1], 1 channel |
| Framework | PyTorch 2.13.0+cu126, Python 3.13 |
| Precision | Weights **FP32**; inference under `autocast` **fp16** on CUDA; global skip forced fp32 |
| GPU | Stated target **H100** (no measurement). All measurements on **RTX 4050 Laptop, 6 GB** |
| CUDA | 12.6 |
| cuDNN | **UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION** |
| Checkpoint | `.pt` ZIP, 5,493,743 B, keys `{state_dict, config}`, inference-only, sha1 `e208d13d62b3…` |
| Batch size | 32 (CUDA default) / 1 (CPU); OOM-halving down to 1 |
| GFLOPs | 44.827 per image (batch 1, 128×128) |
| Model latency | batch 1: **14.56 ms** *and* **2.168 ms** both recorded (unreconciled); batch 32: 6.4642 ms/img; batch 64: 0.634 ms/img |
| End-to-end latency | **19.06 ms/image** (median 5.66 s for 297 images, n=9) |
| Throughput | End-to-end **52.5 img/s**; model-only 154.7 img/s @ b32, 1,577 img/s @ b64 |
| GPU memory | 0.123 GB @ b1 · 3.533 GB @ b32 · 3.45 GB @ b64 |
| PSNR | **23.632 dB** |
| SSIM | **0.60791** |
| LPIPS | **0.19288** |

---

## Items explicitly unavailable

The following are **UNKNOWN — NOT PRESENT IN PROVIDED SHIPPING IMPLEMENTATION**:
cuDNN version · NVIDIA driver version · deployment-host CPU · deployment-host RAM ·
GPU utilization % · H100 latency/throughput/VRAM · isolated preprocessing time ·
isolated model-only time inside `run.py` · isolated postprocessing time ·
`models/knob_datasheet.json` (not generated).
