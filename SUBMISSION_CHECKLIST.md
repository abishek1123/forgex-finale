# ForgeX — KLA PS01 Round 2 submission checklist

Checked against the organisers' four required components. Every box was tested
by running it, not by reading the code.

## The four required components

### 1. Evaluation Script — standalone Python, non-notebook

**`run.py`** — `python run.py <input-dir> <output-dir>`

| requirement | status | how it was verified |
|---|---|---|
| accepts a test-images dir and an output dir | PASS | positional args, in that order |
| loads the trained model | PASS | `models/model.pt`, resolved relative to `run.py` |
| runs inference on all input images | PASS | reads every `.npy` in the input dir; exits non-zero if the output count differs |
| writes outputs to the specified directory | PASS | created automatically if missing; one output per input, same filename |
| runs without manual edits, used as-is | PASS | no paths, sizes or device hardcoded; `--device` auto-detects CUDA and falls back to CPU |

`run.py` is **fully self-contained**: the network definition is inlined, so it
imports only `torch` and `numpy` and depends on no other file in this repo
except the weights. It downloads nothing at runtime.

### 2. Training Script — reproduces the submitted model

**`train_submitted.py`** — `python train_submitted.py --data <dataset>`

No flags to choose, no README archaeology. It prints the exact command it runs
and then runs it. The shipped configuration is `pr50-w50-lp05-120`:

```
--epochs 120 --iters 500 --batch 32 --crop 64 --ch 64 --nb 16
--p-real 0.5 --wide-p 0.5 --w-lpips 0.05 --seed 0 --split-seed 0 --amp
```

`--dry-run` prints the command without training. Runtime ~28 min on an
RTX 4090, ~4.3 h on an RTX 4050 Laptop. Peak VRAM 0.95 GB.

### 3. Denoised Test Outputs

**`outputs/`** — our restoration of every image in the 297-image test set the
organisers supplied, produced by `run.py` with the submitted weights.
Regenerate with:

```
python tools/package_check.py --data <organisers-test-set>
```

That also verifies dtype, finiteness, range and shape on every output.

### 4. Environment Specification — complete pip freeze

**`requirements.txt`** — the complete `pip freeze` output of the environment
that trained `models/model.pt`, exactly as the submission specifies. One line is
*added* — `--extra-index-url https://download.pytorch.org/whl/cu126` — because
`torch==2.13.0+cu126` does not exist on PyPI and the file would not install
without it. No package line is altered or removed.

**`requirements-inference.txt`** — a convenience, not the spec: the two packages
`run.py` actually imports. Provided because a reviewer installing 27 packages to
run a script that needs two is a reasonable thing to want to avoid.

---

## Output contract

| guarantee | how |
|---|---|
| shape `(2H, 2W)`, or `(2H, 2W, 1)` if the input carried a channel axis | verified per file by `tools/package_check.py` |
| dtype float32 | explicit cast |
| finite, no NaN or Inf | `torch.nan_to_num` |
| values inside `[0, 1]` | `.clamp(0, 1)` |
| one output per input, identical filename | set-equality check; non-zero exit on mismatch |

Nothing hardcodes an input resolution: 128→256 and 256→512 both work, and mixed
sizes in one directory are grouped and batched correctly.

## Hardening beyond the checklist

* Runs correctly when invoked by **absolute path from an unrelated working
  directory** — `package_check.py` tests exactly that, in a scratch directory
  containing only the four required items.
* **Cross-machine reproducibility checked, not assumed**: the same configuration
  scores 22.7468 on an RTX 4090 and 22.74 on an RTX 4050.
* Three inference optimisations were written, benchmarked and **rejected on the
  measurement**: 8× test-time augmentation (+0.018 dB for 7.1× the time),
  pinned staging buffers with CUDA streams (+0.04 s on a 1.94 s run-to-run
  spread), and a `channels_last` layout fix (−0.06 s on a 0.73 s spread). All
  three produced bit-identical output, so none shipped. The timing runs are in
  `docs/bench_results.txt`.

## Results (organisers' 297-image test set)

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Bicubic ×2 | 20.455 | 0.5099 | 0.4655 |
| **ForgeX — 1.37 M params** | **23.632** | **0.6079** | **0.1929** |
| gain | **+3.18 dB** | +0.098 | **59% lower** |
