# Reproducibility

What this repo guarantees, what it does not, and why we chose it that way.
If a number in `docs/results.csv` ever looks wrong, start here.

## The short version

| Thing | Guarantee |
|---|---|
| Same machine, same seed, same code -> same trained model | yes |
| Same machine, same weights -> bit-identical `run.py` output | yes |
| Different GPU, same weights -> same score to 2 decimal places | yes (measured) |
| Different GPU, same weights -> bit-identical pixels | **no, and we do not want it** |
| Different GPU, same seed -> same trained model | no |

## What is pinned

- `.python-version` -> 3.13
- `requirements.txt` -> full freeze incl. `torch==2.13.0+cu126`, CUDA 12.6 index
- `requirements-cpu.txt` -> same pins without the CUDA suffix, for reviewers
- `requirements-inference.txt` -> the minimum to run `run.py`
- `--seed` seeds Python, NumPy and Torch; `--split-seed` seeds the train/val split
  separately, so changing the amount of validation data does not reshuffle training
- every row of `results.csv` carries `git_sha` and `weight_sha1` (see below)
- `models/model.pt` is hash-gated by `tools/verify_shipped.py`

## The decision: we did NOT enable full determinism

We do not set `torch.use_deterministic_algorithms(True)`, and we leave
`cudnn.benchmark` at its default. This was a choice, not an oversight.

**Why not.** Deterministic cuDNN kernels forbid the fast non-deterministic
reduction paths. On our hardware that costs roughly 15-25% of training throughput
and forces `CUBLAS_WORKSPACE_CONFIG` on every launcher, for a property we cannot
deliver anyway: cuDNN algorithm selection, TF32, and the atomics inside
`PixelShuffle`/backward reductions differ across GPU architectures. Determinism
would be true only *within* one machine -- which is what we already have.

**Why it doesn't matter here.** We measured it. The shipped checkpoint scored
**22.7468 dB on a RTX 4090** and **22.74 dB on the RTX 4050** it was trained on.
The gap is below the third decimal place, and roughly 100x smaller than the
0.284 dB that separates the best and worst checkpoints in
`docs/queue/testset_results.csv`. Nothing we conclude from this repo turns on a
difference that small.

**Where it *would* matter.** If two architectures in a comparison land within
~0.02 dB of each other, the ranking is noise, not a result. In that case rerun
both with 3 seeds and report mean +/- spread rather than reaching for
deterministic kernels.

### `cudnn.benchmark`

Measured in `docs/timesweep.csv`: enabling it costs **+2.63 s** of warm-up and
changes **288 of 297** output arrays. It buys nothing on our fixed 128x128 input
shape -- there is only one shape to tune for -- so it stays off. Inference is
already shape-stable.

## Provenance: how a number is tied to a model

`src/validate.py` writes two extra columns on every row:

- **`git_sha`** -- short commit, with `+dirty` appended if the working tree had
  uncommitted edits. `no-git` means it was run outside a checkout, and that row
  should be treated as untraceable.
- **`weight_sha1`** -- sha1 of the checkpoint file itself.

This exists because of a real failure on this project. **Four checkpoint files
here are exactly 5,493,743 bytes and three of them are the wrong model.** Size,
filename and folder all failed to distinguish them; a round-1 checkpoint (`v1`,
epoch 76, 28.43 dB on the *old* test set) was quoted as a current baseline twice.
The hash is the only identifier that cannot drift. See `docs/CHECKPOINTS.md`.

Rule: **a results row with an empty or `+dirty` `git_sha` is not a result.** It is
a note to yourself. Do not put it on a slide.

## Reproducing the shipped model

```
python src/train.py --data data/train --out runs/ship \
  --epochs 120 --p-real 0.5 --wide-p 0.5 --w-lpips 0.05 \
  --seed 0 --split-seed 0 --amp
```

Expect ~22.74 dB +/- a few hundredths on different hardware, not the same bytes.
Verify with:

```
python src/validate.py --data data/train --ckpt runs/ship/best.pt --tag repro
python tools/verify_shipped.py
```

`verify_shipped.py` checks the *shipped* file's hash against the known-good
`e208d13d...`; a retrain will not match it, and is not supposed to.

## What is not captured

- GPU driver and cuDNN version (torch pin fixes the API, not the kernels)
- the dataset itself -- see `docs/DATA.md`; it is not in git and never will be
- wall-clock timings, which are machine-specific by definition; `ms_per_img` in
  `results.csv` is comparable only between rows measured on the same GPU
