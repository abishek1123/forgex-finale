# Checkpoint index

**Identity is the sha1. Never the filename, never the size.**

Three different wrong models have been deployed to `models/model.pt` during this
project. Two of them scored *higher* on PSNR than the correct one, so no
metric-level check caught either. Only a hash comparison does.

Verify at any time with:

```
python tools/verify_shipped.py
```

## The one that ships

| | |
|---|---|
| path | `models/model.pt` |
| sha1 | `e208d13d62b3ded7b19954c273e80355e718a5f4` |
| size | 5,493,743 bytes |
| run | `pr50-w50-lp05-120` |
| config | 120 epochs, ch 64, nb 16, `--p-real 0.5 --wide-p 0.5 --w-lpips 0.05`, seed 0, split-seed 0 |
| params | 1,368,705 |
| score | **23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS** on the organisers' 297-image test set |

## Deploy-size files — all 5,493,743 bytes, three different models

`dir` and `ls -l` cannot tell these apart. Only the hash can.

| sha1 (first 8) | verdict | where |
|---|---|---|
| `e208d13d` | **CORRECT** | `models/model.pt` |
| `e208d13d` | **CORRECT** | `../checkpoints/SHIPPED--pr50-w50-lp05-120--deploy.pt` |
| `92f45544` | **WRONG** — `r2-preal1` | `../checkpoints/WRONG--r2-preal1--do-not-ship.pt` |
| `8dc5b0a9` | **WRONG** — `loss-lp05-120` | `../kla-restore/models/WRONG--loss-lp05-120--do-not-ship.pt` |
| `a27b4d96` | **WRONG** — `v1`, round 1 | `../kla-restore/models/WRONG--v1-round1--do-not-ship.pt` (5,495,073 B) |

## Full checkpoints — 16,496,845 bytes, two different models

| sha1 (first 8) | model | where |
|---|---|---|
| `ad3a4473` | `pr50-w50-lp05-120` — **ours** | `../checkpoints/SHIPPED--pr50-w50-lp05-120--full.pt` |
| `ad3a4473` | same file | `../podscores/pr50-w50-lp05-120--full--SHIPPED.pt` |
| `ea9330e1` | `loss-lp05-120` — not ours | `../kla-restore/runs/loss-lp05-120--full--NOT-SHIPPED.pt` |

## Why `r2-preal1` must never ship

It is the highest-PSNR checkpoint we have: **23.8657** against the shipped
23.632, a gain of 0.234 dB. It is also the most fragile thing we have built.

| model | wide-p | gain over bicubic @ σ_mul 0.45 |
|---|---|---|
| `pr50-w50-120` | 0.5 | 6.03 dB |
| shipped | 0.5 | 5.91 dB |
| `pr50-w00-120` | 0.0 | 2.74 dB |
| **`r2-preal1`** | **0.0** | **1.88 dB** |

Four decibels behind under stronger noise. The field splits on `wide-p`, not on
`p_real`: every model trained without the wide degradation family collapses.
Picking `r2-preal1` because it tops the leaderboard is the exact mistake the
nine-level noise sweep exists to prevent.

## Not on this machine

Ten of the eleven 120-epoch queue checkpoints were trained on a pod that no
longer exists. Only their scores came back, in `docs/queue/testset_results.csv`.
In particular **`pr50-w50-120`** — the swap candidate if LPIPS turns out to carry
less than ~35% of the score — must be retrained:

```
python src/train.py --data <dataset> --epochs 120 --seed 0 --split-seed 0 --amp \
  --out runs/pr50-w50-120 --p-real 0.5 --wide-p 0.5
```

That is the shipped model's exact command **minus `--w-lpips 0.05`**.
