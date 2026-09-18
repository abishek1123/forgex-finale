# HACKATHON CONTEXT — live state

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> The Grand Finale ships **`e1f_gate-120`** — 1,346,360 params, **23.8297 PSNR / 0.62104 SSIM / 0.18126 LPIPS**, `models/model.pt` sha1 `f95377a21e86…`. Numbers below are kept as the record of how that decision was reached, not as a description of what ships.
>
> Current: [`docs/SHIP_E1F_GATE.md`](docs/SHIP_E1F_GATE.md)


**The present.** What is true right now, what is running, what we decided today.
For anything settled before the finale — the physics, the architecture, every
pre-finale number, the traps — read `docs/HANDOFF.md`. That file is the past and
changes rarely. This one is the present and changes hourly.

Where the two disagree, **this file wins**, because it is newer.

---

## STATUS — keep this block current, ignore the rest if you are in a hurry

```
UPDATED      : <YYYY-MM-DD HH:MM> by <name>
PHASE        : 0 inspect / 1 baseline / 2 diagnose / 3 experiment / 4 validate / 5 freeze / 6 present
BEST MODEL   : models/model.pt   sha1 e208d13d...   (pre-finale baseline)
BEST SCORE   : 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS   on the OLD test set
NEW DATA     : not yet received
RUNNING NOW  : nothing
BLOCKED ON   : metric weighting unknown
NEXT DECISION: <what we decide next, and roughly when>
```

---

## 1. Problem

Blind restoration of degraded SEM images for semiconductor inspection
(KLA PS01). 128×128 noisy float32 `.npy` in [0,1] → 256×256 float32 `.npy`.
Joint denoising and ×2 super-resolution, degradation parameters unknown at test
time.

**If the finale changes this — different sizes, different dtype, a different
task — write the change here and say so out loud:**

> CHANGE: _(none yet)_

## 2. Objective

Beat the organisers' baseline on their metric, on their data, with a model that
runs fast enough to be usable inline, and be able to explain every choice.

Secondary, and not optional: **the existing pipeline must keep working at all
times.** It is the fallback. Nothing gets broken in pursuit of something better.

## 3. Evaluation metrics

**KNOWN:** _(fill in the moment the organisers tell us)_

| metric | weight | source |
|---|---|---|
| PSNR | ? | |
| SSIM | ? | |
| LPIPS | ? | |
| inference time | ? | |
| other | ? | |

**ASSUMED UNTIL TOLD:** all three quality metrics matter, time is scored or at
least noticed.

**Why this is the single most important unknown:** our shipped model is 9th of 11
on PSNR and 2nd on LPIPS. It wins only if LPIPS carries roughly 35% or more of
the score. If PSNR dominates, `pr50-w50-120` is the better checkpoint. **Ask a
mentor this on day 1.** See `docs/HANDOFF.md` §7.1.

## 4. Current baseline

Whatever the organisers give us as a reference, plus bicubic interpolation.

| baseline | score on new data | measured by | when |
|---|---|---|---|
| bicubic | | | |
| organisers' | | | |
| **ours, unchanged** | | | |

**Rule: fill the third row before changing anything.** An improvement you cannot
measure against your own starting point is not an improvement.

## 5. Current model

1.37 M parameter EDSR-style residual CNN — parameter-free variance-stabilising
stem, 16 residual blocks, PixelShuffle ×2, zero-init tail over a global bicubic
skip. Full description and the reason for each piece: `docs/HANDOFF.md` §3.

## 6. Dataset information — NEW DATA

_Fill this in during Phase 0. Every row is evidence, not opinion._

| | old data | new data |
|---|---|---|
| images (train / val / test) | ~9,571 files | |
| input size | 128×128 | |
| output size | 256×256 | |
| dtype / range | float32 [0,1] | |
| file format | `.npy` | |
| paired? | yes | |
| scale factor | ×2 | |
| noise σ_add² | 0.00004 | |
| noise σ_mul² | 0.03649 | |
| `c` (within-block var) | 0.45944 | |
| speckle vs Poisson ΔBIC | 1118.3 → speckle | |
| blur present? | mild | |
| content / morphologies | 10 NFFA categories | |
| anything new entirely | — | |

**Verdict:** _(same distribution / shifted / different problem)_

## 7. Experiments

One row per experiment, in `docs/EXPERIMENTS.csv`. **Every experiment gets a row,
including the ones that fail — especially those.** A negative result you can cite
is worth more than one you have to re-run.

Live summary:

| ID | owner | what | status | outcome |
|---|---|---|---|---|
| E0 | | baseline, unchanged, on new data | | |
| A1-A8 | Abishek | 8 architectures at matched 1.37 M params, 40 ep, 1k subset | done 17 Sep | see `docs/arch_study_summary.txt` — transformers +0.026 SSIM at 2-4x cost; stem/skip null |

## 8. Results

Scores on the **new** data. Old-data numbers stay in `docs/HANDOFF.md` §4 and
must never be mixed into this table — that confusion has already cost this
project twice.

| model | PSNR | SSIM | LPIPS | ms/img | git sha | notes |
|---|---|---|---|---|---|---|
| bicubic | | | | | | |
| ours, unchanged (E0) | | | | | | |

## 9. Known weaknesses

Carried in from `docs/HANDOFF.md` §6.1 and §7:

- No architecture comparison was completed. The matched-parameter study against
  NAFNet / Restormer / SwinIR exists in `kla2/` but never ran past the smoke test.
- We have no evidence that long-range modelling doesn't help. SEM images of
  semiconductor structures are periodic, which is what attention exploits and a
  16-block CNN's ~33-pixel receptive field cannot see.
- We report PSNR / SSIM / LPIPS only. **A 2026 benchmark shows SR models winning
  on SSIM while detecting fewer defects than bicubic** (`docs/LITERATURE.md`,
  finding 2). One downstream metric would close this.
- Our synthetic noise has the right variance. We have not verified it has the
  right *spatial* structure.

**Found during the finale:**

- **The stem and skip ablation is a null result.** Matched-parameter study, 17 Sep,
  40 epochs on a 1,000-image subset, single seed: variance-stabilising stem worth
  **+0.005 dB**, bicubic skip worth **+0.020 dB**. All four corners of the 2x2
  factorial span 0.020 dB — a tenth of the epoch-to-epoch noise. Both were designed
  for training stability and early convergence, which by epoch 40 is invisible, and
  validation here is in-distribution so it cannot see a robustness benefit. **Do not
  claim a dB figure for either.** The defensible claim is narrower: parameter-free,
  and the network starts as exact bicubic interpolation.
- **Transformers beat us at matched parameters, on SSIM.** SwinIR +0.026 SSIM at
  3.9x cost, Restormer +0.026 SSIM at 2.0x cost. The SSIM gap is larger than the
  spread across our entire 11-checkpoint field, so it is real, not noise. PSNR gaps
  (+0.07 to +0.09 dB) are small. Predicted by `docs/LITERATURE.md`: periodic
  semiconductor structure is what attention exploits and a 16-block CNN cannot see.
- **NAFNet slightly dominates us at equal cost** — +0.027 dB, +0.014 SSIM, 9%
  faster. No trade-off to point at.
- Everything on that chart spans 0.12 dB. `--wide-p` is worth 3.72 dB. The
  architecture question is now measured and settled as second-order.

## 10. Git workflow

`CONTRIBUTING.md`. One line: branch as `exp/<name>-<topic>`, push the branch,
open a PR, Abishek merges. Nobody else touches `main`. Nobody touches
`models/model.pt` except through `swap.py`.

## 11. Environment setup

`docs/LAPTOP.md`. One line: `setup.ps1`, then `tools/doctor.py` must say
LAPTOP READY.

| machine | GPU | can train | can demo |
|---|---|---|---|
| Abishek — GODSEYE | RTX 4050, 6.4 GB | yes | yes |
| Anmol — LAPTOP-GEE5U19H | RTX 3050, 6.4 GB | yes | yes |
| Hardik — omsairam-04 | none, CPU | no | yes |

## 12. Current best model

```
FILE   : models/model.pt
SHA1   : e208d13d62b3ded7b19954c273e80355e718a5f4
NAME   : pr50-w50-lp05-120
SOURCE : pre-finale baseline
```

**Changing this is a deliberate act.** `swap.py` only, never a file copy, and
never on presentation day. When it changes, record the old sha1 here so the
change is reversible:

> PREVIOUS: _(none yet)_

## 13. Current best configuration

```
--epochs 120 --p-real 0.5 --wide-p 0.5 --w-lpips 0.05 --seed 0 --split-seed 0 --amp
```

Architecture: `Restorer(ch=64, nb=16, scale=2, res_scale=0.1)` — 1,368,705 params.

## 14. Ideas to test

Ordered by expected value per hour, not by how interesting they are.
**Do not start one before E0 is measured.**

| # | idea | why we think so | cost | who |
|---|---|---|---|---|
| 1 | re-fit the degradation law on the new data | the largest measured effect on this project is the training degradation distribution: 3.72 dB vs 0.39 dB from a 78× capacity change | 1–2 h | |
| 2 | re-sweep `--wide-p` on the new data | same reason; it is the flag robustness actually splits on | 3–4 h | |
| 3 | one downstream metric (defect recall or edge-position error) | closes the biggest hole in the evaluation story | 2 h | |
| ~~4~~ | ~~matched-parameter architecture study~~ | **DONE 17 Sep** — all eight ran | — | Abishek |
| 5 | bias-free CNN ablation | published mechanism for out-of-range noise collapse | 2 h | |
| 6 | retrain `pr50-w50-120` | hedge if the metric turns out PSNR-weighted | overnight | |

**Ideas rejected, and why** — write them here so nobody re-proposes them at 2am:

> _(none yet)_

## 15. Experiments currently running

| what | where | started | expected done | owner | check by |
|---|---|---|---|---|---|
| | | | | | |

**Clear this table when a run finishes.** A stale row here means somebody is
waiting on something that already finished.

## 16. Final selected solution

_Filled in at code freeze. Until then it is the pre-finale baseline._

```
MODEL    :
SHA1     :
CONFIG   :
SCORE    :
CHOSEN   : <date/time>  BECAUSE: <one sentence, evidence-based>
FROZEN AT: <git tag / commit>
```

## 17. Presentation storyline

Eleven beats. Owner in brackets — that person can deliver it cold.

1. **Problem** — degraded SEM imagery blocks inspection []
2. **Why it matters** — throughput and measurement precision at the fab []
3. **Challenge** — blind, and restoration can destroy the very evidence being inspected []
4. **Our approach** — one joint model, physics in the stem []
5. **Why this approach** — measured: capacity 0.39 dB vs degradation training 3.72 dB []
6. **Results** — quantitative, on their data []
7. **Generalisation** — eight axes, worst case still positive []
8. **Practicality** — ms/image, MB, runs on a laptop []
9. **Demo** — live before/after []
10. **Impact** — where this sits in an inspection line []
11. **Future work** — the three things in §9, named honestly []

Full talk track and the nine anticipated questions: the war-room board.

## 18. Demo procedure

```
PRIMARY    : live run.py on a judge-chosen image -> before/after + metrics + timing
BACKUP     : pre-rendered outputs on the USB, same images, same slides
EMERGENCY  : PDF only, narrated
```

Exact steps, in order, so anyone can run it:

1. `..\.venv\Scripts\python.exe tools\doctor.py` → LAPTOP READY
2. `..\.venv\Scripts\python.exe tools\verify_shipped.py` → correct hash
3. `..\.venv\Scripts\python.exe run.py <in> <out>`
4. Open the comparison view
5. Read out: PSNR, time per image, parameter count

**Never depends on the network.** Rehearsed on battery at least three times
before it is shown to anyone.

---

## How to update this file without wasting time

1. **The STATUS block is the only mandatory part.** Update it when the phase
   changes, when a run starts or finishes, and when the best model changes.
   Nothing else is required.
2. **Append, don't rewrite.** Add a row. Don't reorganise at 2am.
3. **Commit straight to `main`.** This is a document, not code — no PR needed.
   `git add HACKATHON_CONTEXT.md && git commit -m "ctx: <what changed>" && git push`
4. **One owner per section.** §6 and §8 belong to whoever runs the data and the
   scoring. §12, §13, §16 belong to Abishek. §17 and §18 belong to whoever owns
   the presentation. If it is your section, you update it — don't wait to be asked.
5. **If a finding is not in this file or in `docs/EXPERIMENTS.csv`, it does not
   exist.** A result that lives only in someone's terminal scrollback is lost the
   moment that window closes, and the other two never hear it.
6. **Regenerate the context bundle after a significant update** so any assistant
   gets the current state: `python tools/bundle_context.py`
