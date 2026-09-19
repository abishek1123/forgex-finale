# Project instructions

> **Historical / superseded — not the shipped model.** This document describes `pr50-w50-lp05-120` (1,368,705 params; 23.632 PSNR / 0.60791 SSIM / 0.19288 LPIPS), the model shipped at **Round 2**.
>
> The Grand Finale ships **`e1f_gate-120`** — 1,346,360 params, **23.8297 PSNR / 0.62104 SSIM / 0.18126 LPIPS**, `models/model.pt` sha1 `f95377a21e86…`. Numbers below are kept as the record of how that decision was reached, not as a description of what ships.
>
> Current: [`docs/SHIP_E1F_GATE.md`](docs/SHIP_E1F_GATE.md)


Team ForgeX — KLA PS01, "AI-Based Restoration of Degraded Images for
Semiconductor Inspection", SEMICON India 2026 Grand Finale.
Abishek SR · Anmol BA · Hardik — VIT Vellore.

Blind restoration of degraded SEM images: a 128x128 noisy float32 `.npy` in
[0,1] goes in, a 256x256 float32 `.npy` comes out. Joint denoising and 2x
super-resolution, with the degradation parameters unknown at test time.

## Read these before doing anything

1. **`docs/HANDOFF.md`** — the source of truth for everything established before
   the finale. Problem, degradation physics, architecture, every verified number
   with the file it came from, the traps, the decisions and why.
2. **`HACKATHON_CONTEXT.md`** — live state during the finale: the new dataset,
   what is running now, the current best result, decisions taken today. Read it
   second; it overrides HANDOFF where the two disagree, because it is newer.

If neither has been read this session, read them before answering a question
about this project. Do not reconstruct context from the repository by inference
when a document states it.

## Hard rules

**Every number must trace to a file.** When stating a metric, name the CSV it
came from (`docs/results.csv`, `docs/queue/testset_results.csv`,
`docs/timesweep.csv`, `docs/capacity_sweep.csv`, `docs/EXPERIMENTS.csv`). A
figure you cannot source does not go in a document, a slide, or an answer.

**28.43 dB is a poisoned number.** It appears in `docs/results.csv` as `v1`
epoch 76. It is a **round 1** score on a **different test set**; the identical
checkpoint scores 23.1721 on the current one. The `3.74 M` parameter count that
travels with it belongs to a capacity ablation, not to any shipped model. If
anyone quotes either figure, stop them. Current shipped numbers: **23.632 dB
PSNR / 0.60791 SSIM / 0.19288 LPIPS / 2.168 ms per image / 1.369 M parameters.**

**Checkpoint identity is sha1, never filename or size.** Four files in this
project are exactly 5,493,743 bytes and three of them are the wrong model. The
correct one hashes to `e208d13d62b3ded7b19954c273e80355e718a5f4`. Verify with
`tools/verify_shipped.py`.

**`models/model.pt` changes only through `swap.py`.** Not `cp`, not an editor,
not a file manager. `swap.py` hash-checks, backs up, re-scores and restores
itself on a mismatch.

**Never push to `main`.** `main` is protected and only Abishek may merge. Work on
`exp/<name>-<topic>`, push the branch, open a pull request. Never `git push
--force`, never `git push --all`, never rewrite published history.

**Never run `git` commands on Abishek's machine without being asked.** Propose
the command and let him run it. A stale `.git/index.lock` has already cost this
project time.

**No vendor or assistant attribution anywhere in this repository.** No
`Co-Authored-By` trailers, no tool names in commit messages, code comments,
documentation or slides. Abishek's name only.

**Private keys never appear in chat.** `.pub` files only; keys move by USB.

## Environment

Python 3.13. Always use the virtualenv interpreter, never bare `python`:

```
.\.venv\Scripts\python.exe <script>        # Windows
./.venv/bin/python <script>                # Linux / pod
```

Check any machine with `python tools/doctor.py` — eleven checks ending in
`LAPTOP READY`, including an end-to-end run of `run.py` on a synthetic image.
Set a machine up from scratch with `setup.ps1` or `setup.sh`.

## Commands

```
python run.py <input_dir> <output_dir>          # inference; --tta --device cpu --weights
python tools/doctor.py                          # is this laptop able to demo
python tools/verify_shipped.py                  # are the shipped weights correct
python tools/package_check.py                   # full packaging verification
python src/validate.py --data <d> --ckpt <c> --tag <name>   # score, appends a traceable row
python src/train.py --data <d> --out runs/<name> --epochs 120 \
    --p-real 0.5 --wide-p 0.5 --seed 0 --split-seed 0 --amp
```

`run.py` is self-contained — it inlines the network and imports only torch and
numpy. `models/model.pt` is committed, so a clone can run inference immediately.

## Map

```
run.py                       inference entry point
models/model.pt              shipped weights (in git)
src/                         model, train, dataset, degrade, losses, metrics, validate
tools/                       doctor, verify_shipped, package_check, bench, scorecard
docs/HANDOFF.md              the project, fully
docs/LAPTOP.md               environment setup and what each failure means
docs/CHECKPOINTS.md          the four-identical-sizes problem
docs/DATA.md                 dataset layout and transfer
docs/REPRODUCIBILITY.md      the determinism decision and why
docs/ENGINEERING_LOG.md      42 KB chronological record
CONTRIBUTING.md              the five git rules
HACKATHON_CONTEXT.md         live state during the finale
docs/EXPERIMENTS.csv         one row per experiment
```

Never commit: `data/`, `runs/`, `.npy`, `.pt` (except `models/model.pt`),
`.venv/`, or anything containing a credential.

## Working during the finale

The finale gives us a new, unseen dataset and roughly two days. The existing
solution is the **baseline, not the answer** — but it is also the fallback, so it
stays working at all times.

Measure before changing. The two largest measured effects on this project are the
degradation distribution used in training (3.72 dB at high noise) and, far
behind it, model capacity (0.39 dB across a 78x parameter span). Proposals that
change the training distribution or the data understanding outrank proposals that
change the architecture, unless evidence from the new data says otherwise.

Anything learned in a session that matters — a finding about the new data, a
result, a decision — must be written into `HACKATHON_CONTEXT.md` or
`docs/EXPERIMENTS.csv` and committed. A finding that exists only in a chat window
is lost the moment that window closes, and the other two people never hear it.
