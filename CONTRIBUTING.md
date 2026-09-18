# Contributing

Three people, one submission, one week. These rules exist so that nobody has to
ask "is main okay right now?" -- the answer is always yes.

## The five rules

**1. `main` always passes `package_check`.**

Before anything lands on `main`:

```
python tools/package_check.py
python tools/verify_shipped.py
```

Both must pass. If `main` is red, fixing it outranks whatever you were doing.
Do not push work on top of a red `main` -- you inherit the blame for it.

**2. Work on `exp/<name>-<topic>`.**

```
git switch -c exp/abishek-nafnet
```

One branch per idea. `exp/anmol-loss-sweep`, `exp/hardik-tta-timing`. Branch off
current `main`, not off each other's branches. A dead experiment is deleted, not
merged -- but write down what it showed in `docs/` first, because a negative
result you can cite is worth more than one you have to re-run.

**3. Nobody touches `models/model.pt` except through `swap.py`.**

Not `cp`, not drag-and-drop, not "just this once". `swap.py` verifies the hash,
backs up the previous file, re-scores, and restores automatically if the score
does not match. It also refuses `r2-preal1` without `--force`, which is the whole
point.

Four files in this project are exactly 5,493,743 bytes and three of them are the
wrong model. One of them scores *higher* PSNR than the one we ship and collapses
by 4 dB at high noise. This rule is what stands between us and shipping it.
See `docs/CHECKPOINTS.md`.

**4. No `--force`, no `push --all`, no rewriting published history.**

`git push --force` on a shared branch destroys a teammate's work silently. If you
think you need it, you need someone else's eyes instead. `git push --all` pushes
branches you did not mean to publish -- push branches by name.

To undo a bad commit on your own branch: `git revert`. History stays honest.

**5. Abishek merges to `main`.**

Anyone can open the merge; one person lands it. Not seniority -- it is so that one
person always knows what is in the submission. If Abishek is unreachable and it is
urgent, land it and say so in the group immediately, in writing.

## Committing

- Small commits with a real message. `fix bug` tells the person reading it in
  three days nothing. `validate: record git sha and weight hash per row` does.
- Never commit: `data/`, `runs/`, `.npy`, `.pt` (except `models/model.pt`),
  `.venv/`, previews, anything with a credential in it. `.gitignore` covers these;
  if you are reaching for `git add -f`, stop and ask.
- **Never commit a private key.** `.pub` files only. SSH keys move by USB.
- Run `git status` before every commit and actually read it.

## Before you ask someone to look at your branch

```
python tools/package_check.py                 # structure
python src/validate.py --ckpt <your ckpt> --tag <branch>   # a number
```

Bring the `results.csv` row. A claim without a row -- with its `git_sha` and
`weight_sha1` filled in -- is not reviewable. See `docs/REPRODUCIBILITY.md`.

## Environment

Python 3.13 (`.python-version`). `requirements.txt` for CUDA machines,
`requirements-cpu.txt` for everyone else. If you install something, freeze it into
the right file in the same commit -- a dependency that only exists on your laptop
is a broken build for the other two.

## If you break something

Say so in the group chat first, fix second. Everything here is recoverable from
git and from `checkpoints/`; the only unrecoverable thing is the two hours nobody
spends looking for a bug you already know about.
