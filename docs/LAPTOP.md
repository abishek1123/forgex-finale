# Laptop Ready

Getting any machine from nothing to "can run the demo". Four commands, then a
check that either says READY or names exactly what is wrong.

## The whole thing

**Windows**

```powershell
git clone https://github.com/abishek1123/forgex-kla-ps01.git
cd forgex-kla-ps01
.\setup.ps1
```

**Linux / macOS / a RunPod pod**

```bash
git clone https://github.com/abishek1123/forgex-kla-ps01.git
cd forgex-kla-ps01
bash setup.sh
```

`setup` creates `.venv`, picks the GPU or CPU requirements file for you, installs,
creates the output directories, and then runs `tools/doctor.py`. It is safe to
re-run at any time — it reuses an existing `.venv` rather than rebuilding it.

If PowerShell refuses to run the script (`running scripts is disabled on this
system`):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**DONE when** the last line reads `LAPTOP READY -- this machine can run the demo.`

## After setup, always use the venv python

```powershell
.\.venv\Scripts\python.exe run.py <input_dir> <output_dir>
```

Not bare `python`. The system python has no torch, and the error it gives looks
like a broken repository rather than a wrong interpreter.

## The check, any time

```powershell
.\.venv\Scripts\python.exe tools\doctor.py
```

Run it when you sit down, after any install, after any merge, and before the
demo. It takes about fifteen seconds and the last of its checks actually runs
`run.py` on a synthetic 128×128 image and verifies a finite float32 256×256
array comes back. Imports succeeding is not the same as the pipeline working.

## What each failure means

| Check | FAIL means | Fix |
|---|---|---|
| `python version` | *(warning only)* your Python is not 3.13 | Fine for the demo. Results may differ in the last decimal; don't use this machine for numbers that go on a slide. |
| `virtualenv active` | *(warning only)* you're on the system python | Use `.venv\Scripts\python.exe`. |
| `numpy` / `torch` | not installed in this interpreter | Re-run setup. If it fails again, you're running the wrong python. |
| `GPU` | never fails — reports CPU-only | CPU works. Inference is slower; training is not realistic. |
| `model weights` | missing, or the **wrong model** | The hash is checked against four known checkpoints. If it names a wrong one, `git checkout main -- models/model.pt`. See `docs/CHECKPOINTS.md`. |
| `repo files` | an incomplete clone | `git status`, then `git checkout main -- .` |
| `dataset` | *(warning only)* no data found | Inference and the demo don't need it. Training does — see `docs/DATA.md`. |
| `git commit` | *(warning only)* not a checkout, or dirty tree | `(uncommitted changes)` before a demo is worth looking at. |
| `END TO END` | the pipeline does not work on this machine | This is the one that matters. Read the error it prints; if it mentions CUDA or memory, try `--device cpu`. |

## Three commands that matter

```powershell
# 1. restore a directory of degraded .npy files
.\.venv\Scripts\python.exe run.py <input_dir> <output_dir>

# 2. confirm the shipped weights are the right ones
.\.venv\Scripts\python.exe tools\verify_shipped.py

# 3. score a checkpoint, appending a traceable row to results.csv
.\.venv\Scripts\python.exe src\validate.py --data data\train --ckpt models\model.pt --tag <name>
```

`run.py` flags worth knowing: `--device cpu` (no GPU or a CUDA problem),
`--tta` (8× self-ensemble, slower, slightly better), `--batch N` (override the
automatic batch size), `--weights <path>` (score a different checkpoint without
swapping the shipped one).

Training:

```powershell
.\.venv\Scripts\python.exe src\train.py --data data\train --out runs\<name> `
  --epochs 120 --p-real 0.5 --wide-p 0.5 --seed 0 --split-seed 0 --amp
```

## What lives where

| Path | What |
|---|---|
| `run.py` | inference entry point — self-contained, needs only torch and numpy |
| `models/model.pt` | the shipped weights, **in the repo** — a clone has everything it needs |
| `src/` | model, training, dataset, degradation, metrics, validation |
| `tools/doctor.py` | this check |
| `tools/verify_shipped.py` | hash gate on the shipped weights |
| `tools/package_check.py` | full packaging verification |
| `outputs/`, `runs/` | created by setup; `runs/` is gitignored |
| `data/`, `*.npy`, `*.pt` | **never committed** (except `models/model.pt`) |

## Why there is no config file

Every path is already a command-line argument — `run.py` takes the input and
output directories, `train.py` and `validate.py` take `--data` and `--ckpt`. A
config layer on top would be one more thing to keep in sync when tomorrow's
dataset arrives in an unexpected shape. When it does, point the flags at it.

## If a laptop dies

Any other machine that passes the doctor can take over immediately: the clone
carries the code and the weights, and nothing in the demo path needs the network.
That is the entire reason all three machines are set up in advance rather than
one.

## Laptop Ready checklist

Tick these per machine, tonight.

- [ ] Repo cloned, `git log --oneline -1` matches the other laptops
- [ ] `setup` run to completion
- [ ] `tools/doctor.py` prints **LAPTOP READY**
- [ ] `model weights` line says **correct**
- [ ] `END TO END` line passes
- [ ] `run.py` run once by hand on a real test image, output opened and looked at
- [ ] The person who owns the laptop ran all of the above themselves
- [ ] Wi-Fi switched off, `doctor.py` run again, still READY
