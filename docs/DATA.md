# Datasets — what they are and how to get them

None of this is in the repo: `data/` is gitignored and the sets total ~3 GB.
A fresh clone has code and weights but no data, and this file is how you fix that.

`bootstrap` checks for these paths and tells you which are missing.

## Expected layout

Every script takes explicit `--data` / `--test` paths, so nothing is truly
hardcoded — but all the defaults assume the datasets sit **beside** the repo:

```
kla/
├── forgex-kla-ps01/            ← this repo
├── semicon_train_data/
│   └── semicon_train_data/     ← note the nested folder; this is the real root
│       ├── GT/                 4,785 clean .npy
│       └── NoisyLR/            4,785 degraded .npy
├── semicon_test_data/
│   ├── GT/                     297 clean .npy
│   └── NoisyLR/                297 degraded .npy
├── data512/                    256→512 scale-generalisation set (OOD)
└── nffa/                       ten labelled NFFA morphologies (OOD)
```

## The sets

| set | files | size | what it is | needed for |
|---|---|---|---|---|
| `semicon_train_data` | 9,570 `.npy` | ~1.5 GB | KLA's real paired training data, 4,785 pairs | training, degradation fitting |
| `semicon_test_data` | 594 `.npy` | ~96 MB | the organisers' 297-image test set | **every reported number** |
| `data512` | — | — | 256→512, a scale the model never trained on | OOD: scale |
| `nffa` | — | — | ten labelled morphologies, public NFFA data | OOD: content |

The training folder on disk measures ~3.0 GB, of which ~1.5 GB is `.npy`.
The remainder is archives and intermediates that are **not** needed — copy only
`GT/` and `NoisyLR/` when moving it.

## Getting it onto a new machine

**By USB — preferred at the venue.** Copy the four folders. No network, no
transfer time, nothing to go wrong. This is what the go-bag is for.

**Over the network, machine to machine:**

```
# sender
tar -czf semicon.tar.gz semicon_train_data semicon_test_data
runpodctl send semicon.tar.gz          # prints a one-time code, then WAITS

# receiver — use the real code it printed, not a placeholder
runpodctl receive <code>
tar xzf semicon.tar.gz
```

Expect roughly 50 minutes at a 7 Mbps upload. That is the sender's ISP cap, not
a tool limit — no client will beat it.

**To a pod:** same as above. Check `df -h` first and make sure you are in
`/workspace` (the volume) and not `/` (the container disk, which is wiped).

## Verifying you got it all

```
python tools/check_data.py --data ../semicon_train_data/semicon_train_data
```

Or by hand — the counts must match exactly, and GT/NoisyLR must be the same
filenames:

```
GT/  and  NoisyLR/   →  4,785 files each   (train)
GT/  and  NoisyLR/   →    297 files each   (test)
```

A mismatch means a partial copy. Partial data trains silently and scores wrong.

## What is NOT here

There is **no unused real SEM data**. `semicon_sub700` is a subset of the
training set (19% overlap measured, ~21% expected). The `train/` folder is
**round-1 photographs**, not SEM — its content fingerprint is f90 0.287 against
0.735 for real SEM. Both were checked; neither adds anything.
