#!/usr/bin/env python3
"""Reproduce the submitted model, exactly.

    python train_submitted.py --data <path-to-dataset>

<path> must contain GT/ and NoisyLR/ as supplied by the organisers:

    <data>/GT/000000.npy        256x256 float32 in [0,1]
    <data>/NoisyLR/000000.npy   128x128 float32, may fall outside [0,1]

This is a thin, deliberately unclever wrapper around src/train.py. It exists so
that reproducing models/model.pt requires no flags, no README archaeology and
no judgement about which of eleven configurations we actually shipped. It
prints the exact command it runs before running it.

Runtime: ~28 min on an RTX 4090, ~4.3 h on an RTX 4050 Laptop (6 GB, 75 W).
Peak VRAM 0.95 GB.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# THE SHIPPED CONFIGURATION.  Run name: pr50-w50-lp05-120
#
#   --p-real 0.5   50% of training samples are REAL KLA pairs
#   --wide-p 0.5   of the remaining 50% synthetic, half from the WIDE
#                  degradation family and half from the calibrated NARROW one,
#                  giving 50% real / 25% wide / 25% narrow overall
#   --w-lpips 0.05 perceptual term; see docs/ENGINEERING_LOG.md 12 for the
#                  eleven-model comparison that chose it
#   --seed / --split-seed are separate on purpose: the validation split must
#                  stay fixed while training randomness varies
# ---------------------------------------------------------------------------
CONFIG = [
    "--epochs", "120", "--iters", "500", "--batch", "32", "--crop", "64",
    "--ch", "64", "--nb", "16",
    "--p-real", "0.5", "--wide-p", "0.5", "--w-lpips", "0.05",
    "--seed", "0", "--split-seed", "0", "--amp",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir containing GT/ and NoisyLR/")
    ap.add_argument("--out", default="runs/submitted", help="where to write checkpoints")
    ap.add_argument("--resume", default="", help="resume from a checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    a = ap.parse_args()

    for sub in ("GT", "NoisyLR"):
        d = os.path.join(a.data, sub)
        if not os.path.isdir(d):
            sys.exit(f"ERROR: {d} not found. --data must contain GT/ and NoisyLR/.")
    ng = len([f for f in os.listdir(os.path.join(a.data, "GT")) if f.endswith(".npy")])
    nl = len([f for f in os.listdir(os.path.join(a.data, "NoisyLR")) if f.endswith(".npy")])
    if ng != nl:
        sys.exit(f"ERROR: GT has {ng} files, NoisyLR has {nl}. Refusing to train on mismatched pairs.")
    print(f"dataset: {ng} matched pairs")

    cmd = [sys.executable, os.path.join(HERE, "src", "train.py"),
           "--data", a.data, "--out", a.out] + CONFIG
    if a.resume:
        cmd += ["--resume", a.resume]

    print("\nshipped configuration — run name pr50-w50-lp05-120\n")
    print("  " + " ".join(cmd) + "\n")
    if a.dry_run:
        return

    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)

    ck = os.path.join(HERE, a.out, "last.pt")
    print(f"\ndone. checkpoint: {ck}")
    print("Score it against the shipped weights with:")
    print(f"  python src/validate.py --data {a.data} --ckpt {a.out}/last.pt --baseline")
    print("Then, to ship it:  copy it over models/model.pt")


if __name__ == "__main__":
    main()
