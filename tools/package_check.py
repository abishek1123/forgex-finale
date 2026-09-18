"""End-to-end packaging verification for the KLA PS01 submission.

Does three things, in order, and fails loudly on any of them:

  1. Restores a held-out sample with run.py and writes them to outputs/.
  2. Builds a before/after preview figure from the same images.
  3. Copies ONLY the four required items into a scratch directory, invokes
     run.py there by absolute path from an unrelated working directory, and
     checks the results are identical to step 1.

Usage:
    python tools/package_check.py --data <path-to-round2-data> [--n 24]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# src/dataset.py imports its siblings by bare name (`from degrade import ...`),
# so src/ itself must be on the path, not just the repo root.
sys.path.insert(0, os.path.join(ROOT, "src"))


def _wrong_interpreter(exc):
    """A missing dependency here is almost always the wrong python, not a bug.

    Without this you get a bare ModuleNotFoundError pointing at src/dataset.py,
    which reads as 'the repo is broken' when it means 'you are outside the venv'.
    """
    name = exc.name or "a dependency"
    if name in ("dataset", "degrade", "model", "metrics") and not os.path.isfile(
            os.path.join(ROOT, "src", name + ".py")):
        print("FAIL: src/%s.py is missing -- this checkout is incomplete." % name)
        sys.exit(1)
    print("FAIL: cannot import " + name +
          " -- this is an environment problem, not a code problem.\n")
    print("  interpreter : " + sys.executable)
    print("  python      : %d.%d.%d" % sys.version_info[:3])
    want = os.path.join(ROOT, ".python-version")
    if os.path.isfile(want):
        with open(want) as f:
            print("  repo wants  : " + f.read().strip() + "   (.python-version)")
    print("\nActivate the virtualenv, or call its python directly:")
    if os.name == "nt":
        print(r"    .venv\Scripts\python.exe tools\package_check.py")
    else:
        print("    .venv/bin/python tools/package_check.py")
    print("\nNo venv yet:")
    print("    python -m venv .venv")
    print("    pip install -r requirements.txt        # CUDA machine")
    print("    pip install -r requirements-cpu.txt    # everyone else")
    sys.exit(1)


try:
    import numpy as np

    from dataset import make_split, list_ids   # noqa: E402
except ModuleNotFoundError as _e:
    _wrong_interpreter(_e)


def die(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def run_script(script, indir, outdir, cwd=None):
    # --no-trt on every invocation. Step 3 compares the repo run against a run
    # from a directory holding ONLY the four required files -- which has no
    # models/engines/, so it would take the PyTorch path while the repo run took
    # TensorRT. The two agree to ~1e-4 (0.0001 dB), which is correct but not
    # byte-identical, so the comparison must pin both sides to one runtime.
    cmd = [sys.executable, script, indir, outdir, "--no-trt"]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        die("run.py exited %d" % r.returncode)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing GT/ and NoisyLR/")
    ap.add_argument("--n", type=int, default=0,
                    help="restore only the first N inputs (0 = all of them)")
    ap.add_argument("--from-split", action="store_true",
                    help="restore held-out training images instead of --data's own NoisyLR")
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    lr_dir = os.path.join(a.data, "NoisyLR")
    for d in (gt_dir, lr_dir):
        if not os.path.isdir(d):
            die("missing directory: " + d)

    if a.from_split:
        _, sample = make_split(gt_dir, n_val=200, seed=0)
        print("restoring HELD-OUT TRAINING images")
    else:
        # Requirement 3 of the submission: "Denoised Test Outputs -- model output
        # on the test set". So by default we restore every image in --data.
        sample = list_ids(gt_dir)
        print("restoring EVERY image in %s" % a.data)
    if a.n:
        sample = sample[:a.n]
    print("  %d images:  %s ... %s" % (len(sample), sample[0], sample[-1]))

    outdir = os.path.join(ROOT, "outputs")
    os.makedirs(outdir, exist_ok=True)
    stage = os.path.join(outdir, "_inputs")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    for f in os.listdir(outdir):          # start from a clean outputs/
        if f.endswith(".npy"):
            os.remove(os.path.join(outdir, f))
    for i in sample:
        shutil.copy2(os.path.join(lr_dir, i + ".npy"), os.path.join(stage, i + ".npy"))

    # ---------------------------------------------------------------- step 1
    print("\n[1/3] restoring with run.py from the repo root")
    run_script(os.path.join(ROOT, "run.py"), stage, outdir, cwd=ROOT)

    produced = sorted(f for f in os.listdir(outdir) if f.endswith(".npy"))
    if len(produced) != len(sample):
        die("expected %d outputs, found %d" % (len(sample), len(produced)))
    if {f[:-4] for f in produced} != set(sample):
        die("output filenames do not match input filenames")

    ref = {}
    for f in produced:
        x = np.load(os.path.join(outdir, f))
        if x.dtype != np.float32:
            die("%s is %s, expected float32" % (f, x.dtype))
        if not np.isfinite(x).all():
            die("%s contains non-finite values" % f)
        if x.min() < 0.0 or x.max() > 1.0:
            die("%s outside [0,1]: [%.4f, %.4f]" % (f, x.min(), x.max()))
        lr = np.load(os.path.join(stage, f))
        want = (lr.shape[0] * 2, lr.shape[1] * 2)
        if x.shape[:2] != want:
            die("%s shape %s, expected %s" % (f, x.shape, want))
        ref[f] = x
    print("      %d outputs: float32, finite, in [0,1], exactly 2x input" % len(produced))

    # ---------------------------------------------------------------- step 2
    print("\n[2/3] before / after preview")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pick = sample[:4]
        f, ax = plt.subplots(3, len(pick), figsize=(3.1 * len(pick), 9.2))
        for c, i in enumerate(pick):
            lr = np.load(os.path.join(lr_dir, i + ".npy")).squeeze()
            hr = np.load(os.path.join(gt_dir, i + ".npy")).squeeze()
            out = ref[i + ".npy"].squeeze()
            for r, (img, lab) in enumerate([(lr, "degraded input"),
                                            (out, "restored"),
                                            (hr, "ground truth")]):
                ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1,
                                interpolation="nearest")
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if c == 0:
                    ax[r, c].set_ylabel(lab, fontsize=11)
            ax[0, c].set_title(i, fontsize=10)
        f.tight_layout()
        p = os.path.join(outdir, "preview.png")
        f.savefig(p, dpi=130)
        print("      wrote " + p)
    except ImportError:
        print("      matplotlib not installed -- preview skipped (not a failure)")

    # ---------------------------------------------------------------- step 3
    print("\n[3/3] clean-directory run: only the four required items")
    scratch = tempfile.mkdtemp(prefix="kla_pkg_")
    try:
        os.makedirs(os.path.join(scratch, "models"))
        for rel in ("run.py", "requirements.txt", "README.md"):
            shutil.copy2(os.path.join(ROOT, rel), os.path.join(scratch, rel))
        shutil.copy2(os.path.join(ROOT, "models", "model.pt"),
                     os.path.join(scratch, "models", "model.pt"))

        listing = sorted(os.listdir(scratch))
        print("      scratch contains: %s" % listing)

        clean_out = os.path.join(scratch, "restored")
        # invoked by ABSOLUTE path, from a working directory that is not the
        # script's directory -- this is what caught a bug in round 1
        run_script(os.path.join(scratch, "run.py"), stage, clean_out,
                   cwd=tempfile.gettempdir())

        got = sorted(f for f in os.listdir(clean_out) if f.endswith(".npy"))
        if got != produced:
            die("clean run produced %d files, repo run produced %d"
                % (len(got), len(produced)))
        worst = 0.0
        for f in got:
            d = float(np.abs(np.load(os.path.join(clean_out, f)) - ref[f]).max())
            worst = max(worst, d)
        print("      max |clean - repo| = %.3e" % worst)
        if worst > 0.0:
            print("      NOTE: not bit-identical. On GPU with autocast a value")
            print("      up to ~2.4e-4 (the fp16 quantum) is expected.")
            if worst > 1e-3:
                die("clean-directory run differs by more than fp16 precision")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    shutil.rmtree(stage, ignore_errors=True)
    print("\nPASS: all three checks. outputs/ now holds %d restored images."
          % len(produced))


if __name__ == "__main__":
    main()
