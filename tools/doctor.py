#!/usr/bin/env python3
"""Laptop Ready check. Run this first on any machine, and after anything breaks.

    python tools/doctor.py

Prints one line per check and exits 0 only if every REQUIRED check passed. The
last check actually runs run.py on a synthetic image end to end -- imports
succeeding is not the same as the pipeline working.
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GOOD_SHA = "f95377a21e86b02ad3add4f68c784b439ae36e53"
KNOWN = {
    GOOD_SHA: "e1f_gate-120 (the shipped model)",
    "61587e96f554f32d7538651664b58e4dfe02f2ac":
        "e1f_gate-120 TRAINING checkpoint -- right weights, wrong container; "
        "deploy it with swap.py, which strips it to {state_dict, config}",
    "e208d13d62b3ded7b19954c273e80355e718a5f4":
        "pr50-w50-lp05-120  SUPERSEDED (the Round 2 model)",
    "92f45544f57d9f5f949cda80e9a76c3066af4647": "r2-preal1  *** WRONG MODEL ***",
    "8dc5b0a9ec00fc17e110442fa47136ca38db3f32": "loss-lp05-120  *** WRONG MODEL ***",
    "a27b4d964922cce66fabeea874f901b9cba1ef0a": "v1 round 1  *** WRONG MODEL ***",
}

W, results = 34, []


def check(name, required=True):
    def deco(fn):
        results.append((name, fn, required))
        return fn
    return deco


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------------- checks
@check("python version", required=False)
def _python():
    v = "%d.%d.%d" % sys.version_info[:3]
    want = os.path.join(ROOT, ".python-version")
    if os.path.isfile(want):
        with open(want) as f:
            w = f.read().strip()
        if not v.startswith(w):
            return False, "%s  (repo asks for %s)" % (v, w)
    return True, v


@check("virtualenv active", required=False)
def _venv():
    inside = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return inside, (os.path.basename(sys.prefix) if inside
                    else "no -- using the system python")


@check("numpy")
def _numpy():
    import numpy
    return True, numpy.__version__


@check("torch")
def _torch():
    import torch
    return True, torch.__version__


@check("GPU")
def _gpu():
    import torch
    if not torch.cuda.is_available():
        return True, "none -- CPU only (works, just slower)"
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    return True, "%s, %.1f GB, sm_%d%d" % (
        p.name, p.total_memory / 1e9, p.major, p.minor)


@check("model weights")
def _weights():
    p = os.path.join(ROOT, "models", "model.pt")
    if not os.path.isfile(p):
        return False, "MISSING at models/model.pt"
    h = sha1(p)
    known = KNOWN.get(h)
    if h == GOOD_SHA:
        return True, "correct  (%s)" % h[:12]
    if known:
        return False, "%s  %s" % (h[:12], known)
    return False, "%s  UNRECOGNISED -- see docs/CHECKPOINTS.md" % h[:12]


@check("repo files")
def _files():
    need = ["run.py", "src/model.py", "src/train.py", "src/validate.py",
            "src/dataset.py", "src/degrade.py", "src/metrics.py",
            "tools/verify_shipped.py", "tools/package_check.py"]
    missing = [f for f in need if not os.path.isfile(os.path.join(ROOT, *f.split("/")))]
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, "%d/%d present" % (len(need), len(need))


@check("dataset", required=False)
def _data():
    for cand in ("data/train", "../semicon_train_data", "../semicon_test_data"):
        p = os.path.join(ROOT, *cand.split("/"))
        if os.path.isdir(p):
            n = sum(len(fs) for _, _, fs in os.walk(p))
            return True, "%s  (%d files)" % (cand, n)
    return False, "not found -- fine for inference, see docs/DATA.md"


@check("disk space", required=False)
def _disk():
    free = shutil.disk_usage(ROOT).free / 1e9
    return free > 5, "%.1f GB free" % free


@check("git commit", required=False)
def _git():
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode:
            return False, "not a git checkout"
        d = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        return True, r.stdout.strip() + (" (uncommitted changes)" if d.stdout.strip() else " (clean)")
    except Exception:
        return False, "git not on PATH"


@check("END TO END: run.py")
def _e2e():
    import numpy as np
    tmp = tempfile.mkdtemp(prefix="doctor_")
    try:
        din, dout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(din)
        rng = np.random.default_rng(0)
        x = rng.random((128, 128), dtype=np.float32) * 0.6 + 0.2
        np.save(os.path.join(din, "doctor_probe.npy"), x)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), din, dout],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            lines = [l.strip() for l in (r.stderr + "\n" + r.stdout).splitlines() if l.strip()]
            msg = next((l for l in reversed(lines) if "Error" in l or "error" in l),
                       lines[-1] if lines else "no output")
            return False, "run.py exited %d: %s" % (r.returncode, msg[:70])
        outs = [f for f in os.listdir(dout)] if os.path.isdir(dout) else []
        if not outs:
            return False, "run.py wrote nothing"
        y = np.load(os.path.join(dout, outs[0]))
        if y.shape != (256, 256):
            return False, "output shape %s, expected (256, 256)" % (y.shape,)
        if y.dtype != np.float32:
            return False, "output dtype %s, expected float32" % y.dtype
        if not np.isfinite(y).all():
            return False, "output contains NaN or inf"
        return True, "128x128 -> %s %s, range %.3f-%.3f" % (
            y.shape, y.dtype, float(y.min()), float(y.max()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------- main
def main():
    print()
    print("  ForgeX laptop check")
    print("  %s | %s" % (platform.node(), platform.platform()))
    print("  " + "-" * (W + 46))
    failed = []
    for name, fn, required in results:
        try:
            ok, detail = fn()
        except ImportError as e:
            ok, detail = False, "not installed (%s)" % (e.name or e)
        except Exception as e:
            ok, detail = False, "%s: %s" % (type(e).__name__, str(e)[:60])
        tag = "  ok  " if ok else (" FAIL " if required else " warn ")
        print("  [%s] %-*s %s" % (tag, W, name, detail))
        if not ok and required:
            failed.append(name)
    print("  " + "-" * (W + 46))
    if failed:
        print("  NOT READY -- %d required check(s) failed: %s"
              % (len(failed), ", ".join(failed)))
        print("  See docs/LAPTOP.md for the fix for each one.")
        print()
        return 1
    print("  LAPTOP READY -- this machine can run the demo.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
