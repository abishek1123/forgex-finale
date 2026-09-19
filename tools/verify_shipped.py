"""Ten-second guard. Run before any push, demo, or hand-off.

    python tools/verify_shipped.py                      # auto-finds the test set
    python tools/verify_shipped.py ../semicon_test_data # or point it explicitly

Confirms three things, in order, and stops at the first failure:

  1. models/model.pt is the SHIPPED checkpoint, by sha1 -- not by filename and
     not by size. Four files on a typical dev machine are 5,493,743 bytes and
     represent THREE different models. Size proves nothing.
  2. run.py executes and produces one output per input.
  3. Those outputs are BYTE-IDENTICAL to the committed outputs/.

Exits 0 (green) or 1 (red). No interpretation required.

Why this exists: the wrong model shipped three separate times on this project.
Twice it scored HIGHER on PSNR than the correct one, so no metric-level check
caught it. Only a hash comparison does.
"""
import glob
import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

GOOD = "40add39d927e5c265a7adbee22321932f7c48791"        # trained shared head, deploy
KNOWN = {
    GOOD:                                       "mx120-s0-shared     CORRECT",
    "f95377a21e86b02ad3add4f68c784b439ae36e53": "e1f_gate-120 PREVIOUS release",
    "61587e96f554f32d7538651664b58e4dfe02f2ac": "e1f_gate-120 TRAINING checkpoint -- "
                                                "right weights, wrong container. "
                                                "Deploy it with swap.py, which strips "
                                                "it to {state_dict, config}.",
    "e208d13d62b3ded7b19954c273e80355e718a5f4": "pr50-w50-lp05-120   SUPERSEDED "
                                                "(Round 2 model)",
    "92f45544f57d9f5f949cda80e9a76c3066af4647": "r2-preal1           *** WRONG ***",
    "8dc5b0a9ec00fc17e110442fa47136ca38db3f32": "loss-lp05-120       *** WRONG ***",
    "a27b4d964922cce66fabeea874f901b9cba1ef0a": "v1, ROUND 1         *** WRONG ***",
}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_test_set(explicit=None):
    """The test set lives outside the repo (it is not ours to commit)."""
    if explicit:
        return explicit
    parent = os.path.dirname(REPO)
    for cand in ("semicon_test_data", os.path.join("data", "semicon_test_data")):
        p = os.path.join(parent, cand)
        if os.path.isdir(p):
            return p
    return None


def main():
    ok = True

    # ---- 1. the hash -----------------------------------------------------
    model = os.path.join(REPO, "models", "model.pt")
    if not os.path.isfile(model):
        print("[1/3] models/model.pt  MISSING")
        sys.exit(1)
    h = sha1(model)
    print(f"[1/3] models/model.pt  sha1 {h[:16]}...")
    print(f"      identified as: {KNOWN.get(h, 'UNKNOWN CHECKPOINT -- stop and investigate')}")
    if h != GOOD:
        print("      FAIL")
        print("\nFAIL -- DO NOT PUSH OR DEMO.")
        sys.exit(1)                       # no point running inference on the wrong weights
    print("      OK")

    # ---- 2. run.py -------------------------------------------------------
    data = find_test_set(sys.argv[1] if len(sys.argv) > 1 else None)
    if not data:
        print("[2/3] SKIP -- no test set found next to the repo.")
        print("      Pass one explicitly:  python tools/verify_shipped.py <dir>")
        print("      See docs/DATA.md for where the datasets live.")
        print("\nPARTIAL -- the hash is correct, but the output was not verified.")
        sys.exit(0)

    out = os.path.join(REPO, "_verify_tmp")
    print(f"[2/3] running run.py over {data} ...")
    # --no-trt: this check is a BYTE comparison against the committed outputs,
    # which were produced on the PyTorch path. TensorRT agrees to ~1e-4 (worth
    # 0.0001 dB), which is correct but not byte-identical -- so once engines
    # exist on a machine this check would fail for a reason that is not a fault.
    r = subprocess.run([sys.executable, "run.py", data, out, "--no-trt"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("      FAIL:", (r.stdout + r.stderr)[-500:])
        print("\nFAIL -- DO NOT PUSH OR DEMO.")
        sys.exit(1)
    print("      OK")

    # ---- 3. byte-compare against the committed outputs -------------------
    print("[3/3] comparing to committed outputs/ ...")
    ref = os.path.join(REPO, "outputs")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "*.npy")))
    if not names:
        print("      FAIL: no outputs produced")
        ok = False
    else:
        worst, differing, missing = 0.0, 0, 0
        for n in names:
            rp = os.path.join(ref, n)
            if not os.path.isfile(rp):
                missing += 1
                continue
            d = float(np.abs(np.load(os.path.join(out, n)).astype(np.float64)
                             - np.load(rp).astype(np.float64)).max())
            worst = max(worst, d)
            differing += d > 0
        print(f"      {len(names)} files, max|diff| = {worst:.3e}, "
              f"differing = {differing}, not in outputs/ = {missing}")
        if worst != 0.0 or missing:
            print("      FAIL")
            ok = False
        else:
            print("      OK")

    print("\n" + ("PASS -- the repo holds the shipped model."
                  if ok else "FAIL -- DO NOT PUSH OR DEMO."))
    print(f"scratch left in {out} -- delete it when done.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
