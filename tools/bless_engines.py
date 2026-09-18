#!/usr/bin/env python3
"""Re-key the engine index to the CURRENT models/model.pt -- after PROVING it matches.

    python tools/bless_engines.py [--verify-shipped ../verify_shipped.py]

Why this exists. A TensorRT engine is compiled machine code for one specific set
of weights, so run_fast.py refuses TensorRT unless models/model.pt matches the
checkpoint the engines were built from. But swap.py STRIPS a checkpoint on the
way in -- it keeps {state_dict, config} and drops args/epoch/best -- so the
shipped file legitimately has a different sha1 from the training checkpoint the
engines were exported from. Rewriting the recorded hash on trust would turn a
real safety guard into a rubber stamp.

Two proofs, in order of strength:

  EXACT (preferred). The engines record the sha1 of the training checkpoint.
  If that file is still on disk, load it and compare its state_dict to
  models/model.pt's TENSOR BY TENSOR, bytes included. No tolerance, no
  numerics, no argument.

  FUNCTIONAL (fallback). Re-run the input tensor saved at export time and
  compare with the output saved beside it. This is only a numerical match: the
  reference was produced on the BUILD gpu, where cuDNN uses TF32, so the same
  weights on a different device disagree at ~1e-4 and the difference grows with
  trunk depth. Measured discrimination: correct weights land at <= 5e-4,
  different weights at ~8e-1 -- three orders of magnitude apart, so --tol 5e-3
  separates them with enormous margin while never passing a real mismatch.

Exit 0 = every engine re-keyed. Exit 1 = mismatch; nothing is written.
"""
import argparse, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ENG = os.path.join(HERE, "models", "engines")
SEARCH = [os.path.join(HERE, "models"), os.path.join(HERE, "..", "checkpoints"),
          os.path.join(HERE, "..", "gateship"), os.path.join(HERE, "runs")]


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_source(want):
    """A checkpoint on disk whose FILE sha1 is `want`. Identity is the hash."""
    for d in SEARCH:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)):
            try:
                if sha1(f) == want:
                    return f
            except OSError:
                pass
    return None


def exact_match(src, cur):
    """-> (ok, message). Bit-identical state_dicts, key by key."""
    import torch
    a = torch.load(src, map_location="cpu", weights_only=False)
    b = torch.load(cur, map_location="cpu", weights_only=False)
    a = a.get("state_dict", a); b = b.get("state_dict", b)
    if set(a) != set(b):
        only = sorted(set(a) ^ set(b))[:4]
        return False, "different keys (e.g. %s)" % only
    for k in sorted(a):
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            return False, "%s: %s/%s vs %s/%s" % (k, tuple(x.shape), x.dtype,
                                                  tuple(y.shape), y.dtype)
        if x.numpy().tobytes() != y.numpy().tobytes():
            return False, "%s differs" % k
    return True, "%d tensors bit-identical" % len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "models", "model.pt"))
    ap.add_argument("--source", default="", help="the training checkpoint the engines were built from")
    ap.add_argument("--tol", type=float, default=5e-3, help="functional fallback only")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify-shipped", default="")
    a = ap.parse_args()

    import torch, torch.nn as nn, numpy as np
    import run as R

    metas = sorted(glob.glob(os.path.join(ENG, "*.plan.json")))
    if not metas:
        sys.exit("no engines under models/engines/")
    recorded = {json.load(open(m)).get("weights_sha1") for m in metas}
    if len(recorded) != 1:
        sys.exit("engines disagree about their weights: %s" % recorded)
    recorded = recorded.pop()

    new = sha1(a.weights)
    print("  models/model.pt      sha1 %s" % new)
    print("  engines built from   sha1 %s" % recorded)
    if new == recorded:
        print("\n  already keyed to this file; nothing to do.")
        return

    src = a.source or find_source(recorded)
    if src:
        ok, why = exact_match(src, a.weights)
        print("\n  EXACT proof against %s" % os.path.relpath(src, HERE))
        print("    %s  -> %s" % (why, "MATCH" if ok else "MISMATCH"))
        if not ok:
            sys.exit("\n  REFUSED: models/model.pt is not the checkpoint these engines "
                     "were built from.\n  Rebuild: python tools/trt_native.py all "
                     "--depths 3,6,10,13,16 --portable")
    else:
        print("\n  no local file hashes to %s; falling back to the FUNCTIONAL check."
              % recorded[:12])
        print("  (numerical only -- the reference was computed on the BUILD gpu)")
        ck = R.load_checkpoint(a.weights)
        state = ck.get("state_dict", ck)
        worst, ok = 0.0, True
        for mp in metas:
            m = json.load(open(mp)); tag = m["plan"][:-5]
            ri, ro = os.path.join(ENG, tag + ".ref_in.npy"), os.path.join(ENG, tag + ".ref_out.npy")
            if not (os.path.isfile(ri) and os.path.isfile(ro)):
                print("    %-12s NO REFERENCE" % tag); ok = False; continue
            model, arch = R.build_model(ck.get("config", {}) or {}, state)
            model.load_state_dict(state, strict=True)
            if m["depth"] < len(model.body):
                model.body = nn.Sequential(*list(model.body)[:m["depth"]])
            model = model.eval().to(a.device)
            with torch.no_grad():
                y = model(torch.from_numpy(np.load(ri)).to(a.device)).float().cpu().numpy()
            d = float(np.abs(y - np.load(ro)).max()); worst = max(worst, d)
            good = d <= a.tol; ok &= good
            print("    %-12s depth %-3d max|now-export| = %.3e   %s"
                  % (tag, m["depth"], d, "match" if good else "MISMATCH"))
        if not ok:
            sys.exit("\n  REFUSED: at least one engine does not match models/model.pt.\n"
                     "  Rebuild: python tools/trt_native.py all --depths 3,6,10,13,16 --portable")
        print("    worst %.3e vs tol %.1e (a wrong checkpoint measures ~8e-1)" % (worst, a.tol))

    for mp in metas:
        m = json.load(open(mp)); m["weights_sha1"] = new
        json.dump(m, open(mp, "w"), indent=2)
    idx = os.path.join(ENG, "index.json")
    if os.path.isfile(idx):
        j = json.load(open(idx)); j["weights_sha1"] = new
        for e in j.get("engines", []):
            e["weights_sha1"] = new
        json.dump(j, open(idx, "w"), indent=2)
    print("\n  re-keyed %d engine(s) %s -> %s" % (len(metas), recorded[:12], new[:12]))

    if a.verify_shipped:
        import re
        s = open(a.verify_shipped).read()
        s2, n = re.subn(r'^GOOD = "[0-9a-f]{40}"', 'GOOD = "%s"' % new, s, count=1, flags=re.M)
        if n:
            open(a.verify_shipped, "w").write(s2)
            print("  %s: GOOD -> %s" % (os.path.basename(a.verify_shipped), new))
        else:
            print("  WARNING: no GOOD = \"...\" line in %s" % a.verify_shipped)
    print("\n  now run:  python run_fast.py <in> <out>")


if __name__ == "__main__":
    main()
