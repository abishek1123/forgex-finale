#!/usr/bin/env python3
"""Build ONE self-contained file carrying the whole project context.

    python tools/bundle_context.py                 -> docs/CONTEXT_BUNDLE.md
    python tools/bundle_context.py --out x.md      -> somewhere else
    python tools/bundle_context.py --lean          -> skip the literature map

Use it when an assistant cannot read the repository directly and you need
something to paste or upload. Regenerate it whenever the documents change --
never hand-edit the bundle, edit the sources.

Everything in the bundle is a copy of a file that is in git. There is no content
here that does not exist somewhere else, which is the point: one source of truth,
many ways to deliver it.
"""
import argparse
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (path, heading, how to render). Order matters -- this is reading order.
DOCS = [
    ("PROJECT_INSTRUCTIONS.md",                     "Project instructions",          "md"),
    ("docs/HANDOFF.md",               "The project, in full",          "md"),
    ("HACKATHON_CONTEXT.md",          "Live state",                    "md"),
    ("docs/CHECKPOINTS.md",           "Checkpoints",                   "md"),
    ("docs/REPRODUCIBILITY.md",       "Reproducibility",               "md"),
    ("docs/DATA.md",                  "Data",                          "md"),
    ("CONTRIBUTING.md",               "How the team works",            "md"),
    ("docs/LITERATURE.md",            "Literature",                    "md"),
]

TABLES = [
    ("docs/queue/testset_results.csv", "The 11-checkpoint field (organisers' test set)"),
    ("docs/capacity_sweep.csv",        "Capacity sweep (held-out split)"),
    ("docs/timesweep.csv",             "Timing and capacity on the test set"),
    ("docs/noise_sweep_queue.csv",     "Nine-level noise sweep"),
    ("docs/per_category.csv",          "Per-morphology gain over bicubic"),
    ("docs/EXPERIMENTS.csv",           "Experiments run during the finale"),
    ("docs/results.csv",               "Every scored run, chronological"),
]

LEAN_SKIP = {"docs/LITERATURE.md", "docs/results.csv"}


def read(rel):
    p = os.path.join(ROOT, *rel.split("/"))
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read().rstrip("\n")


def git_sha():
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode:
            return "no-git"
        d = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")
    except Exception:
        return "no-git"


def demote(md, by=1):
    """Push every ATX heading down a level so the bundle has one hierarchy.

    The source file's own H1 is dropped -- the bundle supplies its own heading
    for each section, and keeping both reads as a stutter.
    """
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines = lines[i + 1:]
            while lines and not lines[0].strip():
                lines.pop(0)
            break
        if line.strip() and not line.startswith("#"):
            break
    out, fence = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("#"):
            line = "#" * by + line
        out.append(line)
    return "\n".join(out)


def csv_as_table(text, max_rows=40):
    rows = [r for r in text.split("\n") if r.strip() and not r.lstrip().startswith("#")]
    comments = [r for r in text.split("\n") if r.lstrip().startswith("#")]
    if not rows:
        return "_(empty)_"
    head, body = rows[0].split(","), [r.split(",") for r in rows[1:]]
    trimmed = len(body) - max_rows
    body = body[:max_rows]
    n = len(head)
    out = []
    if comments:
        out += [l.lstrip("# ").rstrip() for l in comments] + [""]
    out.append("| " + " | ".join(h.strip() for h in head) + " |")
    out.append("|" + "---|" * n)
    for r in body:
        r = (r + [""] * n)[:n]
        out.append("| " + " | ".join(c.strip() for c in r) + " |")
    if trimmed > 0:
        out.append("")
        out.append("_(%d further rows omitted -- read the CSV itself)_" % trimmed)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "CONTEXT_BUNDLE.md"))
    ap.add_argument("--lean", action="store_true",
                    help="skip the literature map and the full results log")
    a = ap.parse_args()

    sha = git_sha()
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts, included, missing = [], [], []

    parts.append(
        "# ForgeX — complete project context\n\n"
        "Team ForgeX (Abishek SR, Anmol BA, Hardik) — KLA PS01, "
        "\"AI-Based Restoration of Degraded Images for Semiconductor Inspection\", "
        "SEMICON India 2026.\n\n"
        "**Generated %s from commit `%s` by `tools/bundle_context.py`.** Every "
        "section below is a copy of a file in the repository. Do not edit this "
        "file — edit the source and regenerate.\n\n"
        "If you are an AI assistant: read the whole thing before answering any "
        "question about this project. Then state the shipped model's PSNR, SSIM "
        "and LPIPS back to whoever gave you this, so they know you have it.\n"
        % (when, sha))

    body = []
    for rel, title, _ in DOCS:
        if a.lean and rel in LEAN_SKIP:
            continue
        txt = read(rel)
        if txt is None:
            missing.append(rel)
            continue
        included.append(rel)
        body.append("---\n\n## %s\n\n`%s`\n\n%s" % (title, rel, demote(txt, 2)))

    tbl = []
    for rel, title in TABLES:
        if a.lean and rel in LEAN_SKIP:
            continue
        txt = read(rel)
        if txt is None:
            missing.append(rel)
            continue
        included.append(rel)
        tbl.append("### %s\n\n`%s`\n\n%s" % (title, rel, csv_as_table(txt)))
    if tbl:
        body.append("---\n\n## Measurements\n\n"
                    "These are the numbers of record. A figure not traceable to "
                    "one of these tables does not go in a document or on a "
                    "slide.\n\n" + "\n\n".join(tbl))

    parts.append("\n\n".join(body))

    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    text = "\n\n".join(parts) + "\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    words = len(text.split())
    print("\n  wrote %s" % os.path.relpath(out, ROOT))
    print("  %d files, %s characters, ~%s words, ~%s tokens"
          % (len(included), format(len(text), ","), format(words, ","),
             format(int(words * 1.35), ",")))
    print("  from commit %s" % sha)
    if missing:
        print("\n  not found (skipped): %s" % ", ".join(missing))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
