#!/usr/bin/env python3
"""Pull N images per category from the NFFA-EUROPE SEM dataset.

    python tools/fetch_nffa.py --list
    python tools/fetch_nffa.py --out ../nffa --n 200

KLA's round-2 data is almost certainly derived from this set: the ten
categories in their brief match it verbatim, and the natives are 1024x728 --
big enough to cut the real 512x512 crops the round-2 training set never had.

The whole archive is 12.1 GB across ten per-category tars. We do not want it.
TAR is a SEQUENTIAL format, so we stream each archive, take the first N images,
and hang up. That pulls roughly (N x mean image size) per category instead of
the whole file -- about 1.5 GB for N=200 rather than 12.1 GB.

Only the stdlib plus `requests`.
"""
import argparse, io, json, os, ssl, sys, tarfile, time
import urllib.request

# ---------------------------------------------------------------- TLS setup
# Windows Python does not read the OS certificate store, and antivirus TLS
# interception breaks certifi's bundle too. Try, in order: the OS store via
# truststore, then certifi, then the stdlib default. --insecure is a last
# resort the user has to ask for.
INSECURE = False


def _ctx():
    if INSECURE:
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c
    try:
        import truststore                      # uses the Windows cert store
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    return ssl.create_default_context()


def _open(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "forgex/1.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=_ctx())

RECORD = "qtq9v-ys023"
API    = "https://b2share.eudat.eu/api/records/{rec}"
FILE   = "https://b2share.eudat.eu/records/{rec}/files/{name}?download=1"

# name -> approximate archive size, for ordering smallest-first
CATEGORIES = [
    ("Fibres.tar",                      84),
    ("Porous_Sponge.tar",              119),
    ("Films_Coated_Surface.tar",       198),
    ("Tips.tar",                       677),
    ("Biological.tar",                 700),
    ("Powder.tar",                     856),
    ("Patterned_surface.tar",         2000),
    ("Nanowires.tar",                 2100),
    ("Particles.tar",                 2300),
    ("MEMS_devices_and_electrodes.tar", 3100),
]
IMG = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def listing(rec):
    """Ask the API what files exist. B2SHARE returns several shapes depending on
    version -- files as a list of dicts, as a bucket URL, or under metadata --
    so try each rather than assuming one."""
    with _open(API.format(rec=rec), timeout=60) as r:
        meta = json.load(r)

    def rows(v):
        out = []
        if isinstance(v, list):
            for f in v:
                if isinstance(f, dict):
                    k = f.get("key") or f.get("filename") or f.get("name")
                    if k:
                        out.append((k, f.get("size", 0)))
                elif isinstance(f, str):
                    out.append((f.rsplit("/", 1)[-1], 0))
        return out

    for v in (meta.get("files"), meta.get("contents"),
              (meta.get("metadata") or {}).get("_files")):
        got = rows(v)
        if got:
            return got

    # files given as a bucket URL -> fetch it
    for u in (meta.get("files") if isinstance(meta.get("files"), str) else None,
              (meta.get("links") or {}).get("files")):
        if isinstance(u, str) and u.startswith("http"):
            with _open(u, timeout=60) as r2:
                sub = json.load(r2)
            got = rows(sub.get("contents") or sub.get("files") or sub)
            if got:
                return got

    raise ValueError(f"unrecognised API shape; top-level keys: {sorted(meta)[:12]}")


def grab(rec, name, n, outdir, retries=3):
    """Stream one tar, write the first n images, then stop reading."""
    dest = os.path.join(outdir, name[:-4])
    os.makedirs(dest, exist_ok=True)
    have = len([f for f in os.listdir(dest) if f.lower().endswith(IMG)])
    if have >= n:
        print(f"  {name:<34} already have {have}, skipping", flush=True)
        return have

    url = FILE.format(rec=rec, name=name)
    for attempt in range(retries):
        got, read = have, 0
        t0 = time.perf_counter()
        try:
            with _open(url) as resp:
                # 'r|' = streaming mode: never seeks, so we can abort early
                with tarfile.open(fileobj=resp, mode="r|") as tf:
                    for m in tf:
                        if not m.isfile() or not m.name.lower().endswith(IMG):
                            continue
                        data = tf.extractfile(m).read()
                        read += len(data)
                        with open(os.path.join(dest, os.path.basename(m.name)), "wb") as fh:
                            fh.write(data)
                        got += 1
                        if got >= n:
                            break
            el = time.perf_counter() - t0
            print(f"  {name:<34} {got:>4} images   {read/1e6:>7.1f} MB   {el:>5.0f}s", flush=True)
            return got
        except Exception as e:
            print(f"  {name:<34} attempt {attempt+1} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../nffa")
    ap.add_argument("--n", type=int, default=200, help="images per category")
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--only", default="", help="comma-separated category names")
    ap.add_argument("--list", action="store_true", help="print the record's files and exit")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification -- last resort when antivirus "
                         "intercepts HTTPS and no CA bundle works")
    a = ap.parse_args()

    global INSECURE
    INSECURE = a.insecure
    if INSECURE:
        print("WARNING: TLS verification disabled for this run.\n", flush=True)
    else:
        try:
            import truststore; print("TLS: using the OS certificate store (truststore)")
        except Exception:
            try:
                import certifi; print("TLS: using certifi's CA bundle")
            except Exception:
                print("TLS: stdlib default (install truststore or certifi if this fails)")

    if a.list:
        try:
            for k, s in listing(a.record):
                print(f"{k:<40}{s/1e6:>10.1f} MB")
        except Exception as e:
            print(f"API listing failed ({e}). Falling back to the known names:")
            for k, s in CATEGORIES:
                print(f"{k:<40}{s:>10} MB (approx)")
        return

    names = [c for c, _ in CATEGORIES]
    if a.only:
        want = {w.strip().lower() for w in a.only.split(",")}
        names = [c for c in names if c[:-4].lower() in want or c.lower() in want]

    os.makedirs(a.out, exist_ok=True)
    print(f"NFFA-EUROPE SEM  ->  {os.path.abspath(a.out)}   "
          f"{a.n} images x {len(names)} categories\n", flush=True)
    t0 = time.perf_counter()
    total = 0
    for name in names:                      # smallest archives first
        total += grab(a.record, name, a.n, a.out)
    print(f"\n{total} images in {(time.perf_counter()-t0)/60:.1f} min -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
