#!/usr/bin/env python3
"""Download external inputs with provenance.

Each file is written under data/<tier>/<archive-relative path> and one JSON
line (url, retrieved_utc, sha256, bytes, tier) is appended to
data/MANIFEST.jsonl. Tier is "raw" (may drive reconstruction) or
"reference" (validation only; withheld until geometry is frozen).

Usage: fetch.py raw|reference URL [URL ...]
       fetch.py raw|reference --tree URL/   (recursive directory listing)
"""
import concurrent.futures, datetime, hashlib, json, os, re, sys, threading, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MANIFEST = os.path.join(ROOT, "data", "MANIFEST.jsonl")
PSA = "https://archives.esac.esa.int/psa/ftp/"
LOCK = threading.Lock()


def local_path(tier, url):
    rel = url[len(PSA):] if url.startswith(PSA) else urllib.parse.urlparse(url).netloc + urllib.parse.urlparse(url).path
    return os.path.join(ROOT, "data", tier, urllib.parse.unquote(rel))


def known():
    if not os.path.exists(MANIFEST):
        return {}
    out = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        out[(r["tier"], r["url"])] = r
    return out


def fetch(tier, url, seen):
    dest = local_path(tier, url)
    if (tier, url) in seen and os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    rec = {"url": url, "tier": tier, "path": os.path.relpath(dest, ROOT),
           "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    with LOCK:
        with open(MANIFEST, "a") as f:
            f.write(json.dumps(rec) + "\n")
        seen[(tier, url)] = rec
    print(rec["path"], rec["bytes"])


def tree(url):
    html = urllib.request.urlopen(url, timeout=120).read().decode("latin-1")
    for href in re.findall(r'href="([^"?/][^"]*)"', html):
        u = urllib.parse.urljoin(url, href)
        if href.endswith("/"):
            yield from tree(u)
        else:
            yield u


def main():
    tier, args = sys.argv[1], sys.argv[2:]
    assert tier in ("raw", "reference")
    seen = known()
    urls = list(tree(args[1])) if args[0] == "--tree" else args
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(fetch, tier, u, seen) for u in urls]:
            f.result()


if __name__ == "__main__":
    main()
