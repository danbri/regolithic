#!/usr/bin/env bash
# Fetch the "Botanical Garden - Victoria House (VR Ready)" Gaussian Splat
# from PlayCanvas CDN. This scene is a streamed multi-tile LOD bundle
# (lod-meta.json + per-tile meta.json + per-tile WebP textures).
#
# Verifies:
#   - SHA-256 of lod-meta.json (root manifest pin)
#   - aggregate SHA-256 over (sorted_path, content) tuples (full-bundle pin)
#
# See SOURCE.md for attribution + license (CC BY 4.0).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="https://d28zzqy0iyovbz.cloudfront.net/6f697c4d/v1"
LOD_META_SHA256="4c8497336285f1b643073e390e61ac47783edd58a27ca8b406851372e1f2c0a3"
BUNDLE_SHA256="ff327961a15bbf532c42c15cc901a9876cd265bc253aeaa1b8cb9bee26b6e7c2"

command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }

python3 - "$here" "$base" "$LOD_META_SHA256" "$BUNDLE_SHA256" <<'PY'
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

here, base, lod_sha_want, bundle_sha_want = sys.argv[1:5]
root = Path(here)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(path):
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base}/{path}"
    urllib.request.urlretrieve(url, dest)
    return dest

# 1. Root manifest
lm = fetch("lod-meta.json")
got = sha256_file(lm)
if got != lod_sha_want:
    print(f"lod-meta.json sha256 mismatch: got {got}, want {lod_sha_want}", file=sys.stderr)
    sys.exit(1)
lod = json.loads(lm.read_bytes())

# 2. Discover every referenced file (tile meta.json files + their webp refs)
paths = ["lod-meta.json"]
tile_metas = list(lod["filenames"])

def fetch_tile(tile_meta_path):
    fetch(tile_meta_path)
    tile_dir = tile_meta_path.rsplit("/", 1)[0]
    tm = json.loads((root / tile_meta_path).read_bytes())
    refs = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, str) and o.endswith(".webp"):
            refs.add(o)
    walk(tm)
    out = [tile_meta_path]
    for w in sorted(refs):
        p = f"{tile_dir}/{w}"
        fetch(p)
        out.append(p)
    return out

# Skip already-downloaded files where possible (size > 0 is good enough; the
# aggregate hash catches any corruption at the end).
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(fetch_tile, tm) for tm in tile_metas]
    for f in as_completed(futs):
        paths.extend(f.result())

# 3. Aggregate hash over (sorted path, NUL, content)
h = hashlib.sha256()
for p in sorted(set(paths)):
    h.update(p.encode())
    h.update(b"\0")
    with open(root / p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
agg = h.hexdigest()
if agg != bundle_sha_want:
    print(f"bundle aggregate sha256 mismatch: got {agg}, want {bundle_sha_want}", file=sys.stderr)
    sys.exit(1)

total = sum((root / p).stat().st_size for p in set(paths))
print(f"  ok: {len(set(paths))} files, {total/1024/1024:.1f} MiB, bundle sha256 verified")
PY
