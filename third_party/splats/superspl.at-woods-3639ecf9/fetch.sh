#!/usr/bin/env bash
# Fetch the "Woods" Gaussian Splat scene from PlayCanvas CDN and verify SHA-256.
# See SOURCE.md for attribution + license.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="https://d28zzqy0iyovbz.cloudfront.net/3639ecf9/v1"

# filename  sha256
files=(
  "scene.compressed.ply  d01d964bf40e8da207353aa8838ee782aafabbaf69cfcb0e55322e99d3102d91"
)

sha_cmd() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

for entry in "${files[@]}"; do
  name="${entry%% *}"
  want="${entry##* }"
  dest="$here/$name"
  if [ -f "$dest" ] && [ "$(sha_cmd "$dest")" = "$want" ]; then
    echo "  skip $name (sha256 ok)"
    continue
  fi
  echo "  get  $name"
  curl -fsSL --retry 4 --retry-delay 2 -o "$dest" "$base/$name"
  got="$(sha_cmd "$dest")"
  if [ "$got" != "$want" ]; then
    echo "  SHA-256 mismatch for $name (got $got, want $want)" >&2
    exit 1
  fi
done
