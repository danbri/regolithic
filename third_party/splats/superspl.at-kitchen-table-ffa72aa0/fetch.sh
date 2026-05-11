#!/usr/bin/env bash
# Fetch the "Kitchen Table" Gaussian Splat scene (SOG format) and verify SHA-256.
# See SOURCE.md for attribution + license.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="https://d28zzqy0iyovbz.cloudfront.net/ffa72aa0/v1"

# filename  sha256
files=(
  "meta.json            2f97954b56368e988547b7d6167ba10aab2c69b9a045b4db74da7e99cced5119"
  "means_l.webp         628b75ff2346d73e29d67a9313f055435c0eceef08ce19e6d2f0919cbf97233c"
  "means_u.webp         373a316648fa88d7e2e5357def3c94ae53d334c65f3ed26ba0b424ff2cff3f38"
  "scales.webp          f01d6e8c0a663a34e1ab649eeca987a069687b066b89ab50af3d8ef78f796bc3"
  "quats.webp           3c68f3173383fe60db75260eec8105ff0bd5fbd0b0fa2490dd828a96bb959986"
  "sh0.webp             d34e0c7973647864a32fb0999e3326dabca5828b97050e2ff420fd62bf38340a"
  "shN_centroids.webp   30218046a78dbfb9a12c74e1591ac23fd592b525846a789ce46659cd8b9f4563"
  "shN_labels.webp      647355143c0449131b4a55a0db3d56c52491e8daa71d7d4975e4b2fdeef96904"
)

sha_cmd() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

for entry in "${files[@]}"; do
  name="$(echo "$entry" | awk '{print $1}')"
  want="$(echo "$entry" | awk '{print $2}')"
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
