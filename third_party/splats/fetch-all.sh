#!/usr/bin/env bash
# Run every per-scene fetch.sh under third_party/splats/.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
fail=0
for f in "$here"/*/fetch.sh; do
  [ -x "$f" ] || chmod +x "$f"
  echo "==> $(dirname "$f" | xargs basename)"
  if ! "$f"; then
    echo "  FAILED: $f" >&2
    fail=1
  fi
done
exit "$fail"
