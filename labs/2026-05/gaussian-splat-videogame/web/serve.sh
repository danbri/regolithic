#!/usr/bin/env bash
# Serve the repo root over HTTP so relative paths to third_party/ resolve.
# Open: http://localhost:8000/labs/2026-05/gaussian-splat-videogame/web/
#
# For WebXR (Quest 3) you need HTTPS or localhost. Quest browser can reach a
# dev machine over the local network with a self-signed cert (e.g. `mkcert`)
# or by enabling "Allow access to localhost devices" in chrome://flags.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
port="${PORT:-8000}"

echo "Serving repo root: $root"
echo "Open: http://localhost:$port/labs/2026-05/gaussian-splat-videogame/web/"
cd "$root"
exec python3 -m http.server "$port"
