#!/bin/sh
# Re-create data/ from the public archives. Checksums are in data/MANIFEST.jsonl.
set -e
cd "$(dirname "$0")/.."
B=https://archives.esac.esa.int/psa/ftp/CASSINI-HUYGENS/DISR/HP-SSA-DISR-2-3-EDR-RDR-V1.3
E=$B/EXTRAS
python3 scripts/fetch.py raw $(python3 -c "import json;print(' '.join(r['url'] for r in map(json.loads,open('data/MANIFEST.jsonl')) if r['tier']=='raw'))")
python3 scripts/fetch.py reference --tree https://archives.esac.esa.int/psa/ftp/Guest-Storage-Facility/IPGP_Titan_Huygens_V1.0/
