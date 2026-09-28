#!/bin/sh
# Fetch the open-access (CC BY) paper(s) in this folder and verify SHA-256.
# The Elsevier papers listed in SOURCE.md are not redistributable and must be supplied locally.
set -e
cd "$(dirname "$0")"
f=daudon2020_ess7_e2020EA001127.pdf
[ -f $f ] || curl -sSL -o $f "https://repository.arizona.edu/bitstreams/d070c5b6-c20c-4add-b82c-36b4325c8163/download"
echo "4bc9fe5e0fd87f8cdcaf47812797cbef16f16b0f6ad069bda2f45002d97b388b  $f" | sha256sum -c -
