#!/usr/bin/env python3
"""Paired comparison of held-out NCC between sweep configs and the baseline.
Usage: paired.py SWEEPDIR"""
import glob, json, os, sys
import numpy as np
root = sys.argv[1]
def load(cfg):
    out = {}
    for f in glob.glob(os.path.join(root, cfg, "f*/heldout.json")):
        for e in json.load(open(f)):
            out[(e["num"])] = e["ncc"]
    return out
base = load("baseline")
rng = np.random.default_rng(0)
for d in sorted(glob.glob(os.path.join(root, "*/"))):
    cfg = os.path.basename(d.rstrip("/"))
    if cfg == "baseline":
        continue
    x = load(cfg)
    k = sorted(set(x) & set(base))
    diff = np.array([x[i] - base[i] for i in k])
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("%-14s n=%d  mean diff %+.4f  95%% CI [%+.4f, %+.4f]  better in %d/%d views" % (cfg, len(k), diff.mean(), lo, hi, (diff > 0).sum(), len(k)))
