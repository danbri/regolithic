#!/usr/bin/env python3
"""Split-half consistency of the v3 chain: two DTMs from disjoint exposure halves.
Usage: split_half.py TOOLCHAIN_DIR [SEED]   (HDTM_SYNTH must be set for synthetic dirs)
Prints the correlation of the two halves (planes removed) at 0 and 100 m smoothing."""
import os, subprocess, sys
import numpy as np
from scipy.ndimage import gaussian_filter
d = sys.argv[1]; seed = sys.argv[2] if len(sys.argv) > 2 else "1"
here = os.path.dirname(os.path.abspath(__file__))
grids = []
for k in (0, 1):
    od = os.path.join(d, "priorba_half%s_%d" % (seed, k))
    if not os.path.exists(os.path.join(od, "sfm.npz")):
        subprocess.run([sys.executable, os.path.join(here, "colmap_prior_ba.py"), d, "--half", "%s:%d" % (seed, k)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    g = os.path.join(od, "grid_k70.npz")
    subprocess.run([sys.executable, os.path.join(here, "grid_points.py"), os.path.join(od, "sfm.npz"), g, "--sigma-m", "70", "--max-sig", "150", "--min-neff", "2"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    grids.append(np.load(g))
m = grids[0]["mask"] & grids[1]["mask"]
ny, nx = m.shape
YY, XX = np.mgrid[0:ny, 0:nx]
out = []
for s in (0, 5):
    hs = []
    for G in grids:
        h = np.nan_to_num(G["h_all_m"])
        if s:
            w = gaussian_filter(m * 1.0, s); h = gaussian_filter(np.where(m, h, 0), s) / np.maximum(w, 1e-6)
        Gm = np.c_[np.ones(m.sum()), XX[m], YY[m]]
        hs.append(h[m] - Gm @ np.linalg.lstsq(Gm, h[m], rcond=None)[0])
    out.append("%dm r %.2f" % (s * 20, np.corrcoef(hs[0], hs[1])[0, 1]))
print("%s split-half (seed %s): overlap %.1f km2, %s" % (os.path.basename(d.rstrip('/')), seed, m.sum() * 4e-4, ", ".join(out)))
