#!/usr/bin/env python3
"""Compare a dense result (result.npz 'h' in km, or npz 'h_m') with a truth grid over a mask.
Usage: eval_grid.py RESULT TRUTH_NPZ [MASK_NPZ] [label]"""
import sys, numpy as np
from scipy.ndimage import gaussian_filter
R = np.load(sys.argv[1]); h = R["h"] * 1e3 if "h" in R.files else R["h_m"]
T = np.load(sys.argv[2])["h_m"]
m = np.isfinite(T) & np.isfinite(h)
if len(sys.argv) > 3 and sys.argv[3].endswith(".npz"):
    m &= np.load(sys.argv[3])["mask"]
lab = sys.argv[-1]
ny, nx = T.shape
YY, XX = np.mgrid[0:ny, 0:nx] * 0.02
G = np.c_[np.ones(m.sum()), XX[m] - XX[m].mean(), YY[m] - YY[m].mean()]
ct = np.linalg.lstsq(G, T[m], rcond=None)[0]; ch = np.linalg.lstsq(G, h[m], rcond=None)[0]
rt, rh = T[m] - G @ ct, h[m] - G @ ch
Ts = gaussian_filter(np.nan_to_num(T - np.nanmean(T)), 5)
rts = Ts[m] - G @ np.linalg.lstsq(G, Ts[m], rcond=None)[0]
print("%-22s %.1f km2 | plane truth (%.0f, %.0f) est (%.0f, %.0f) m/km | corr %.3f (vs 100 m-smoothed truth %.3f) | slope %.2f | rms err %.0f m | std truth %.0f est %.0f" % (
    lab, m.sum() * 4e-4, ct[1], ct[2], ch[1], ch[2], np.corrcoef(rt, rh)[0, 1], np.corrcoef(rts, rh)[0, 1],
    (rh * rt).sum() / (rt * rt).sum(), (rh - rt).std(), rt.std(), rh.std()))
