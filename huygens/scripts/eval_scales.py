#!/usr/bin/env python3
"""Accuracy of a height grid against truth as a function of horizontal scale.
Both are smoothed with a Gaussian (normalised within the mask) and planes removed.
Usage: eval_scales.py RESULT TRUTH_NPZ MASK_NPZ label"""
import sys, numpy as np
from scipy.ndimage import gaussian_filter
R = np.load(sys.argv[1]); h = R["h"] * 1e3 if "h" in R.files else np.nan_to_num(R["h_all_m"] if "h_all_m" in R.files else R["h_m"])
T = np.load(sys.argv[2])["h_m"]; m = np.load(sys.argv[3])["mask"] & np.isfinite(T)
ny, nx = T.shape; YY, XX = np.mgrid[0:ny, 0:nx] * 0.02
def sm(a, s):
    if s == 0: return a
    mm = m.astype(float); return gaussian_filter(np.where(m, a, 0), s) / np.maximum(gaussian_filter(mm, s), 1e-6)
out = []
for s_m in (0, 100, 200, 400, 800):
    s = s_m / 20
    Ts, hs = sm(np.nan_to_num(T), s), sm(h, s)
    G = np.c_[np.ones(m.sum()), XX[m] - XX[m].mean(), YY[m] - YY[m].mean()]
    rt = Ts[m] - G @ np.linalg.lstsq(G, Ts[m], rcond=None)[0]; rh = hs[m] - G @ np.linalg.lstsq(G, hs[m], rcond=None)[0]
    out.append("%4dm: r %.2f k %.2f rms %.0f/%.0f" % (s_m, np.corrcoef(rt, rh)[0, 1], (rh * rt).sum() / (rt * rt).sum(), (rh - rt).std(), rt.std()))
print("%-16s " % sys.argv[4] + " | ".join(out))
