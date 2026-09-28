#!/usr/bin/env python3
"""Compare SfM points (sfm.npz) with a height grid (truth or reference).
Usage: eval_points.py SFMDIR HEIGHT_NPZ[:key] [label]"""
import json, sys, numpy as np
d, spec = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else spec
path, key = (spec.split(":") + ["h_m"])[:2]
T = np.load(path)[key]
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
g = meta["grid_m"] / 1e3
x0, y0 = meta["x_centres_km"][0], meta["y_centres_km"][0]
S = np.load(d + "/sfm.npz")
c = np.clip(np.round((S["E"] - x0) / g).astype(int), 0, T.shape[1] - 1)
r = np.clip(np.round((S["N"] - y0) / g).astype(int), 0, T.shape[0] - 1)
t = T[r, c]
ok = np.isfinite(t)
h, t, s, E, N = S["h_m"][ok], t[ok], S["sigma_h_m"][ok], S["E"][ok], S["N"][ok]
G = np.c_[np.ones(ok.sum()), E - E.mean(), N - N.mean()]
ct, *_ = np.linalg.lstsq(G, t, rcond=None); ch, *_ = np.linalg.lstsq(G, h, rcond=None)
rt, rh = t - G @ ct, h - G @ ch
k = (rh * rt).sum() / (rt * rt).sum()
print("%-18s n=%d  plane (dE,dN m/km) truth (%.1f, %.1f) sfm (%.1f, %.1f) | after plane removal: corr %.3f, "
      "regression slope sfm~truth %.2f, std truth %.0f sfm %.0f, rms diff %.0f m, median sigma %.0f m, "
      "normalised residual rms %.2f" % (label, ok.sum(), ct[1], ct[2], ch[1], ch[2], np.corrcoef(rt, rh)[0, 1], k,
                                          rt.std(), rh.std(), (rh - rt).std(), np.median(s), ((rh - rt) / s).std()))
