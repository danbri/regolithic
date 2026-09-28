#!/usr/bin/env python3
"""Resample the IPGP DTM onto our frozen grid using derived/ipgp_to_ours.json.
Writes derived/ipgp_h_on_grid.npz (h_m with NaN outside coverage)."""
import json, os
import numpy as np, cv2, tifffile
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
R = "data/reference/Guest-Storage-Facility/IPGP_Titan_Huygens_V1.0/"
dtm = tifffile.imread(R + "IPGP_DTM.tif").astype(np.float32)
tfw = [float(x) for x in open(R + "IPGP_DTM.tfw").read().split()]
reg = json.load(open("derived/ipgp_to_ours.json"))
M = np.array(reg["ipgp_world_from_ours_EN"]["M"]); b = np.array(reg["ipgp_world_from_ours_EN"]["b_km"])
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
g = meta["grid_m"] / 1e3
ny, nx = meta["shape"]
E = meta["x_centres_km"][0] + g * np.arange(nx)
N = meta["y_centres_km"][0] + g * np.arange(ny)
EE, NN = np.meshgrid(E, N)
W = np.einsum("ij,jyx->iyx", M, np.stack([EE, NN])) + b[:, None, None]   # IPGP world km
Xm, Ym = W[0] * 1e3, W[1] * 1e3
col = (Xm - tfw[4]) / tfw[0]
row = (Ym - tfw[5]) / tfw[3]
valid = (dtm > 0).astype(np.float32)
h = cv2.remap(dtm, col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
v = cv2.remap(valid, col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
h = np.where(v > 0.999, h, np.nan)
np.savez_compressed("derived/ipgp_h_on_grid.npz", h_m=h)
print("IPGP heights on our grid: %d texels (%.1f km^2), range p1/p99 %.0f/%.0f m, std %.0f m" % (
    np.isfinite(h).sum(), np.isfinite(h).sum() * g * g, *np.nanpercentile(h, [1, 99]), np.nanstd(h)))
