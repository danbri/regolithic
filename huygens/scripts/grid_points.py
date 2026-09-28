#!/usr/bin/env python3
"""Grid SfM points into a DTM with an uncertainty map.

Input: an sfm.npz with E, N (km), h_m, sigma_h_m (per-point height sigma).
Each grid cell takes an inverse-variance-weighted Gaussian average of the
points (kernel SIGMA_M), iterated with Huber reweighting of each point's
residual against the gridded surface to suppress outliers. The per-cell
uncertainty combines the formal propagation (1/sqrt(sum w)) with the local
weighted scatter of residuals, whichever is larger. Cells with
uncertainty above MAX_SIG or too few effective points are masked.

Usage: grid_points.py SFM_NPZ OUT_NPZ [--sigma-m 200] [--max-sig 60]
       [--truth TRUTH_NPZ[:key]]   (report accuracy on synthetic truth)
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sfm"); ap.add_argument("out")
ap.add_argument("--sigma-m", type=float, default=200.0)
ap.add_argument("--max-sig", type=float, default=60.0)
ap.add_argument("--min-neff", type=float, default=4.0)
ap.add_argument("--sig-floor", type=float, default=15.0, help="m, added in quadrature to point sigmas")
ap.add_argument("--huber", type=float, default=2.0)
ap.add_argument("--truth", default=None)
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
meta = json.load(open(os.path.join(ROOT, "products/v1/FREEZE.json")))["meta"]
g = meta["grid_m"] / 1e3
ny, nx = meta["shape"]
xs = meta["x_centres_km"][0] + g * np.arange(nx)
ys = meta["y_centres_km"][0] + g * np.arange(ny)
S = np.load(a.sfm)
E, N, h = S["E"], S["N"], S["h_m"].astype(float)
sp = np.sqrt(S["sigma_h_m"].astype(float) ** 2 + a.sig_floor ** 2)
sk = a.sigma_m / 1e3
# separable Gaussian weights between points and grid rows/cols
Kx = np.exp(-0.5 * ((xs[None, :] - E[:, None]) / sk) ** 2)     # (n, nx)
Ky = np.exp(-0.5 * ((ys[None, :] - N[:, None]) / sk) ** 2)     # (n, ny)
rob = np.ones(len(h))
for it in range(5):
    wp = rob / sp ** 2
    W = np.einsum("ny,nx->yx", Ky * wp[:, None], Kx)
    Z = np.einsum("ny,nx->yx", Ky * (wp * h)[:, None], Kx) / np.maximum(W, 1e-30)
    # residual of each point against the surface at its location (bilinear)
    ci = np.clip((E - xs[0]) / g, 0, nx - 1.001); ri = np.clip((N - ys[0]) / g, 0, ny - 1.001)
    c0, r0 = ci.astype(int), ri.astype(int); fc, fr = ci - c0, ri - r0
    zp = (Z[r0, c0] * (1 - fc) * (1 - fr) + Z[r0, c0 + 1] * fc * (1 - fr) + Z[r0 + 1, c0] * (1 - fc) * fr + Z[r0 + 1, c0 + 1] * fc * fr)
    rn = np.abs(h - zp) / sp
    rob = np.where(rn < a.huber, 1.0, a.huber / rn)
wp = rob / sp ** 2
W = np.einsum("ny,nx->yx", Ky * wp[:, None], Kx)
W2 = np.einsum("ny,nx->yx", Ky ** 2 * (wp ** 2)[:, None], Kx ** 2)
neff = W ** 2 / np.maximum(W2, 1e-30)          # effective number of points per cell
res2 = np.einsum("ny,nx->yx", Ky * (wp * (h - zp) ** 2)[:, None], Kx) / np.maximum(W, 1e-30)
# uncertainty of the weighted mean: formal propagation of point sigmas, and
# the empirical version from the local residual scatter; take the larger
sig_scatter = np.sqrt(np.maximum(res2, 0) / np.maximum(neff, 1e-9))
sig_prop = np.sqrt(np.einsum("ny,nx->yx", Ky ** 2 * ((wp * sp) ** 2)[:, None], Kx ** 2)) / np.maximum(W, 1e-30)
sig = np.maximum(sig_prop, sig_scatter)
mask = (sig <= a.max_sig) & (neff >= a.min_neff)
Zm = np.where(mask, Z, np.nan)
np.savez_compressed(a.out, h_m=Zm, h_all_m=Z, sigma_m=sig, neff=neff, mask=mask, xs=xs, ys=ys)
print("grid: %.1f km^2 valid (%.0f%% of region); median sigma %.0f m; height p5/p95 %.0f/%.0f m" % (
    mask.sum() * g * g, 100 * mask.mean(), np.median(sig[mask]) if mask.any() else np.nan, *np.nanpercentile(Zm, [5, 95])))
if a.truth:
    path, key = (a.truth.split(":") + ["h_m"])[:2]
    T = np.load(path)[key]
    m = mask & np.isfinite(T)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    G = np.c_[np.ones(m.sum()), XX[m] - XX[m].mean(), YY[m] - YY[m].mean()]
    ct, *_ = np.linalg.lstsq(G, T[m], rcond=None); cz, *_ = np.linalg.lstsq(G, Zm[m], rcond=None)
    rt, rz = T[m] - G @ ct, Zm[m] - G @ cz
    k = (rz * rt).sum() / (rt * rt).sum()
    # smooth truth at the gridding scale for a fair comparison
    from scipy.ndimage import gaussian_filter
    Ts = gaussian_filter(np.nan_to_num(T - np.nanmean(T)), a.sigma_m / (g * 1e3))
    rts = Ts[m] - G @ np.linalg.lstsq(G, Ts[m], rcond=None)[0]
    print("vs truth on %.1f km^2: plane truth (%.1f, %.1f) grid (%.1f, %.1f) m/km; after plane removal corr %.3f "
          "(vs truth smoothed at kernel scale %.3f), slope %.2f, rms error %.0f m, median stated sigma %.0f m, error/sigma rms %.2f" % (
              m.sum() * g * g, ct[1], ct[2], cz[1], cz[2], np.corrcoef(rt, rz)[0, 1], np.corrcoef(rts, rz)[0, 1], k,
              (rz - rt).std(), np.median(sig[m]), ((rz - rt) / sig[m]).std()))
