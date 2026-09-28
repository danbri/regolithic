#!/usr/bin/env python3
"""Turn a seeded dense fit into a DTM product with calibrated uncertainty.

Heights: the dense result (result.npz, h in km) smoothed with a Gaussian of
SMOOTH_M metres, normalised within the support mask of the gridded SfM
points (grid.npz), mean removed. Uncertainty: the gridding sigma, smoothed
the same way, times a calibration factor. The factor is estimated on
synthetic cases with known truth (--calibrate TRUTH ...), as the rms of
(estimate - truth) / sigma after removing best-fit planes from both
(regional slope uncertainty is reported separately).

Usage:
  make_product.py calibrate --smooth-m 300 RESULT GRID TRUTH [RESULT GRID TRUTH ...]
  make_product.py build --smooth-m 300 --calib F RESULT GRID OUTDIR
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["calibrate", "build"])
ap.add_argument("files", nargs="+")
ap.add_argument("--smooth-m", type=float, default=300.0)
ap.add_argument("--calib", type=float, default=1.0)
a = ap.parse_args()
CELL = 20.0


def product(result, grid):
    R = np.load(result); G = np.load(grid)
    h = R["h"] * 1e3
    m = G["mask"].astype(bool)
    s = a.smooth_m / CELL
    w = gaussian_filter(m.astype(float), s)
    hs = gaussian_filter(np.where(m, h, 0), s) / np.maximum(w, 1e-6)
    sg = np.sqrt(gaussian_filter(np.where(m, G["sigma_m"] ** 2, 0), s) / np.maximum(w, 1e-6))
    hs = hs - hs[m].mean()
    return hs, sg, m


def detrend(z, m):
    ny, nx = z.shape
    YY, XX = np.mgrid[0:ny, 0:nx]
    Gm = np.c_[np.ones(m.sum()), XX[m], YY[m]]
    c = np.linalg.lstsq(Gm, z[m], rcond=None)[0]
    return z[m] - Gm @ c, c[1:] * 1e3 / CELL


if a.mode == "calibrate":
    ratios, rows = [], []
    for i in range(0, len(a.files), 3):
        res, grid, truth = a.files[i:i + 3]
        hs, sg, m = product(res, grid)
        T = np.load(truth)["h_m"]
        m = m & np.isfinite(T)
        w = gaussian_filter(m.astype(float), a.smooth_m / CELL)
        Ts = gaussian_filter(np.where(m, np.nan_to_num(T), 0), a.smooth_m / CELL) / np.maximum(w, 1e-6)
        rt, pt = detrend(Ts, m); rh, ph = detrend(hs, m)
        e = (rh - rt) / sg[m]
        ratios.append(e)
        rows.append("%s: relief corr %.2f, scale %.2f, rms err %.1f m (truth relief %.1f m), median sigma %.1f m, err/sigma rms %.2f; plane truth (%.0f, %.0f) est (%.0f, %.0f) m/km" % (
            os.path.dirname(res), np.corrcoef(rt, rh)[0, 1], (rh * rt).sum() / (rt * rt).sum(), (rh - rt).std(), rt.std(),
            np.median(sg[m]), e.std(), pt[0], pt[1], ph[0], ph[1]))
    print("\n".join(rows))
    print("calibration factor (pooled err/sigma rms) at %.0f m smoothing: %.2f" % (a.smooth_m, np.concatenate(ratios).std()))
else:
    res, grid, out = a.files
    hs, sg, m = product(res, grid)
    os.makedirs(out, exist_ok=True)
    sig = sg * a.calib
    rr, pl = detrend(hs, m)
    np.savez_compressed(os.path.join(out, "dtm.npz"), h_all_m=hs, h_m=np.where(m, hs, np.nan), sigma_m=sig, mask=m)
    info = dict(smooth_m=a.smooth_m, calib=a.calib, area_km2=float(m.sum() * (CELL / 1e3) ** 2),
                relief_std_after_plane_m=float(rr.std()), plane_m_per_km=pl.tolist(),
                sigma_median_m=float(np.median(sig[m])), height_p5_p95=np.percentile(hs[m], [5, 95]).tolist())
    json.dump(info, open(os.path.join(out, "product_info.json"), "w"), indent=1)
    print(json.dumps(info))
