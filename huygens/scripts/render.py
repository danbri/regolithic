#!/usr/bin/env python3
"""Render a reconstruction: brightness map, hillshade, colour heights.

Usage: render.py RUNDIR [RUNDIR ...]  -> RUNDIR/overview.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hillshade(h, res, az=315, alt=35, exag=3.0):
    gy, gx = np.gradient(h * exag, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(90 - alt)
    return np.clip(np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect), 0, 1)


for run in sys.argv[1:]:
    r = np.load(os.path.join(run, "result.npz"))
    h, A, xs, ys, res = r["h"] * 1e3, r["A"], r["xs"], r["ys"], float(r["res"]) * 1e3
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    lo, hi = np.percentile(A, [1, 99])
    ax[0].imshow(A, origin="lower", extent=ext, cmap="gray", vmin=lo, vmax=hi)
    ax[0].set_title("Surface brightness A")
    ax[1].imshow(hillshade(h, res), origin="lower", extent=ext, cmap="gray")
    ax[1].set_title("Hillshade (3x vertical exaggeration)")
    lo, hi = np.percentile(h, [1, 99])
    im = ax[2].imshow(h, origin="lower", extent=ext, cmap="terrain", vmin=lo, vmax=hi)
    ax[2].contour(xs, ys, h, levels=np.arange(np.floor(lo / 25) * 25, hi, 25), colors="k", linewidths=0.3)
    ax[2].set_title("Height (m, mean removed), 25 m contours")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xlabel("km east of landing site")
        a.set_ylabel("km north")
        a.plot(0, 0, "r+") if (ext[0] < 0 < ext[1] and ext[2] < 0 < ext[3]) else None
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "overview.png"), dpi=90)
    print(run, "h p1/p99 %.0f/%.0f m" % tuple(np.percentile(h, [1, 99])))
