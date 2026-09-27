#!/usr/bin/env python3
"""Decide the PGM row/column orientation of G-images.

Images of one exposure share a viewpoint, so HRI resampled into the MRI
frame (and MRI into SLI) through the camera model must match in the overlap
for the correct orientation only. Reports normalised cross-correlation of
image gradients for each flip hypothesis.
"""
import collections, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from hdtm.io import g_images, read_pgm, IMAGER_BY_WIDTH
from hdtm import camera


def warp(src_img, src, dst, dst_shape, fx, fy):
    H, W = dst_shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    d = camera.pix_to_ray(dst, cc, rr, H, fx, fy)
    c, r, z = camera.ray_to_pix(src, d, src_img.shape[0], fx, fy)
    ok = (z > 0) & (c >= 2) & (c <= src_img.shape[1] - 3) & (r >= 2) & (r <= src_img.shape[0] - 3)
    out = cv2.remap(src_img, c.astype(np.float32), r.astype(np.float32), cv2.INTER_LINEAR)
    return out, ok


def hp(a):
    return a - cv2.GaussianBlur(a, (0, 0), 3)


def ncc(a, b, m):
    if m.sum() < 200:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


by = collections.defaultdict(dict)
for d in g_images():
    img = read_pgm(d["path"])
    if img.shape[0] != 256:
        continue
    by[round(d["mt"], 1)][IMAGER_BY_WIDTH[img.shape[1]]] = img
res = collections.defaultdict(list)
for t, g in sorted(by.items()):
    for src, dst in (("HRI", "MRI"), ("MRI", "SLI")):
        if src in g and dst in g:
            for fx in (False, True):
                for fy in (False, True):
                    w, m = warp(g[src], src, dst, g[dst].shape, fx, fy)
                    m &= cv2.erode(m.astype(np.uint8), np.ones((9, 9))) > 0
                    res[(src, dst, fx, fy)].append(ncc(hp(w), hp(g[dst]), m))
for k, v in res.items():
    v = np.array(v)
    print(k, "n=%d median NCC=%.3f" % (np.isfinite(v).sum(), np.nanmedian(v)))
