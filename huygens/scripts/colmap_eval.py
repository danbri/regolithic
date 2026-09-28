#!/usr/bin/env python3
"""Georeference a COLMAP model to the App. 3 camera centres and export points.

A 7-parameter similarity (rotation, translation, scale) is fitted from
the model's camera centres to the prior centres, with residuals weighted
by the prior sigmas (horizontal SIG_H, vertical SIG_V metres). Points are
transformed to local ENU, filtered (track length, reprojection error,
inside the region), and written as OUTDIR/<model>_points.npz in the
format of scripts/sfm.py (E, N km; h_m relative to the mean) so that
scripts/eval_points.py can compare them with a height grid.
Bootstrap over cameras gives the spread of the fitted plane slope.

Usage: colmap_eval.py TOOLCHAIN_OUTDIR [--model sparse_incremental/0]
"""
import argparse, json, math, os, sys
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", default="sparse_incremental/0")
ap.add_argument("--sig-h", type=float, default=200.0)
ap.add_argument("--sig-v", type=float, default=50.0)
ap.add_argument("--min-track", type=int, default=3)
ap.add_argument("--max-err", type=float, default=2.0)
ap.add_argument("--boot", type=int, default=200)
a = ap.parse_args()
REGION = (-3.5, 1.5, 1.5, 6.5)
meta = {m["name"]: m for m in json.load(open(os.path.join(a.out, "cameras.json")))}
rec = pycolmap.Reconstruction(os.path.join(a.out, a.model))
names, Cm, Cp = [], [], []
for iid, im in rec.images.items():
    if im.has_pose and im.name in meta:
        names.append(im.name); Cm.append(im.projection_center()); Cp.append(meta[im.name]["C_m"])
Cm, Cp = np.array(Cm), np.array(Cp)
sig = np.array([a.sig_h, a.sig_h, a.sig_v])


def fit(idx):
    def res(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        return ((np.exp(p[6]) * (Cm[idx] @ R.T) + p[3:6] - Cp[idx]) / sig).ravel()
    # init: unweighted Umeyama
    mu_m, mu_p = Cm[idx].mean(0), Cp[idx].mean(0)
    A, B = Cm[idx] - mu_m, Cp[idx] - mu_p
    U, S, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R0 = U @ D @ Vt
    s0 = (S * np.diag(D)).sum() / (A ** 2).sum()
    p0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), mu_p - s0 * R0 @ mu_m, math.log(s0)]
    sol = least_squares(res, p0)
    return Rotation.from_rotvec(sol.x[:3]).as_matrix(), sol.x[3:6], math.exp(sol.x[6]), sol.fun.reshape(-1, 3) * sig


R, t, s, r = fit(np.arange(len(Cm)))
print("model %s: %d cameras; similarity scale %.4f; centre residual RMS east %.0f north %.0f up %.0f m" % (
    a.model, len(Cm), s, *np.sqrt((r ** 2).mean(0))))
P, err, tl = [], [], []
for pid, p in rec.points3D.items():
    P.append(p.xyz); err.append(p.error); tl.append(p.track.length())
P, err, tl = np.array(P), np.array(err), np.array(tl)


def to_enu(R, t, s):
    return (s * (P @ R.T) + t) / 1e3


E = to_enu(R, t, s)
x0, x1, y0, y1 = REGION
ok = (tl >= a.min_track) & (err <= a.max_err) & (E[:, 0] > x0) & (E[:, 0] < x1) & (E[:, 1] > y0) & (E[:, 1] < y1)
# robust height outlier cut (MAD) after plane removal
G = np.c_[np.ones(ok.sum()), E[ok, 0] - E[ok, 0].mean(), E[ok, 1] - E[ok, 1].mean()]
c, *_ = np.linalg.lstsq(G, E[ok, 2], rcond=None)
rr = E[ok, 2] - G @ c
mad = 1.48 * np.median(abs(rr - np.median(rr)))
keep = np.where(ok)[0][abs(rr - np.median(rr)) < 5 * mad]
Ek = E[keep]
G = np.c_[np.ones(len(Ek)), Ek[:, 0] - Ek[:, 0].mean(), Ek[:, 1] - Ek[:, 1].mean()]
c, *_ = np.linalg.lstsq(G, Ek[:, 2], rcond=None)
print("points: %d total, %d kept in region (track >= %d, err <= %.1f px, 5-MAD cut); plane slope dE %.1f dN %.1f m/km; relief std after plane %.0f m" % (
    len(P), len(keep), a.min_track, a.max_err, c[1] * 1e3, c[2] * 1e3, 1e3 * (Ek[:, 2] - G @ c).std()))
# bootstrap over cameras: spread of the plane slope from the georeference
rng = np.random.default_rng(0)
sl = []
for _ in range(a.boot):
    idx = rng.integers(0, len(Cm), len(Cm))
    if len(np.unique(idx)) < 4:
        continue
    Rb, tb, sb, _ = fit(idx)
    Eb = to_enu(Rb, tb, sb)[keep]
    cb, *_ = np.linalg.lstsq(np.c_[np.ones(len(Eb)), Eb[:, 0] - Eb[:, 0].mean(), Eb[:, 1] - Eb[:, 1].mean()], Eb[:, 2], rcond=None)
    sl.append(cb[1:] * 1e3)
sl = np.array(sl)
print("bootstrap over cameras (%d): slope dE %.1f [%.1f, %.1f], dN %.1f [%.1f, %.1f] m/km (2.5-97.5%%)" % (
    len(sl), np.median(sl[:, 0]), *np.percentile(sl[:, 0], [2.5, 97.5]), np.median(sl[:, 1]), *np.percentile(sl[:, 1], [2.5, 97.5])))
h_m = (Ek[:, 2] - Ek[:, 2].mean()) * 1e3
tag = a.model.replace("/", "_")
np.savez_compressed(os.path.join(a.out, tag + "_points.npz"), E=Ek[:, 0], N=Ek[:, 1], h_m=h_m,
                    sigma_h_m=np.full(len(Ek), 1.0), nviews=tl[keep], err_px=err[keep],
                    sim3=np.r_[Rotation.from_matrix(R).as_rotvec(), t, s], slope_boot=sl)
os.makedirs(os.path.join(a.out, tag + "_eval"), exist_ok=True)
np.savez_compressed(os.path.join(a.out, tag + "_eval", "sfm.npz"), E=Ek[:, 0], N=Ek[:, 1], h_m=h_m,
                    sigma_h_m=np.full(len(Ek), 1.0), nviews=tl[keep])
