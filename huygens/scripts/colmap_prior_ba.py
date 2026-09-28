#!/usr/bin/env python3
"""Navigation-constrained bundle adjustment on COLMAP / LightGlue tracks.

Correspondences come from the modern toolchain (scripts/colmap_sfm.py:
DISK + LightGlue, COLMAP geometric verification and triangulation at the
App. 3 prior poses, model sparse_priorpose_tri). The geometry is solved
here with the full navigation priors that COLMAP's pose-prior adjuster
does not support: per-exposure rotation (SIG_ROT deg) and position
(SIG_H, SIG_V m) about App. 3, with the SLI/MRI/HRI images of one
exposure sharing one correction. Robust (Huber) reprojection error with
pixel sigma PX, L-BFGS, outlier rejection. Optional profile over the
point cloud's plane slope, as in scripts/sfm.py, to handle the
tilt/attitude degeneracy.

Writes OUTDIR/priorba/sfm.npz (E, N km; h_m; sigma_h_m; nviews) and log.

Usage: colmap_prior_ba.py TOOLCHAIN_OUTDIR [--px 2.5] [--tilt-scan -120,-80,-40,0,40,80]
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import pycolmap
from hdtm import recon
from hdtm.views import load_views

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", default="sparse_priorpose_tri")
ap.add_argument("--px", type=float, default=2.5)
ap.add_argument("--huber", type=float, default=2.0)
ap.add_argument("--reject", type=float, default=6.0)
ap.add_argument("--sig-rot", type=float, default=1.0)
ap.add_argument("--sig-h", type=float, default=0.2)
ap.add_argument("--sig-v", type=float, default=0.05)
ap.add_argument("--min-track", type=int, default=3)
ap.add_argument("--tilt-scan", default="")
ap.add_argument("--half", default="", help="SEED:K (K = 0 or 1): use only the exposures in random half K (split-half test)")
a = ap.parse_args()
REGION = (-3.5, 1.5, 1.5, 6.5)
od = os.path.join(a.out, "priorba" + ("_half" + a.half.replace(":", "_") if a.half else "")); os.makedirs(od, exist_ok=True)
logf = open(os.path.join(od, "log.txt"), "w")


def log(*x):
    s = " ".join(str(y) for y in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()


meta = json.load(open(os.path.join(a.out, "cameras.json")))
nums = [m["num"] for m in meta]
targs = json.load(open(os.path.join(a.out, "args.json")))
allv = {v["num"]: v for v in load_views(targs.get("alt_min", 0.3), targs.get("alt_max", 20.0))}
views = [allv[n] for n in nums]
scene = recon.Scene(views, *REGION, 0.08)
name_to_idx = {m["name"]: i for i, m in enumerate(meta)}
rec = pycolmap.Reconstruction(os.path.join(a.out, a.model))

# tracks -> observations in hdtm pixel convention (col = x - 0.5, row = 255.5 - y)
use_view = np.ones(len(meta), bool)
if a.half:
    hs, hk = (int(x) for x in a.half.split(":"))
    exps_ = sorted(scene.exp_index)
    side = np.random.default_rng(hs).permutation(len(exps_)) % 2
    exp_side = {e: side[i] for i, e in enumerate(exps_)}
    use_view = np.array([exp_side[round(v["mt"], 1)] == hk for v in views])
tid, vi, uv, X0 = [], [], [], []
for pid, p in rec.points3D.items():
    els = [el for el in p.track.elements if use_view[name_to_idx[rec.images[el.image_id].name]]]
    if len(els) < a.min_track:
        continue
    t = len(X0)
    X0.append(p.xyz / 1e3)
    for el in els:
        im = rec.images[el.image_id]
        xy = im.points2D[el.point2D_idx].xy
        tid.append(t); vi.append(name_to_idx[im.name]); uv.append((xy[0] - 0.5, 255.5 - xy[1]))
tid = torch.tensor(tid); vi = torch.tensor(vi); uv_obs = torch.tensor(np.array(uv), dtype=torch.float32)
X0 = np.array(X0)
log("tracks %d (length >= %d), observations %d" % (len(X0), a.min_track, len(tid)))
nE = len(scene.exp_index)
sr = math.radians(a.sig_rot)


def project(X, w, dC, v):
    ev = scene.ev[v]
    B = recon.so3_exp(w[ev]) @ scene.B0[v]
    C = scene.C0[v] + dC[ev]
    d = torch.einsum("ni,nij->nj", X - C, B)
    cam = scene.cam[v]
    z = (d * cam[:, 0:3]).sum(1)
    x = (cam[:, 10] - 1) / 2 + (d * cam[:, 3:6]).sum(1) / z / cam[:, 9]
    y = 127.5 + (d * cam[:, 6:9]).sum(1) / z / cam[:, 9]
    return torch.stack([x, y], 1)


def run(h_shift=None, target=None, Gp=None, verbose=True):
    X = torch.nn.Parameter(torch.tensor(X0 if h_shift is None else X0 + np.c_[np.zeros((len(X0), 2)), h_shift], dtype=torch.float32))
    w = torch.nn.Parameter(torch.zeros(nE, 3)); dC = torch.nn.Parameter(torch.zeros(nE, 3))
    keep = torch.ones(len(tid), dtype=torch.bool)

    def prior():
        return 0.5 * (((w / sr) ** 2).sum() + ((dC[:, :2] / a.sig_h) ** 2).sum() + ((dC[:, 2] / a.sig_v) ** 2).sum())

    def rob(u, o):
        rn = ((u - o) / a.px).norm(dim=1)
        return torch.where(rn < a.huber, 0.5 * rn ** 2, a.huber * rn - 0.5 * a.huber ** 2)

    def f():
        v = rob(project(X[tid[keep]], w, dC, vi[keep]), uv_obs[keep]).sum() + prior()
        if target is not None:
            v = v + 1e8 * ((Gp[1:] @ X[:, 2] - target) ** 2).sum()
        return v
    for rnd in range(4):
        opt = torch.optim.LBFGS([X, w, dC], max_iter=500, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def cl():
            opt.zero_grad(); v = f(); v.backward(); return v
        opt.step(cl)
        with torch.no_grad():
            err = (project(X[tid], w, dC, vi) - uv_obs).norm(dim=1)
            nk = err < a.reject
            nk &= torch.bincount(tid[nk], minlength=len(X0))[tid] >= a.min_track
            if verbose:
                log("  round %d: RMS %.2f px on %d obs (-> %d kept); rot rms %.2f deg; dC rms %.0f m" % (
                    rnd, float(err[keep].pow(2).mean().sqrt()), int(keep.sum()), int(nk.sum()),
                    math.degrees(float(w.norm(dim=1).pow(2).mean().sqrt())), 1e3 * float(dC.norm(dim=1).pow(2).mean().sqrt())))
            if torch.equal(nk, keep):
                break
            keep = nk
    with torch.no_grad():
        cap = 0.5 * (a.reject / a.px) ** 2
        cost = float(rob(project(X[tid], w, dC, vi), uv_obs).clamp(max=cap).sum() + prior())
    return cost, X.detach(), w.detach(), dC.detach(), keep


Gd = np.c_[np.ones(len(X0)), X0[:, 0] - X0[:, 0].mean(), X0[:, 1] - X0[:, 1].mean()]
Gp = np.linalg.pinv(Gd)
if a.tilt_scan:
    base = Gp[1:] @ X0[:, 2]
    grid = [float(v) / 1e3 for v in a.tilt_scan.split(",")]
    R_ = []
    for s1 in grid:
        for s2 in grid:
            tgt = np.array([s1, s2])
            c, *_ = run(h_shift=Gd[:, 1:] @ (tgt - base), target=torch.tensor(tgt, dtype=torch.float32),
                        Gp=torch.tensor(Gp, dtype=torch.float32), verbose=False)
            R_.append((s1 * 1e3, s2 * 1e3, c)); log("  profile (%+.0f, %+.0f) m/km: cost %.1f" % (s1 * 1e3, s2 * 1e3, c))
    R_ = np.array(R_)
    D = np.c_[np.ones(len(R_)), R_[:, 0], R_[:, 1], R_[:, 0] ** 2, R_[:, 0] * R_[:, 1], R_[:, 1] ** 2]
    q, *_ = np.linalg.lstsq(D, R_[:, 2], rcond=None)
    Hq = np.array([[2 * q[3], q[4]], [q[4], 2 * q[5]]])
    mn = np.linalg.solve(Hq, -q[1:3]); cov = np.linalg.inv(Hq)
    log("profile minimum dh/dE %.1f dh/dN %.1f m/km; formal 1-sigma %.1f %.1f; fit rms %.1f" % (
        mn[0], mn[1], math.sqrt(abs(cov[0, 0])), math.sqrt(abs(cov[1, 1])), (R_[:, 2] - D @ q).std()))
    json.dump(dict(grid=R_.tolist(), min=mn.tolist(), cov=cov.tolist()), open(os.path.join(od, "profile.json"), "w"))
    tgt = mn / 1e3
    cost, X, w, dC, keep = run(h_shift=Gd[:, 1:] @ (tgt - base), target=torch.tensor(tgt, dtype=torch.float32),
                               Gp=torch.tensor(Gp, dtype=torch.float32))
else:
    cost, X, w, dC, keep = run()
nv = torch.bincount(tid[keep], minlength=len(X0)).numpy()
# per-point height sigma (poses fixed at the solution) and max ray intersection angle
Xo = X[tid[keep]].clone().requires_grad_(True)
uvp = project(Xo, w, dC, vi[keep])
J = torch.zeros(int(keep.sum()), 2, 3)
for kk in range(2):
    J[:, kk, :] = torch.autograd.grad(uvp[:, kk].sum(), Xo, retain_graph=True)[0]
Nm = torch.zeros(len(X0), 3, 3)
Nm.index_add_(0, tid[keep], torch.einsum("nki,nkj->nij", J, J) / a.px ** 2)
sig = np.full(len(X0), np.nan)
Nn = Nm.numpy().astype(np.float64)
for t in np.where(nv >= a.min_track)[0]:
    try:
        cv_ = np.linalg.inv(Nn[t])
        if cv_[2, 2] > 0:
            sig[t] = math.sqrt(cv_[2, 2]) * 1e3
    except np.linalg.LinAlgError:
        pass
with torch.no_grad():
    ev_ = scene.ev[vi[keep]]
    Cc = (scene.C0[vi[keep]] + dC[ev_]).numpy()
    Xk = X[tid[keep]].numpy()
    rays = Cc - Xk
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
maxang = np.zeros(len(X0))
tk = tid[keep].numpy()
for t in np.unique(tk):
    R_t = rays[tk == t]
    if len(R_t) > 1:
        maxang[t] = np.degrees(np.arccos(np.clip((R_t @ R_t.T).min(), -1, 1)))
P = X.numpy()
x0, x1, y0, y1 = REGION
ok = (nv >= a.min_track) & np.isfinite(sig) & (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
P = P[ok]
G = np.c_[np.ones(len(P)), P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean()]
c, *_ = np.linalg.lstsq(G, P[:, 2], rcond=None)
rr = P[:, 2] - G @ c
mad = 1.48 * np.median(abs(rr - np.median(rr)))
k = abs(rr - np.median(rr)) < 5 * mad
P = P[k]
log("points in region %d; plane dE %.1f dN %.1f m/km; relief std after plane %.0f m" % (len(P), c[1] * 1e3, c[2] * 1e3, 1e3 * rr[k].std()))
np.savez_compressed(os.path.join(od, "sfm.npz"), E=P[:, 0], N=P[:, 1], h_m=(P[:, 2] - P[:, 2].mean()) * 1e3,
                    sigma_h_m=sig[ok][k], max_angle_deg=maxang[ok][k], nviews=nv[ok][k], w=w.numpy(), dC=dC.numpy())
log("median sigma_h %.0f m (px sigma %.1f); median max intersection angle %.1f deg" % (
    np.median(sig[ok][k]), a.px, np.median(maxang[ok][k])))
json.dump(vars(a), open(os.path.join(od, "args.json"), "w"))
