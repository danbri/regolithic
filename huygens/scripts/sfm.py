#!/usr/bin/env python3
"""Structure from motion for the DISR descent images over the landing-site region.

Iterated guided-matching SfM:
  0. Start from a surface S (flat, or a given height grid) and poses
     (App. 3 priors, or those of a flat-ground run).
  1. Orthorectify each view onto S with the current poses.
  2. Reference mosaic R = per-view-normalised mean of the orthos.
  3. Tracks: at a grid of nodes, match each view's ortho window against R
     (band-passed phase correlation plus one refinement step). A match puts
     the node's feature at ortho location Q_v in view v; lifting Q_v onto S
     and projecting it through view v with the current poses gives the
     observed image coordinate. The ortho was made with the same surface
     and poses, so this is the actual image position of the feature.
  4. Bundle adjustment: 3D track points and per-exposure pose corrections
     (priors: App. 3), robust (Huber) reprojection error, L-BFGS, outlier
     rejection.
  5. New surface S = sigma-weighted Gaussian interpolation of the points;
     repeat from 1.
The tilt/pose degeneracy is handled by an optional profile over the
plane slope of the point cloud (--profile with --tilt-scan): each grid
value is held fixed while all else is optimised, and a quadratic fit to
the costs gives the minimum and its curvature.

Outputs OUTDIR/sfm.npz (points with sigma_h, views per point, gridded
DTM, poses), log.txt, optional profile.json.

Usage: sfm.py FLATRUN OUTDIR [options]   (FLATRUN supplies grid, views, poses)
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, cv2
import torch.nn.functional as Fn
from hdtm import recon
from hdtm.views import load_views

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--win", type=int, default=16)
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--min-resp", type=float, default=0.15)
ap.add_argument("--min-views", type=int, default=3)
ap.add_argument("--px-sigma", type=float, default=0.5)
ap.add_argument("--huber", type=float, default=2.0, help="Huber threshold in units of px-sigma")
ap.add_argument("--reject", type=float, default=3.0, help="px")
ap.add_argument("--sig-rot", type=float, default=1.0)
ap.add_argument("--sig-pos-h", type=float, default=0.2)
ap.add_argument("--sig-pos-v", type=float, default=0.05)
ap.add_argument("--dtm-sigma-m", type=float, default=150.0)
ap.add_argument("--iters", type=int, default=3, help="match / adjust / resurface cycles")
ap.add_argument("--init-h", default=None, help="npz h_m grid for the initial surface (diagnostics)")
ap.add_argument("--init-flat-poses", action="store_true", help="start at the flat run's poses instead of App. 3")
ap.add_argument("--tilt-scan", default="", help="comma list of plane slopes (m/km) for the profile in the last cycle")
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, "log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()


r = np.load(os.path.join(args.run, "result.npz"))
rargs = json.load(open(os.path.join(args.run, "args.json")))
view_nums = set(r["view_nums"].tolist())
views = [v for v in load_views(0.3, rargs.get("alt_max", 20.0)) if v["num"] in view_nums]
xs, ys, res = r["xs"], r["ys"], float(r["res"])
scene = recon.Scene(views, xs[0] - res / 2, xs[-1] + res / 2, ys[0] - res / 2, ys[-1] + res / 2, res)
nE, nV = len(scene.exp_index), len(views)
wF = torch.zeros(nE, 3); dCF = torch.zeros(nE, 3)
for k, t in enumerate(r["exps"]):
    j = scene.exp_index.get(round(float(t), 1))
    if j is not None:
        wF[j] = torch.tensor(r["w"][k]); dCF[j] = torch.tensor(r["dC"][k])
ny, nx = scene.X.shape
sr = math.radians(args.sig_rot)


def bandpass(a):
    return cv2.GaussianBlur(a, (0, 0), 0.8) - cv2.GaussianBlur(a, (0, 0), 4.0)


def make_orthos(S_km, w, dC):
    P = torch.stack([scene.X, scene.Y, torch.tensor(S_km, dtype=torch.float32)], -1)
    ortho, mask = [], []
    with torch.no_grad():
        for i in range(nV):
            gx, gy, valid = scene.project(i, P, w, dC)
            o = Fn.grid_sample(scene.images[i], torch.stack([gx, gy], -1)[None], mode="bilinear", align_corners=True)[0, 0]
            ortho.append(o.numpy().astype(np.float32)); mask.append(valid.numpy())
    return ortho, mask


def project_obs(X, w, dC, vidx):
    ev = scene.ev[vidx]
    B = recon.so3_exp(w[ev]) @ scene.B0[vidx]
    C = scene.C0[vidx] + dC[ev]
    d = torch.einsum("ni,nij->nj", X - C, B)
    cam = scene.cam[vidx]
    a, rr, u = cam[:, 0:3], cam[:, 3:6], cam[:, 6:9]
    s, Wd = cam[:, 9], cam[:, 10]
    z = (d * a).sum(1)
    return torch.stack([(Wd - 1) / 2 + (d * rr).sum(1) / z / s, 127.5 + (d * u).sum(1) / z / s], 1), z


def build_tracks(S_km, w, dC):
    ortho, mask = make_orthos(S_km, w, dC)
    acc = np.zeros((ny, nx), np.float32); cnt = np.zeros((ny, nx), np.float32)
    for o, m in zip(ortho, mask):
        if m.sum() > 100:
            acc += np.where(m, o / np.median(o[m]), 0); cnt += m
    R = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1).astype(np.float32)
    Rbp = bandpass(R)
    W = args.win
    hann = cv2.createHanningWindow((W, W), cv2.CV_32F)
    obs, nodes = [], []
    for r0 in range(0, ny - W + 1, args.stride):
        for c0 in range(0, nx - W + 1, args.stride):
            sl = (slice(r0, r0 + W), slice(c0, c0 + W))
            ref = Rbp[sl]
            if ref.std() < 1e-3:
                continue
            rows = []
            for i in range(nV):
                if mask[i][sl].mean() < 0.98:
                    continue
                p = bandpass(ortho[i][sl])
                if p.std() < 1e-4:
                    continue
                (sx, sy), resp = cv2.phaseCorrelate(ref, p, hann)
                if abs(sx) > W / 4 or abs(sy) > W / 4:
                    continue
                back = cv2.warpAffine(ortho[i], np.float32([[1, 0, sx], [0, 1, sy]]), (nx, ny),
                                      flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
                (ax, ay), resp = cv2.phaseCorrelate(ref, bandpass(back[sl]), hann)
                sx, sy = sx + ax, sy + ay
                if resp < args.min_resp or abs(sx) > W / 4 or abs(sy) > W / 4:
                    continue
                cc, cr = c0 + (W - 1) / 2 + sx, r0 + (W - 1) / 2 + sy
                rows.append((i, cc, cr))
            if len(rows) >= args.min_views:
                t = len(nodes)
                nodes.append((r0 + (W - 1) / 2, c0 + (W - 1) / 2))
                obs += [(t, i, cc, cr) for i, cc, cr in rows]
    obs = np.array(obs)
    # lift ortho locations onto the surface and project with the same poses -> image observations
    cc, cr = obs[:, 2].astype(np.float32), obs[:, 3].astype(np.float32)
    E = xs[0] + cc * res; N = ys[0] + cr * res
    H = cv2.remap(S_km.astype(np.float32), cc, cr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    vi = torch.tensor(obs[:, 1].astype(int))
    with torch.no_grad():
        uv, _ = project_obs(torch.tensor(np.c_[E, N, H], dtype=torch.float32), w, dC, vi)
    nd = np.array(nodes)
    nodeEN = np.c_[xs[0] + nd[:, 1] * res, ys[0] + nd[:, 0] * res]
    nodeH = cv2.remap(S_km.astype(np.float32), nd[:, 1].astype(np.float32), nd[:, 0].astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    return dict(tid=torch.tensor(obs[:, 0].astype(int)), vi=vi, uv=uv, nodeEN=nodeEN, nodeH=nodeH)


def run_ba(T, h_init, w_init, dC_init, target=None, Gpinv=None, verbose=True):
    tid, vi, uv_obs, nodeEN = T["tid"], T["vi"], T["uv"], T["nodeEN"]
    n = len(nodeEN)
    X = torch.nn.Parameter(torch.tensor(np.c_[nodeEN, h_init], dtype=torch.float32))
    w = torch.nn.Parameter(w_init.clone()); dC = torch.nn.Parameter(dC_init.clone())
    keep = torch.ones(len(tid), dtype=torch.bool)

    def prior_term():
        return 0.5 * (((w / sr) ** 2).sum() + ((dC[:, :2] / args.sig_pos_h) ** 2).sum() + ((dC[:, 2] / args.sig_pos_v) ** 2).sum())

    def robust(uv, obs_):
        rn = ((uv - obs_) / args.px_sigma).norm(dim=1)
        k = args.huber
        return torch.where(rn < k, 0.5 * rn ** 2, k * rn - 0.5 * k ** 2)

    def objective():
        uv, _ = project_obs(X[tid[keep]], w, dC, vi[keep])
        f = robust(uv, uv_obs[keep]).sum() + prior_term()
        if target is not None:
            f = f + 1e8 * ((Gpinv[1:] @ X[:, 2] - target) ** 2).sum()
        return f

    for rnd in range(4):
        opt = torch.optim.LBFGS([X, w, dC], lr=1, max_iter=500, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad(); f = objective(); f.backward(); return f
        opt.step(closure)
        with torch.no_grad():
            uv, _ = project_obs(X[tid], w, dC, vi)
            err = (uv - uv_obs).norm(dim=1)
            nk = err < args.reject
            nk &= torch.bincount(tid[nk], minlength=n)[tid] >= args.min_views
            if verbose:
                log("    BA round %d: reprojection RMS %.2f px on %d kept obs (-> %d); rot rms %.2f deg, dC rms %.0f m" % (
                    rnd, float(err[keep].pow(2).mean().sqrt()), int(keep.sum()), int(nk.sum()),
                    math.degrees(float(w.norm(dim=1).pow(2).mean().sqrt())), float(dC.norm(dim=1).pow(2).mean().sqrt()) * 1e3))
            if torch.equal(nk, keep):
                break
            keep = nk
    with torch.no_grad():   # comparable cost: all observations, robust loss capped at the rejection radius
        uv, _ = project_obs(X[tid], w, dC, vi)
        cap = 0.5 * (args.reject / args.px_sigma) ** 2
        cost = float(robust(uv, uv_obs).clamp(max=cap).sum() + prior_term())
    return cost, X.detach(), w.detach(), dC.detach(), keep


def point_sigmas(T, X, w, dC, keep):
    tid, vi = T["tid"], T["vi"]
    Xo = X[tid[keep]].clone().requires_grad_(True)
    uv, _ = project_obs(Xo, w, dC, vi[keep])
    J = torch.zeros(int(keep.sum()), 2, 3)
    for k in range(2):
        J[:, k, :] = torch.autograd.grad(uv[:, k].sum(), Xo, retain_graph=True)[0]
    Nm = torch.zeros(len(X), 3, 3)
    Nm.index_add_(0, tid[keep], torch.einsum("nki,nkj->nij", J, J) / args.px_sigma ** 2)
    nv = torch.bincount(tid[keep], minlength=len(X)).numpy()
    sig = np.full(len(X), np.nan)
    for t in np.where(nv >= args.min_views)[0]:
        try:
            c = np.linalg.inv(Nm[t].numpy().astype(np.float64))
            if c[2, 2] > 0:
                sig[t] = math.sqrt(c[2, 2])
        except np.linalg.LinAlgError:
            pass
    return sig, nv


def grid_surface(P, sig):
    EE, NN = scene.X.numpy(), scene.Y.numpy()
    sg = args.dtm_sigma_m / 1e3
    num = np.zeros_like(EE); den = np.zeros_like(EE)
    for (e, n_, h), s in zip(P, sig):
        k = np.exp(-0.5 * ((EE - e) ** 2 + (NN - n_) ** 2) / sg ** 2) / s ** 2
        num += k * h; den += k
    far = den < 1e-4 * den.max()
    S = num / np.maximum(den, 1e-30)
    fill = np.average(P[:, 2], weights=1 / sig ** 2)
    return np.where(far, fill, S), np.where(far, np.nan, S)


# ---- main loop
if args.init_h:
    S = np.nan_to_num(np.load(args.init_h)["h_m"]) / 1e3
else:
    S = np.zeros((ny, nx))
w_cur = wF.clone() if args.init_flat_poses else torch.zeros(nE, 3)
dC_cur = dCF.clone() if args.init_flat_poses else torch.zeros(nE, 3)
for it in range(args.iters):
    t0 = time.time()
    T = build_tracks(S, w_cur, dC_cur)
    log("cycle %d: tracks %d, observations %d (%.1f views/track), %.0f s" % (it, len(T["nodeEN"]), len(T["tid"]),
                                                                           len(T["tid"]) / len(T["nodeEN"]), time.time() - t0))
    last = it == args.iters - 1
    if last and args.profile and args.tilt_scan:
        nd = T["nodeEN"]
        Gd = np.c_[np.ones(len(nd)), nd[:, 0] - nd[:, 0].mean(), nd[:, 1] - nd[:, 1].mean()]
        Gp = np.linalg.pinv(Gd)
        base_slope = Gp[1:] @ T["nodeH"]
        grid = [float(v) / 1e3 for v in args.tilt_scan.split(",")]
        res_ = []
        for a_ in grid:
            for b_ in grid:
                tgt = np.array([a_, b_])
                h0 = T["nodeH"] + Gd[:, 1:] @ (tgt - base_slope)
                c_, X_, w_, dC_, k_ = run_ba(T, h0, w_cur, dC_cur, target=torch.tensor(tgt, dtype=torch.float32),
                                             Gpinv=torch.tensor(Gp, dtype=torch.float32), verbose=False)
                res_.append((a_ * 1e3, b_ * 1e3, c_))
                log("  profile slope (%+.0f, %+.0f) m/km: cost %.1f" % (a_ * 1e3, b_ * 1e3, c_))
        R_ = np.array(res_)
        D = np.c_[np.ones(len(R_)), R_[:, 0], R_[:, 1], R_[:, 0] ** 2, R_[:, 0] * R_[:, 1], R_[:, 1] ** 2]
        q, *_ = np.linalg.lstsq(D, R_[:, 2], rcond=None)
        Hq = np.array([[2 * q[3], q[4]], [q[4], 2 * q[5]]])
        mn = np.linalg.solve(Hq, -q[1:3])
        cov = np.linalg.inv(Hq)
        log("  profile minimum: dh/dE %.1f, dh/dN %.1f m/km; formal 1-sigma %.1f, %.1f m/km; quadratic fit rms %.1f" % (
            mn[0], mn[1], math.sqrt(abs(cov[0, 0])), math.sqrt(abs(cov[1, 1])), (R_[:, 2] - D @ q).std()))
        json.dump(dict(grid=R_.tolist(), min=mn.tolist(), cov=cov.tolist()), open(os.path.join(args.out, "profile.json"), "w"), indent=1)
        # final adjustment with the point-cloud plane held at the profile minimum
        tgt = mn / 1e3
        h0 = T["nodeH"] + Gd[:, 1:] @ (tgt - base_slope)
        cost, X, w_cur, dC_cur, keep = run_ba(T, h0, w_cur, dC_cur, target=torch.tensor(tgt, dtype=torch.float32),
                                              Gpinv=torch.tensor(Gp, dtype=torch.float32))
    else:
        cost, X, w_cur, dC_cur, keep = run_ba(T, T["nodeH"], w_cur, dC_cur)
    sig, nv = point_sigmas(T, X, w_cur, dC_cur, keep)
    ok = np.isfinite(sig)
    P = X.numpy()[ok]
    Gd = np.c_[np.ones(ok.sum()), P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean()]
    pl = np.linalg.lstsq(Gd * (1 / sig[ok])[:, None], P[:, 2] / sig[ok], rcond=None)[0] * 1e3
    log("  points %d, median views %d, median sigma_h %.0f m, height p5/p95 %.0f/%.0f m, weighted plane (%.1f, %.1f) m/km, cost %.1f" % (
        ok.sum(), np.median(nv[ok]), np.median(sig[ok]) * 1e3, *np.percentile((P[:, 2] - P[:, 2].mean()) * 1e3, [5, 95]),
        pl[1], pl[2], cost))
    S, S_nan = grid_surface(P, sig[ok])

h_m = (P[:, 2] - np.average(P[:, 2], weights=1 / sig[ok] ** 2)) * 1e3
dtm = (S_nan - np.nanmean(S_nan)) * 1e3
np.savez_compressed(os.path.join(args.out, "sfm.npz"), E=P[:, 0], N=P[:, 1], h_m=h_m, sigma_h_m=sig[ok] * 1e3,
                    nviews=nv[ok], dtm_m=dtm, xs=xs, ys=ys, w=w_cur.numpy(), dC=dC_cur.numpy(),
                    exps=np.array(sorted(scene.exp_index, key=scene.exp_index.get)))
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"))
