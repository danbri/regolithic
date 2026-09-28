#!/usr/bin/env python3
"""Direct test for measurable terrain parallax.

All views are ortho-projected onto the flat plane z = 0 using the refined
poses of a flat-ground run (reconstruct.py --flat, no holdout). A surface
point at height h then appears in view v displaced by h * q_v, where
q_v = (P_xy - C_xy) / C_z. Between views i and j the local shift is
    d_ij = h * (q_j - q_i) = h * e_ij.
Pose errors and noise do not prefer the direction of e_ij; relief does.

Per window (WIN texels, stride WIN/2) and per view pair, the shift is
measured by phase correlation of band-passed ortho patches. Outputs:
  - RMS of shift components along and across e_ij (relief shows up only
    along), for pairs with |e_ij| above a threshold;
  - split-half reproducibility: pairs are split into two random halves,
    a height per window is solved from each half, and the two height maps
    are correlated across windows;
  - a sparse height map (window centres) with a formal error.

Usage: parallax.py FLATRUN OUTDIR
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, cv2
import torch.nn.functional as Fn
from hdtm import recon
from hdtm.views import load_views

WIN = 32
E_MIN = 0.08         # minimum |e| (km shift per km height) for a pair to count
run, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
r = np.load(os.path.join(run, "result.npz"))
args = json.load(open(os.path.join(run, "args.json")))
view_nums = set(r["view_nums"].tolist())
views = [v for v in load_views(0.3, args.get("alt_max", 20.0)) if v["num"] in view_nums]
xs, ys, res = r["xs"], r["ys"], float(r["res"])
scene = recon.Scene(views, xs[0] - res / 2, xs[-1] + res / 2, ys[0] - res / 2, ys[-1] + res / 2, res)
w = torch.zeros(len(scene.exp_index), 3); dC = torch.zeros(len(scene.exp_index), 3)
for k, t in enumerate(r["exps"]):
    j = scene.exp_index.get(round(float(t), 1))
    if j is not None:
        w[j] = torch.tensor(r["w"][k]); dC[j] = torch.tensor(r["dC"][k])

P = torch.stack([scene.X, scene.Y, torch.zeros_like(scene.X)], -1)
ortho, mask, Q = [], [], []
with torch.no_grad():
    for i in range(len(views)):
        gx, gy, valid = scene.project(i, P, w, dC)
        o = Fn.grid_sample(scene.images[i], torch.stack([gx, gy], -1)[None], mode="bilinear", align_corners=True)[0, 0]
        C = (scene.C0[i] + dC[scene.ev[i]]).numpy()
        q = np.stack([(scene.X.numpy() - C[0]) / C[2], (scene.Y.numpy() - C[1]) / C[2]], -1)
        ortho.append(o.numpy()); mask.append(valid.numpy()); Q.append(q)


def bandpass(a):
    return cv2.GaussianBlur(a, (0, 0), 0.8) - cv2.GaussianBlur(a, (0, 0), 4.0)


hann = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
ny, nx = scene.X.shape
meas = []   # (window id, i, j, dx_km, dy_km, ex, ey, response)
wins = []
for r0 in range(0, ny - WIN + 1, WIN // 2):
    for c0 in range(0, nx - WIN + 1, WIN // 2):
        wid = len(wins)
        wins.append((r0 + WIN / 2, c0 + WIN / 2))
        sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
        inside = [i for i in range(len(views)) if mask[i][sl].mean() > 0.95]
        patches = {i: bandpass(ortho[i][sl].astype(np.float32)) for i in inside}
        inside = [i for i in inside if patches[i].std() > 1e-3]
        for a in range(len(inside)):
            for b in range(a + 1, len(inside)):
                i, j = inside[a], inside[b]
                e = Q[j][r0 + WIN // 2, c0 + WIN // 2] - Q[i][r0 + WIN // 2, c0 + WIN // 2]
                if np.hypot(*e) < E_MIN:
                    continue
                (sx, sy), resp = cv2.phaseCorrelate(patches[i], patches[j], hann)
                if abs(sx) < WIN / 4 and abs(sy) < WIN / 4:
                    # one refinement step: undo the estimate on the full-size ortho and re-measure
                    Mw = np.float32([[1, 0, sx], [0, 1, sy]])
                    back = cv2.warpAffine(ortho[j].astype(np.float32), Mw, (nx, ny), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
                    (ax_, ay_), resp = cv2.phaseCorrelate(patches[i], bandpass(back[sl]), hann)
                    sx, sy = sx + ax_, sy + ay_
                if resp < 0.1 or abs(sx) > WIN / 4 or abs(sy) > WIN / 4:
                    continue
                meas.append((wid, i, j, sx * res, sy * res, e[0], e[1], resp))
M = np.array(meas)
DETREND = "--detrend" in sys.argv
if DETREND:
    # remove a per-pair affine shift field (residual pose error); keeps non-planar relief only
    W_ = np.array(wins)
    for p in {(int(a), int(b)) for a, b in M[:, 1:3]}:
        m = (M[:, 1] == p[0]) & (M[:, 2] == p[1])
        if m.sum() < 4:
            M[m, 3:5] = np.nan
            continue
        rc = W_[M[m, 0].astype(int)]
        G = np.c_[np.ones(m.sum()), rc[:, 0], rc[:, 1]]
        for c in (3, 4):
            coef, *_ = np.linalg.lstsq(G, M[m, c], rcond=None)
            M[m, c] -= G @ coef
    M = M[np.isfinite(M[:, 3])]
out = out + ("_detrend" if DETREND else "")
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "parallax.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")


say("views", len(views), "windows", len(wins), "pair measurements", len(M))
d = M[:, 3:5]; e = M[:, 5:7]
en = np.linalg.norm(e, axis=1)
u = e / en[:, None]
along = (d * u).sum(1)
across = d[:, 0] * -u[:, 1] + d[:, 1] * u[:, 0]
say("shift RMS along parallax %.1f m, across %.1f m (robust: MAD*1.48 along %.1f, across %.1f)" % (
    1e3 * np.sqrt((along ** 2).mean()), 1e3 * np.sqrt((across ** 2).mean()),
    1e3 * 1.48 * np.median(abs(along - np.median(along))), 1e3 * 1.48 * np.median(abs(across - np.median(across)))))


def solve(sel):
    """Least-squares height per window from measurements in sel."""
    h = np.full(len(wins), np.nan); s = np.full(len(wins), np.nan)
    wid = M[sel, 0].astype(int)
    for k in np.unique(wid):
        m = np.where(sel)[0][wid == k]
        if len(m) < 3:
            continue
        ee, dd = e[m], d[m]
        hh = (ee * dd).sum() / (ee * ee).sum()
        res_ = dd - hh * ee
        sig = np.sqrt((res_ ** 2).sum() / max(2 * len(m) - 1, 1))
        h[k] = hh; s[k] = sig / np.sqrt((ee * ee).sum())
    return h, s


rng = np.random.default_rng(0)
cors = []
for rep in range(20):
    # split by view pair identity so that the two halves share no pairs
    ex = {int(k): rng.random() < 0.5 for k in scene.ev.unique()}
    side = np.array([ex[int(scene.ev[int(i)])] for i in M[:, 1]])
    side_j = np.array([ex[int(scene.ev[int(j)])] for j in M[:, 2]])
    sel = side & side_j; sel2 = ~side & ~side_j
    h1, _ = solve(sel); h2, _ = solve(sel2)
    ok = np.isfinite(h1) & np.isfinite(h2)
    cors.append(np.corrcoef(h1[ok], h2[ok])[0, 1] if ok.sum() > 5 else np.nan)
say("split-half correlation of window heights, 20 random splits by exposure (no view shared): median %.3f [min %.3f, max %.3f], windows used ~%d" % (
    np.nanmedian(cors), np.nanmin(cors), np.nanmax(cors), ok.sum()))

# null test: rotate each shift by 90 deg (across component becomes 'along')
d_rot = np.stack([-d[:, 1], d[:, 0]], 1)
d_save = d.copy(); d[:] = d_rot
cn = []
for rep in range(20):
    ex = {int(k): rng.random() < 0.5 for k in scene.ev.unique()}
    side = np.array([ex[int(scene.ev[int(i)])] for i in M[:, 1]])
    side_j = np.array([ex[int(scene.ev[int(j)])] for j in M[:, 2]])
    sel = side & side_j; sel2 = ~side & ~side_j
    h1, _ = solve(sel); h2, _ = solve(sel2)
    ok = np.isfinite(h1) & np.isfinite(h2)
    cn.append(np.corrcoef(h1[ok], h2[ok])[0, 1] if ok.sum() > 5 else np.nan)
d[:] = d_save
say("null (shifts rotated 90 deg): split-half correlation median %.3f [min %.3f, max %.3f]" % (np.nanmedian(cn), np.nanmin(cn), np.nanmax(cn)))

h, s = solve(np.ones(len(M), bool))
W = np.array(wins)
np.savez_compressed(os.path.join(out, "parallax.npz"), h=h, sigma=s, row=W[:, 0], col=W[:, 1],
                    x=np.interp(W[:, 1], np.arange(nx), scene.xs.numpy()), y=np.interp(W[:, 0], np.arange(ny), scene.ys.numpy()), meas=M)
ok = np.isfinite(h)
say("window heights: %d windows, height RMS %.0f m, median formal sigma %.0f m" % (ok.sum(), 1e3 * np.nanstd(h), 1e3 * np.nanmedian(s)))
