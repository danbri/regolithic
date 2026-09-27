#!/usr/bin/env python3
"""Per-texel support for a reconstruction run.

For each grid texel: number of training views that see it, the largest
angle between any two viewing rays (geometric parallax), and local texture
of the brightness field. Writes RUNDIR/support.npz and support.png.

Usage: support.py RUNDIR
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdtm import recon
from hdtm.views import load_views

run = sys.argv[1]
r = np.load(os.path.join(run, "result.npz"))
args = json.load(open(os.path.join(run, "args.json")))
view_nums = set(r["view_nums"].tolist())
views = [v for v in load_views(0.3, args.get("alt_max", 20.0)) if v["num"] in view_nums]
xs, ys = r["xs"], r["ys"]
res = float(r["res"])
scene = recon.Scene(views, xs[0] - res / 2, xs[-1] + res / 2, ys[0] - res / 2, ys[-1] + res / 2, res)
assert scene.X.shape == r["h"].shape, (scene.X.shape, r["h"].shape)
exps = r["exps"]
w = torch.zeros(len(scene.exp_index), 3)
dC = torch.zeros(len(scene.exp_index), 3)
for k, t in enumerate(exps):
    j = scene.exp_index.get(round(float(t), 1))
    if j is not None:
        w[j] = torch.tensor(r["w"][k]); dC[j] = torch.tensor(r["dC"][k])
P = torch.stack([scene.X, scene.Y, torch.tensor(r["h"])], -1)
dirs, masks = [], []
with torch.no_grad():
    for i in range(len(views)):
        gx, gy, valid = scene.project(i, P, w, dC)
        C = scene.C0[i] + dC[scene.ev[i]]
        d = C - P
        dirs.append((d / d.norm(dim=-1, keepdim=True)).numpy())
        masks.append(valid.numpy())
D = np.stack(dirs)          # V, ny, nx, 3
M = np.stack(masks)
nview = M.sum(0)
par = np.zeros(nview.shape)
for i in range(len(views)):
    for j in range(i + 1, len(views)):
        m = M[i] & M[j]
        if m.any():
            ang = np.degrees(np.arccos(np.clip((D[i] * D[j]).sum(-1), -1, 1)))
            par = np.where(m, np.maximum(par, ang), par)
A = r["A"].astype(np.float32)
bp = cv2.GaussianBlur(A, (0, 0), 1) - cv2.GaussianBlur(A, (0, 0), 4)
tex = np.sqrt(cv2.GaussianBlur(bp * bp, (0, 0), 4))
np.savez_compressed(os.path.join(run, "support.npz"), nview=nview, parallax_deg=par, texture=tex)
ext = [xs[0], xs[-1], ys[0], ys[-1]]
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, im, t in zip(ax, (nview, par, tex), ("views seeing texel", "max parallax (deg)", "local texture (band-pass RMS)")):
    h = a.imshow(im, origin="lower", extent=ext)
    a.set_title(t); plt.colorbar(h, ax=a, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(run, "support.png"), dpi=80)
print("views median %.0f, parallax median %.1f deg, texture median %.4f" % (np.median(nview), np.median(par), np.median(tex)))
