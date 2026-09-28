#!/usr/bin/env python3
"""High-resolution surface texture for synthetic injection tests.

Each real view is orthorectified at RES metres onto a given surface with
given refined poses. Views are combined per texel with weights that
strongly favour the finest ground sampling (weight = gsd^-POW), after
per-view brightness normalisation, so the texture keeps the detail of the
sharpest frame covering each place instead of averaging it away.

The texture is only a plausible brightness field for rendering synthetic
frames; residual misregistration between views does not matter for that
use.

Usage: make_texture.py DENSE_RESULT_NPZ OUT_NPZ [--res 5] [--pow 4]
  DENSE_RESULT_NPZ: reconstruct.py result (h in km, poses w/dC, exps)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import torch.nn.functional as Fn
from hdtm import recon
from hdtm.views import load_views

ap = argparse.ArgumentParser()
ap.add_argument("result"); ap.add_argument("out")
ap.add_argument("--res", type=float, default=5.0)
ap.add_argument("--pow", type=float, default=4.0)
a = ap.parse_args()
REGION = (-3.5, 1.5, 1.5, 6.5)
LOWPASS_M = 150.0   # detail above this scale comes from the 20 m mosaic
os.environ.pop("HDTM_SYNTH", None)
R = np.load(a.result)
nums = set(R["view_nums"].tolist())
views = [v for v in load_views(0.3, 20.0) if v["num"] in nums]
res = a.res / 1e3
scene = recon.Scene(views, *REGION, res)
w = torch.zeros(len(scene.exp_index), 3); dC = torch.zeros(len(scene.exp_index), 3)
for k, t in enumerate(R["exps"]):
    j = scene.exp_index.get(round(float(t), 1))
    if j is not None:
        w[j] = torch.tensor(R["w"][k]); dC[j] = torch.tensor(R["dC"][k])
h = torch.tensor(R["h"], dtype=torch.float32)
H = recon.resample(h, tuple(scene.X.shape))
P = torch.stack([scene.X, scene.Y, H], -1)
num = torch.zeros_like(scene.X); den = torch.zeros_like(scene.X)
with torch.no_grad():
    for i, v in enumerate(views):
        gx, gy, valid = scene.project(i, P, w, dC)
        if valid.sum() < 100:
            continue
        o = Fn.grid_sample(scene.images[i], torch.stack([gx, gy], -1)[None], mode="bicubic", align_corners=True)[0, 0]
        # keep only this view's fine detail: divide by its local mean (masked
        # Gaussian, sigma LOWPASS_M); low frequencies come from the 20 m mosaic
        s_px = LOWPASS_M / a.res
        vf = valid.float()[None, None]
        lp = recon.blur(torch.where(valid, o, torch.zeros_like(o))[None, None], s_px) / recon.blur(vf, s_px).clamp_min(1e-3)
        o = torch.where(valid, o / lp[0, 0].clamp_min(1e-6), torch.ones_like(o))
        C = scene.C0[i] + dC[scene.ev[i]]
        rng = (P - C).norm(dim=-1)
        gsd = rng * float(scene.cam[i, 9])                      # km per pixel, at the texel
        wt = torch.where(valid, gsd.clamp_min(1e-4) ** (-a.pow), torch.zeros_like(gsd))
        # feather the view edge so seams do not become sharp synthetic features
        edge = Fn.avg_pool2d(valid.float()[None, None], 41, 1, 20)[0, 0]
        wt = wt * edge ** 2
        num += wt * o; den += wt
detail = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(num))
import tifffile
A20 = torch.tensor(tifffile.imread(os.path.join(os.path.dirname(__file__), "..", "products/v1/brightness.tif")), dtype=torch.float32)
base = recon.blur(recon.resample(A20, tuple(scene.X.shape))[None, None], LOWPASS_M / a.res)[0, 0]
T = base / base.median() * detail
np.savez_compressed(a.out, tex=T.numpy().astype(np.float32), res_m=a.res, x0=float(scene.xs[0]), y0=float(scene.ys[0]))
print("texture", tuple(T.shape), "at %.1f m; brightness p1/p99 %.2f/%.2f" % (a.res, *np.percentile(T.numpy(), [1, 99])))
