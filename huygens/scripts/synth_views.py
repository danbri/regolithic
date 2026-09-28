#!/usr/bin/env python3
"""Render synthetic DISR views of a known surface (injection test).

Each view's pixels are traced to the height field h (fixed-point ray/height
intersection) and take the brightness A there. Pixels whose ray leaves the
grid keep the real image value. Camera poses: App. 3 prior, optionally
perturbed per exposure (rotation sigma ROT deg, position sigma POS km) to
emulate navigation error; the reconstruction is then given the unperturbed
priors, as with real data. Gaussian noise NOISE (fraction of image median)
is added.

Usage: synth_views.py HEIGHT_NPZ OUT_NPZ [--rot DEG] [--pos KM] [--noise F] [--seed S]
The output maps view number -> image; set HDTM_SYNTH=OUT_NPZ to make
hdtm.views.load_views use it.
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import torch.nn.functional as Fn
import tifffile
from hdtm import camera
from hdtm.pose import head_basis, rot
from hdtm.views import load_views

ap = argparse.ArgumentParser()
ap.add_argument("hfile"); ap.add_argument("out")
ap.add_argument("--rot", type=float, default=0.0)
ap.add_argument("--pos", type=float, default=0.0)
ap.add_argument("--noise", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.environ.pop("HDTM_SYNTH", None)
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
g = meta["grid_m"] / 1e3
x0, y0 = meta["x_centres_km"][0], meta["y_centres_km"][0]
A = torch.tensor(tifffile.imread("products/v1/brightness.tif"))[None, None]
h = torch.tensor(np.nan_to_num(np.load(a.hfile)["h_m"]) / 1e3, dtype=torch.float32)[None, None]
ny, nx = h.shape[-2:]
rng = np.random.default_rng(a.seed)
views = load_views(0.3, 20.0)
pert = {}
out = {}
for v in views:
    t = round(v["mt"], 1)
    if t not in pert:
        axis = rng.normal(size=3)
        pert[t] = (rot(axis, math.radians(a.rot) * rng.normal()), rng.normal(size=3) * np.array([a.pos, a.pos, a.pos / 4]))
    Rp, dp = pert[t]
    B = Rp @ head_basis(v["az"], v["pitch"], v["roll"])
    C = v["C"] + dp
    W = camera.WIDTH[v["imager"]]
    cc, rr = np.meshgrid(np.arange(W, dtype=float), np.arange(256, dtype=float))
    d = camera.pix_to_ray(v["imager"], cc, rr) @ B.T            # world directions
    d = torch.tensor(d, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    down = d[..., 2] < -0.05
    t0 = Ct[2] / (-d[..., 2]).clamp_min(0.05)
    P = Ct + t0[..., None] * d
    for _ in range(8):
        gx = (P[..., 0] - x0) / (g * (nx - 1)) * 2 - 1
        gy = (P[..., 1] - y0) / (g * (ny - 1)) * 2 - 1
        hP = Fn.grid_sample(h, torch.stack([gx, gy], -1)[None], align_corners=True, padding_mode="border")[0, 0]
        t_ = (Ct[2] - hP) / (-d[..., 2]).clamp_min(0.05)
        P = Ct + t_[..., None] * d
    gx = (P[..., 0] - x0) / (g * (nx - 1)) * 2 - 1
    gy = (P[..., 1] - y0) / (g * (ny - 1)) * 2 - 1
    inside = down & (gx.abs() < 1) & (gy.abs() < 1)
    val = Fn.grid_sample(A, torch.stack([gx, gy], -1)[None], align_corners=True)[0, 0].numpy()
    img = v["img"].copy()
    med = float(np.median(img))
    ins = inside.numpy()
    # scale brightness to the real image's level inside the footprint
    if ins.sum() > 100:
        s = np.median(img[ins]) / np.median(val[ins])
        img[ins] = val[ins] * s
    img = img + rng.normal(size=img.shape).astype(np.float32) * a.noise * med
    out[str(v["num"])] = img.astype(np.float32)
np.savez_compressed(a.out, **out)
print("rendered", len(out), "views; pose perturbation rot %.2f deg pos %.3f km" % (a.rot, a.pos))
