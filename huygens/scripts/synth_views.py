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
ap.add_argument("--mask-outside", action="store_true", help="replace pixels outside the grid by a featureless constant (for whole-frame matchers)")
ap.add_argument("--texture", default=None, help="npz from make_texture.py (default: 20 m brightness map)")
ap.add_argument("--ss", type=int, default=1, help="supersampling per pixel axis (anti-aliasing)")
ap.add_argument("--match-contrast", action="store_true", help="scale rendered fine detail to the real frame's detail amplitude")
a = ap.parse_args()
os.environ.pop("HDTM_SYNTH", None)
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
g = meta["grid_m"] / 1e3
x0, y0 = meta["x_centres_km"][0], meta["y_centres_km"][0]
h = torch.tensor(np.nan_to_num(np.load(a.hfile)["h_m"]) / 1e3, dtype=torch.float32)[None, None]
ny, nx = h.shape[-2:]
if a.texture:     # high-resolution texture on its own grid (scripts/make_texture.py)
    Tz = np.load(a.texture)
    A = torch.tensor(Tz["tex"], dtype=torch.float32)[None, None]
    tres, tx0, ty0 = float(Tz["res_m"]) / 1e3, float(Tz["x0"]), float(Tz["y0"])
else:             # the 20 m brightness map on the DTM grid
    A = torch.tensor(tifffile.imread("products/v1/brightness.tif"))[None, None]
    tres, tx0, ty0 = g, x0, y0
tny, tnx = A.shape[-2:]
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
    Ct = torch.tensor(C, dtype=torch.float32)

    def trace(dc, dr):
        """Brightness and inside-mask for pixel centres offset by (dc, dr) pixels."""
        cc, rr = np.meshgrid(np.arange(W, dtype=float) + dc, np.arange(256, dtype=float) + dr)
        d = torch.tensor(camera.pix_to_ray(v["imager"], cc, rr) @ B.T, dtype=torch.float32)
        down = d[..., 2] < -0.05
        P = Ct + (Ct[2] / (-d[..., 2]).clamp_min(0.05))[..., None] * d
        for _ in range(8):
            gx = (P[..., 0] - x0) / (g * (nx - 1)) * 2 - 1
            gy = (P[..., 1] - y0) / (g * (ny - 1)) * 2 - 1
            hP = Fn.grid_sample(h, torch.stack([gx, gy], -1)[None], align_corners=True, padding_mode="border")[0, 0]
            P = Ct + ((Ct[2] - hP) / (-d[..., 2]).clamp_min(0.05))[..., None] * d
        gx = (P[..., 0] - x0) / (g * (nx - 1)) * 2 - 1
        gy = (P[..., 1] - y0) / (g * (ny - 1)) * 2 - 1
        inside = down & (gx.abs() < 1) & (gy.abs() < 1)
        tx = (P[..., 0] - tx0) / (tres * (tnx - 1)) * 2 - 1
        ty = (P[..., 1] - ty0) / (tres * (tny - 1)) * 2 - 1
        val = Fn.grid_sample(A, torch.stack([tx, ty], -1)[None], mode="bilinear", align_corners=True)[0, 0]
        return val, inside

    offs = [(0.0, 0.0)] if a.ss <= 1 else [((i + 0.5) / a.ss - 0.5, (j + 0.5) / a.ss - 0.5) for i in range(a.ss) for j in range(a.ss)]
    acc, inside = None, None
    for dc, dr in offs:
        vv, ii = trace(dc, dr)
        acc = vv if acc is None else acc + vv
        inside = ii if inside is None else inside & ii
    val = (acc / len(offs)).numpy()
    img = v["img"].copy()
    med = float(np.median(img))
    ins = inside.numpy()
    # scale brightness to the real image's level inside the footprint
    if ins.sum() > 100:
        s = np.median(img[ins]) / np.median(val[ins])
        rend = val * s
        if a.match_contrast:
            # scale the rendered fine detail (band 1-4 px) to the real frame's detail amplitude
            import cv2
            bp = lambda x: cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 4.0)
            core = cv2.erode(ins.astype(np.uint8), np.ones((9, 9))) > 0
            if core.sum() > 100:
                real_bp, rend_bp = bp(img.astype(np.float32)), bp(rend.astype(np.float32))
                k = real_bp[core].std() / max(rend_bp[core].std(), 1e-6)
                low = cv2.GaussianBlur(rend.astype(np.float32), (0, 0), 4.0)
                rend = low + (rend - low) * k
        img[ins] = rend[ins]
    if a.mask_outside:
        img[~ins] = np.median(img[ins]) if ins.sum() > 100 else med
    img = img + rng.normal(size=img.shape).astype(np.float32) * a.noise * med
    out[str(v["num"])] = img.astype(np.float32)
np.savez_compressed(a.out, **out)
print("rendered", len(out), "views; pose perturbation rot %.2f deg pos %.3f km" % (a.rot, a.pos))
