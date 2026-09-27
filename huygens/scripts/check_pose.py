#!/usr/bin/env python3
"""Test App. 3 attitude sign conventions and the G-image x orientation.

Views between ALT_MIN and ALT_MAX km are ortho-projected onto the plane
z = 0 (landing-site level) with each candidate convention. For every
pair of views with overlapping footprints, band-passed normalised cross
correlation is computed; the convention with the highest median wins.
"""
import itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from hdtm import camera
from hdtm.pose import head_basis
from hdtm.views import load_views

ALT_MIN, ALT_MAX, RES = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0, float(sys.argv[2]) if len(sys.argv) > 2 else 8.0, 0.02


def ortho(v, conv, fx, X, Y):
    B = head_basis(v["az"], v["pitch"], v["roll"], conv)
    d = np.stack([X - v["C"][0], Y - v["C"][1], -v["C"][2] * np.ones_like(X)], -1)
    dh = d @ B                      # (F, R, Dn) components
    dh /= np.linalg.norm(dh, axis=-1, keepdims=True)
    c, r, z = camera.ray_to_pix(v["imager"], dh, 256, flip_x=fx)
    img = v["img"]
    ok = (z > 0) & (c >= 3) & (c <= img.shape[1] - 4) & (r >= 3) & (r <= 252)
    o = cv2.remap(img, c.astype(np.float32), r.astype(np.float32), cv2.INTER_LINEAR)
    return o, ok


def band(a, m):
    a = np.where(m, a, a[m].mean() if m.any() else 0)
    return cv2.GaussianBlur(a, (0, 0), 1.0) - cv2.GaussianBlur(a, (0, 0), 6.0)


def ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


views = load_views(ALT_MIN, ALT_MAX, ("MRI", "HRI"))
print(len(views), "views")
ext = 1.5 * ALT_MAX
xs = np.arange(-ext, ext, RES)
X, Y = np.meshgrid(xs, xs)
results = {}
for az, pi, ro, fx in itertools.product((1, -1), (1, -1), (1, -1), (False, True)):
    conv = dict(az=az, pitch=pi, roll=ro)
    O = [ortho(v, conv, fx, X, Y) for v in views]
    scores = []
    for i in range(len(O)):
        for j in range(i + 1, len(O)):
            m = O[i][1] & O[j][1]
            m = cv2.erode(m.astype(np.uint8), np.ones((7, 7))) > 0
            if m.sum() < 400:
                continue
            scores.append(ncc(band(O[i][0], O[i][1]), band(O[j][0], O[j][1]), m))
    results[(az, pi, ro, fx)] = (np.median(scores) if scores else np.nan, len(scores))
    print(dict(az=az, pitch=pi, roll=ro, flip_x=fx), "median NCC %.3f over %d pairs" % results[(az, pi, ro, fx)], flush=True)
