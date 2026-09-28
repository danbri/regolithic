#!/usr/bin/env python3
"""Build the self-contained 3D viewer page for a terrain product.

Usage: build_viewer.py PRODUCT_DIR OUT_HTML
PRODUCT_DIR must contain dtm.npz (h_m, sigma_m, mask on the v1 grid) and
summary.json (numbers shown in the side panel). The brightness mosaic is
taken from products/v1/brightness.tif and the IPGP comparison from
derived/ipgp_h_on_grid.npz when present.
"""
import base64, io, json, os, sys
import numpy as np
import tifffile
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pdir, out = sys.argv[1], sys.argv[2]
D = np.load(os.path.join(pdir, "dtm.npz"))
summary = json.load(open(os.path.join(pdir, "summary.json")))
meta = json.load(open(os.path.join(ROOT, "products/v1/FREEZE.json")))["meta"]
h = D["h_all_m"].astype(float)
sig = D["sigma_m"].astype(float)
mask = D["mask"].astype(bool)
ny, nx = h.shape
A = tifffile.imread(os.path.join(ROOT, "products/v1/brightness.tif")).astype(float)


def b64_i16(a, scale):
    q = np.clip(np.round(np.nan_to_num(a, nan=-32768 / scale) * scale), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode()


def png_uri(img8):
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


lo, hi = np.percentile(A, [1, 99])
A8 = (np.clip((A - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[::-1]      # north up for the texture
ipgp = None
p = os.path.join(ROOT, "derived/ipgp_h_on_grid.npz")
if os.path.exists(p):
    I = np.load(p)["h_m"].astype(float)
    I = I - np.nanmean(I)
    ipgp = b64_i16(I, 10)
data = dict(nx=nx, ny=ny, cell_m=meta["grid_m"],
            x0_km=meta["x_centres_km"][0], y0_km=meta["y_centres_km"][0],
            h=b64_i16(h - np.nanmean(h[mask]), 10), sig=b64_i16(sig, 10),
            mask=base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode(),
            ipgp=ipgp, tex=png_uri(A8), summary=summary)
tpl = open(os.path.join(ROOT, "viewer/template.html")).read()
html = tpl.replace("/*__DATA__*/null", json.dumps(data))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, "w").write(html)
print("wrote", out, "%.0f KB" % (len(html) / 1024))
