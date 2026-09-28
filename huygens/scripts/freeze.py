#!/usr/bin/env python3
"""Freeze blinded products before the reference DTM is opened.

Inputs: work/final_s300 (all exposures), work/ens_s300/seed* (80% subsets),
work/flat_all (flat ground, refined poses), their support maps.
Outputs in products/v1/: float32 TIFFs (height, ensemble std, brightness,
support layers), refined poses, and FREEZE.json with SHA-256 of every
product and the git commit of the code. Nothing here reads data/reference.
"""
import glob, hashlib, json, os, subprocess, sys, datetime
import numpy as np
import tifffile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
out = "products/v1"
os.makedirs(out, exist_ok=True)
final = np.load("work/final_s300/result.npz")
ens = [np.load(f) for f in sorted(glob.glob("work/ens_s300/seed*/result.npz"))]
H = np.stack([e["h"] for e in ens]) * 1e3
h = final["h"] * 1e3
sup = np.load("work/final_s300/support.npz")
res_m = float(final["res"]) * 1e3
xs, ys = final["xs"], final["ys"]
layers = {
    "height_m": h,
    "height_ensemble_std_m": H.std(0),
    "brightness": final["A"],
    "support_nviews": sup["nview"].astype(np.float32),
    "support_parallax_deg": sup["parallax_deg"].astype(np.float32),
    "support_texture": sup["texture"].astype(np.float32),
}
meta = dict(
    frame="local east/north (km) relative to the Huygens landing site as used by Karkoschka et al. 2007 "
          "(DISR Users' Guide App. 3); height relative to the mean over the grid",
    rows_are="north, increasing with row index (row 0 = southernmost)",
    x_centres_km=[float(xs[0]), float(xs[-1])], y_centres_km=[float(ys[0]), float(ys[-1])], grid_m=res_m,
    shape=list(h.shape))
files = {}
for k, a in layers.items():
    p = os.path.join(out, k + ".tif")
    tifffile.imwrite(p, a.astype(np.float32))
    files[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
p = os.path.join(out, "poses_refined.npz")
np.savez_compressed(p, exps=final["exps"], w_rad=final["w"], dC_km=final["dC"],
                    flat_w_rad=np.load("work/flat_all/result.npz")["w"], flat_dC_km=np.load("work/flat_all/result.npz")["dC"])
files[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "hdtm", "scripts"], capture_output=True, text=True).stdout.strip()
freeze = dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              code_commit=commit, code_dirty=bool(dirty), reference_opened=False, files=files, meta=meta,
              config="reconstruct.py --w-smooth 300 (levels 80/40/20 m); ensemble: --subset 0.8 seeds 1-5",
              summary=dict(height_p1_p99_m=[float(np.percentile(h, 1)), float(np.percentile(h, 99))],
                           ensemble_std_median_m=float(np.median(H.std(0))),
                           ensemble_std_p95_m=float(np.percentile(H.std(0), 95))))
json.dump(freeze, open(os.path.join(out, "FREEZE.json"), "w"), indent=1)
print(json.dumps(freeze["summary"]), "commit", commit, "dirty", bool(dirty))
