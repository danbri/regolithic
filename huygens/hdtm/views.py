"""Assemble views: image + imager + pose prior (App. 3 record at same MT)."""
import numpy as np
from .io import g_images, read_pgm, IMAGER_BY_WIDTH
from .pose import load_appx3


def load_views(alt_min=0.0, alt_max=1e9, imagers=("SLI", "MRI", "HRI")):
    recs = load_appx3()
    mts = np.array([r["mt_s"] for r in recs])
    out = []
    for d in g_images():
        i = int(np.argmin(abs(mts - d["mt"])))
        if abs(mts[i] - d["mt"]) > 0.05:
            continue
        r = recs[i]
        img = read_pgm(d["path"])
        if img.shape[0] != 256:
            continue
        im = IMAGER_BY_WIDTH[img.shape[1]]
        if im not in imagers or not (alt_min <= r["alt_km"] <= alt_max):
            continue
        out.append(dict(num=d["num"], mt=d["mt"], imager=im, img=img,
                        C=np.array([r["x_east_km"], r["y_north_km"], r["alt_km"]]),
                        az=r["azimuth_deg"], pitch=r["pitch_deg"], roll=r["roll_deg"]))
    return out
