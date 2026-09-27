#!/usr/bin/env python3
"""Run the joint height-field reconstruction over the landing-site region.

Usage: reconstruct.py OUTDIR [--holdout K] [--seed S] [--subset FRAC] [--baseline]

--holdout K   withhold every K-th exposure (all its images) for validation
--subset F    random fraction of training exposures (for ensemble spread)
--baseline    keep flat ground and prior poses (only brightness is fitted)
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, cv2
from matplotlib.path import Path
from hdtm import camera, recon
from hdtm.pose import head_basis
from hdtm.views import load_views

REGION = (-3.5, 1.5, 1.5, 6.5)          # km east / north of landing site
LEVELS = [(0.080, 2.0, 300), (0.040, 1.0, 300), (0.020, 0.0, 400)]  # grid res km, image blur px, iters
import os as _os
if _os.environ.get("HDTM_FAST"): LEVELS = [(0.080, 2.0, 300), (0.040, 1.0, 300)]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--holdout", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fold", type=int, default=-1)
ap.add_argument("--subset", type=float, default=1.0)
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--alt-max", type=float, default=20.0)
ap.add_argument("--imagers", default="SLI,MRI,HRI")
ap.add_argument("--levels", type=int, default=3)
ap.add_argument("--w-smooth", type=float, default=1.0)
ap.add_argument("--w-prior", type=float, default=50.0)
ap.add_argument("--sig-rot", type=float, default=1.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, "log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


def footprint_frac(v):
    B = head_basis(v["az"], v["pitch"], v["roll"])
    W = camera.WIDTH[v["imager"]]
    cc, rr = np.meshgrid(np.linspace(0, W - 1, 9), np.linspace(0, 255, 17))
    d = camera.pix_to_ray(v["imager"], cc.ravel(), rr.ravel()) @ B.T
    ok = d[:, 2] < -0.3
    t = v["C"][2] / -d[ok, 2]
    g = v["C"][:2] + t[:, None] * d[ok, :2]
    x0, x1, y0, y1 = REGION
    return float(((g[:, 0] > x0) & (g[:, 0] < x1) & (g[:, 1] > y0) & (g[:, 1] < y1)).mean()) if ok.any() else 0.0


views = [v for v in load_views(0.3, args.alt_max, tuple(args.imagers.split(","))) if footprint_frac(v) > 0.05]
exps = sorted({round(v["mt"], 1) for v in views})
held = set(exps[(args.holdout // 2 if args.fold < 0 else args.fold)::args.holdout]) if args.holdout else set()
rng = np.random.default_rng(args.seed)
train_exps = [t for t in exps if t not in held]
if args.subset < 1:
    train_exps = sorted(rng.choice(train_exps, int(round(len(train_exps) * args.subset)), replace=False).tolist())
train_set = set(train_exps)
log("views", len(views), "exposures", len(exps), "held-out exposures", len(held), "training exposures", len(train_set))
log("imagers", {k: sum(v["imager"] == k for v in views) for k in ("SLI", "MRI", "HRI")})

t0 = time.time()
model = None
for li, (res, isig, iters) in enumerate(LEVELS[:args.levels]):
    scene = recon.Scene(views, *REGION, res)
    train = [i for i, v in enumerate(views) if round(v["mt"], 1) in train_set]
    if model is None:
        model = recon.Model(scene)
        recon.init_albedo(scene, model, isig)
    else:
        model = recon.upsample_model(scene, model)
    log(f"level {li}: res {res*1e3:.0f} m grid {tuple(scene.X.shape)} image blur {isig}")
    recon.fit(scene, model, iters, img_sigma=isig, train=train, fix_geometry=args.baseline, log=log,
              w_smooth=args.w_smooth, w_prior=args.w_prior, sig_rot_deg=args.sig_rot)
log("fit time %.0f s" % (time.time() - t0))


# ---- validation on held-out exposures: band-passed NCC between observation and prediction
def bp(a):
    return cv2.GaussianBlur(a, (0, 0), 0.7) - cv2.GaussianBlur(a, (0, 0), 5.0)


def eval_views(idx):
    out = []
    with torch.no_grad():
        for i in idx:
            obs, pred, m = recon.view_residuals(scene, model, i, 0.0, with_pred=True)
            m = m.numpy()
            m = cv2.erode(m.astype(np.uint8), np.ones((11, 11))) > 0
            if m.sum() < 500:
                continue
            o, p = obs.numpy(), pred.numpy()
            o = np.where(m, o, o[m].mean()); p = np.where(m, p, p[m].mean())
            a, b = bp(o)[m], bp(p)[m]
            a -= a.mean(); b -= b.mean()
            out.append(dict(num=views[i]["num"], imager=views[i]["imager"], alt=float(views[i]["C"][2]),
                            ncc=float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())), n=int(m.sum())))
    return out


held_idx = [i for i, v in enumerate(views) if round(v["mt"], 1) in held]
ev0 = eval_views(held_idx)
if ev0:
    log("held-out views (prior pose)", len(ev0), "median NCC %.3f mean %.3f" % (np.median([e["ncc"] for e in ev0]), np.mean([e["ncc"] for e in ev0])))
if held_idx:
    recon.fit_views_only(scene, model, held_idx, w_prior=args.w_prior, sig_rot_deg=args.sig_rot)
ev = eval_views(held_idx)
if ev:
    log("held-out views", len(ev), "median NCC %.3f mean %.3f" % (np.median([e["ncc"] for e in ev]), np.mean([e["ncc"] for e in ev])))
json.dump(ev, open(os.path.join(args.out, "heldout.json"), "w"), indent=1)
json.dump(ev0, open(os.path.join(args.out, "heldout_priorpose.json"), "w"), indent=1)

np.savez_compressed(os.path.join(args.out, "result.npz"), h=model.h.detach().numpy(), A=model.A.detach().numpy(),
                    xs=scene.xs.numpy(), ys=scene.ys.numpy(), w=model.w.detach().numpy(), dC=model.dC.detach().numpy(),
                    gain=model.gain.detach().numpy(), exps=np.array(sorted(scene.exp_index, key=scene.exp_index.get)),
                    view_nums=np.array([v["num"] for v in views]), res=res)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"))
