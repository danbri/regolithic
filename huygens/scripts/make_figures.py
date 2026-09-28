#!/usr/bin/env python3
"""Regenerate the archival figures for stages that had none (figures/).

Each figure is written to figures/<name>.png. Inputs come from work/ runs;
figures whose inputs are missing are skipped with a message.
Usage: make_figures.py
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
F = "figures"
os.makedirs(F, exist_ok=True)
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
ext = [meta["x_centres_km"][0], meta["x_centres_km"][1], meta["y_centres_km"][0], meta["y_centres_km"][1]]
done = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(F, name), dpi=90); plt.close(fig); done.append(name)


def have(*p):
    ok = all(os.path.exists(x) for x in p)
    if not ok:
        print("skip: missing", [x for x in p if not os.path.exists(x)])
    return ok


# 1. parallax along/across by |e|
for tag, d in (("", "work/parallax_flat_all"), ("_detrend", "work/parallax_flat_all_detrend")):
    if have(d + "/parallax.npz"):
        M = np.load(d + "/parallax.npz")["meas"]
        dd, e = M[:, 3:5], M[:, 5:7]; en = np.linalg.norm(e, axis=1); u = e / en[:, None]
        al = (dd * u).sum(1) * 1e3; ac = (dd[:, 0] * -u[:, 1] + dd[:, 1] * u[:, 0]) * 1e3
        bins = [0.08, 0.15, 0.25, 0.4, 0.7, 2.0]
        rob = lambda x: 1.48 * np.median(abs(x - np.median(x)))
        c = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        A_ = [rob(al[(en >= bins[i]) & (en < bins[i + 1])]) for i in range(len(bins) - 1)]
        C_ = [rob(ac[(en >= bins[i]) & (en < bins[i + 1])]) for i in range(len(bins) - 1)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(en, al, s=3, alpha=.3, label="along e"); ax[0].scatter(en, ac, s=3, alpha=.3, label="across e")
        ax[0].set_xlabel("|e| (km shift per km height)"); ax[0].set_ylabel("shift (m)"); ax[0].set_ylim(-150, 150); ax[0].legend()
        ax[1].plot(c, A_, "o-", label="along"); ax[1].plot(c, C_, "s-", label="across")
        ax[1].set_xlabel("|e| bin centre"); ax[1].set_ylabel("robust RMS shift (m)"); ax[1].legend()
        fig.suptitle("Pair-wise parallax test, real data%s (scripts/parallax.py)" % (", per-pair affine removed" if tag else ""))
        save(fig, "08_parallax_along_across%s.png" % tag)

# 2. injection: truth vs recovered height maps (dense method)
rows = []
for n, t in (("detr_p0", "derived/h_ipgp_detrended.npz"), ("full_p1", "derived/h_ipgp_full.npz")):
    for cfg in ("s300", "s30"):
        p = "work/inject/%s/%s/result.npz" % (n, cfg)
        if have(p, t):
            rows.append((n, cfg, np.load(t)["h_m"], np.load(p)["h"] * 1e3))
if rows:
    fig, ax = plt.subplots(len(rows), 2, figsize=(9, 4 * len(rows)))
    for k, (n, cfg, T, Rh) in enumerate(rows):
        lo, hi = np.nanpercentile(T, [2, 98])
        ax[k, 0].imshow(T, origin="lower", extent=ext, cmap="terrain", vmin=lo, vmax=hi); ax[k, 0].set_title("%s truth (m)" % n)
        im = ax[k, 1].imshow(Rh - np.nanmean(Rh), origin="lower", extent=ext, cmap="terrain", vmin=lo - np.nanmean(T), vmax=hi - np.nanmean(T))
        ax[k, 1].set_title("dense recovery, %s (same colour scale)" % cfg); plt.colorbar(im, ax=ax[k, 1], fraction=.046)
    fig.suptitle("Injection test: dense height field recovers little relief and no tilt")
    save(fig, "09_injection_dense_recovery.png")

# 3. tilt profiles (SfM v2 real, COLMAP-track real)
profs = [(p, lab) for p, lab in (("work/sfm_real_v2/profile.json", "hand-built SfM v2, real"),
                                 ("work/colmap_real/priorba/profile.json", "COLMAP tracks + nav BA, real"),
                                 ("work/colmap_synthm_full/priorba/profile.json", "COLMAP tracks, synthetic tilted truth"),
                                 ("work/colmap_synthm_detr/priorba/profile.json", "COLMAP tracks, synthetic level truth")) if os.path.exists(p)]
if profs:
    fig, ax = plt.subplots(1, len(profs), figsize=(4.6 * len(profs), 4.2))
    ax = np.atleast_1d(ax)
    for a, (p, lab) in zip(ax, profs):
        g = np.array(json.load(open(p))["grid"])
        Es, Ns = sorted(set(g[:, 0])), sorted(set(g[:, 1]))
        Z = np.array([[g[(g[:, 0] == e) & (g[:, 1] == n), 2][0] for e in Es] for n in Ns])
        im = a.imshow(Z - Z.min(), origin="lower", extent=[Es[0], Es[-1], Ns[0], Ns[-1]], aspect="auto", cmap="viridis")
        a.plot(-76, 16, "r*", ms=14, label="IPGP plane"); a.plot(0, 0, "w+", ms=12, label="level")
        a.set_xlabel("dh/dE (m/km)"); a.set_ylabel("dh/dN (m/km)"); a.set_title(lab, fontsize=9); plt.colorbar(im, ax=a, fraction=.046)
    ax[0].legend(fontsize=8)
    fig.suptitle("Tilt profiles: cost above minimum (lower = preferred)")
    save(fig, "10_tilt_profiles.png")

# 4. Monte Carlo summaries
def read_mc(path, pat):
    out = []
    if os.path.exists(path):
        for l in open(path):
            if "dE" in l:
                x = l.split("dE")[1].split()
                out.append((float(x[0]), float(x[2])))
    return np.array(out)


mcs = [("hand-built SfM v2", read_mc("work/mc_tilt.txt", ""), read_mc("work/mc_level.txt", ""), 0.2),
       ("COLMAP tracks + nav BA", read_mc("work/mcc_tilt.txt", ""), read_mc("work/mcc_level.txt", ""), 1.9)]
extra = {"hand-built SfM v2": ([-62.6, 24.2], [-7.2])}
if any(len(m[1]) for m in mcs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (lab, T, L, real) in zip(ax, mcs):
        te = list(T[:, 0]) if len(T) else []
        le = list(L[:, 0]) if len(L) else []
        if lab in extra:
            te += extra[lab][0]; le += extra[lab][1]
        elif "COLMAP" in lab:
            te += [-92.4]; le += [32.0]
        a.scatter(te, np.ones(len(te)), c="C3", label="truth tilted (-75 to -80)")
        a.scatter(le, np.zeros(len(le)), c="C0", label="truth level")
        a.axvline(real, color="k", ls="--", label="real data")
        a.axvline(-78, color="C3", ls=":"); a.axvline(0, color="C0", ls=":")
        a.set_yticks([0, 1]); a.set_yticklabels(["level", "tilted"]); a.set_ylim(-.5, 1.5)
        a.set_xlabel("estimated dh/dE (m/km)"); a.set_title(lab); a.legend(fontsize=7, loc="lower left")
    fig.suptitle("Monte Carlo over navigation-error draws (0.5 deg, 50 m)")
    save(fig, "11_montecarlo_east_slope.png")

# 5. point maps: SfM v2 and COLMAP real, with IPGP outline
H = np.load("derived/ipgp_h_on_grid.npz")["h_m"] if os.path.exists("derived/ipgp_h_on_grid.npz") else None
pm = [(p, lab) for p, lab in (("work/sfm_real_v2/sfm.npz", "hand-built SfM v2 points"),
                              ("work/colmap_real/priorba/sfm.npz", "COLMAP tracks + nav BA points")) if os.path.exists(p)]
if pm:
    fig, ax = plt.subplots(1, len(pm) + (H is not None), figsize=(5 * (len(pm) + 1), 4.6))
    for a, (p, lab) in zip(ax, pm):
        S = np.load(p)
        sc = a.scatter(S["E"], S["N"], c=S["h_m"], s=6, cmap="terrain", vmin=-120, vmax=120)
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_aspect("equal"); a.set_title(lab + " (m)")
        plt.colorbar(sc, ax=a, fraction=.046)
    if H is not None:
        im = ax[-1].imshow(H - np.nanmean(H), origin="lower", extent=ext, cmap="terrain", vmin=-120, vmax=120)
        ax[-1].set_title("IPGP DTM on our grid (Daudon et al. 2020), m"); plt.colorbar(im, ax=ax[-1], fraction=.046)
    fig.suptitle("Real-data point clouds (heights relative to mean; frame: km E/N of landing site)")
    save(fig, "12_point_maps_real.png")

# 6. LightGlue matches on a real HRI pair
if have("work/colmap_real/disk_features.npz", "work/colmap_real/cameras.json"):
    import torch, kornia.feature as KF
    meta_c = json.load(open("work/colmap_real/cameras.json"))
    z = np.load("work/colmap_real/disk_features.npz")
    hri = [m for m in meta_c if m["imager"] == "HRI"]
    pair = None
    for i in range(len(hri)):
        for j in range(i + 1, len(hri)):
            if abs(hri[i]["mt"] - hri[j]["mt"]) > 20:
                pair = (hri[i], hri[j]); break
        if pair:
            break
    lg = KF.LightGlue("disk").eval()
    def pack(m):
        return dict(keypoints=torch.tensor(z[m["name"] + "|kp"])[None], descriptors=torch.tensor(z[m["name"] + "|desc"])[None],
                    image_size=torch.tensor([[m["width"], m["height"]]], dtype=torch.float32))
    best = None
    for a_ in hri:
        for b_ in hri:
            if a_["mt"] < b_["mt"] - 20:
                with torch.no_grad():
                    o = lg({"image0": pack(a_), "image1": pack(b_)})
                n = len(o["matches"][0])
                if best is None or n > best[0]:
                    best = (n, a_, b_, o["matches"][0].numpy())
    n, a_, b_, mt = best
    I0 = cv2.imread("work/colmap_real/images/" + a_["name"], 0); I1 = cv2.imread("work/colmap_real/images/" + b_["name"], 0)
    k0, k1 = z[a_["name"] + "|kp"] - 0.5, z[b_["name"] + "|kp"] - 0.5
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    canvas = np.hstack([I0, np.zeros((256, 10), np.uint8), I1])
    ax.imshow(canvas, cmap="gray")
    off = I0.shape[1] + 10
    for i0, i1 in mt:
        ax.plot([k0[i0, 0], k1[i1, 0] + off], [k0[i0, 1], k1[i1, 1]], lw=.6)
    ax.set_title("DISK + LightGlue: %d matches, %s (MT %.0f s) and %s (MT %.0f s)" % (n, a_["name"], a_["mt"], b_["name"], b_["mt"]), fontsize=9)
    ax.axis("off")
    save(fig, "13_lightglue_matches_real_hri.png")

# 7. contact sheet of COLMAP input images
imgs = sorted(glob.glob("work/colmap_real/images/*.png"))
if imgs:
    tiles = [cv2.imread(p, 0) for p in imgs]
    W = 176
    tiles = [np.pad(t, ((0, 0), (0, W - t.shape[1]))) for t in tiles]
    for t, p in zip(tiles, imgs):
        cv2.putText(t, os.path.basename(p)[:-4], (2, 12), 0, .35, 255, 1)
    cols = 10
    rows_ = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(F, "14_colmap_inputs_contact_sheet.png"), np.vstack(rows_)); done.append("14_colmap_inputs_contact_sheet.png")

# 8. synthetic view example (masked, tilted truth) next to the real frame
if have("work/colmap_synthm_full/views.npz"):
    from hdtm.views import load_views
    Sv = np.load("work/colmap_synthm_full/views.npz")
    os.environ.pop("HDTM_SYNTH", None)
    real = {v["num"]: v["img"] for v in load_views(0.3, 20.0)}
    ks = [k for k in Sv.files if int(k) in real][:40]
    pick = [k for k in ks if real[int(k)].shape[1] == 160][:3]
    fig, ax = plt.subplots(2, len(pick), figsize=(3 * len(pick), 8))
    for c_, k in enumerate(pick):
        ax[0, c_].imshow(real[int(k)][::-1], cmap="gray"); ax[0, c_].set_title("real #%s" % k); ax[0, c_].axis("off")
        ax[1, c_].imshow(Sv[k][::-1], cmap="gray"); ax[1, c_].set_title("synthetic #%s" % k); ax[1, c_].axis("off")
    fig.suptitle("Injection views: real frames vs synthetic renderings (tilted truth, outside masked)")
    save(fig, "15_synthetic_vs_real_views.png")

print("figures written:", done)
