#!/usr/bin/env python3
"""Modern SfM toolchain for the DISR descent images: DISK + LightGlue + COLMAP.

Stages (all outputs under OUTDIR):
  export   PNG images (8-bit, percentile stretch, rows flipped so that the
           camera frame is right-handed), cameras.json with exact PINHOLE
           intrinsics and App. 3 prior poses in local ENU metres.
  features DISK keypoints/descriptors (kornia), images upsampled by UPS.
  match    candidate pairs from prior-pose footprint overlap; LightGlue
           matching; COLMAP database; COLMAP geometric verification.
  map      (a) COLMAP incremental mapping with position priors
               (use_prior_position), then Sim3 alignment to the priors;
           (b) triangulation from the App. 3 prior poses followed by
               COLMAP bundle adjustment (intrinsics fixed).
  export points in ENU with per-point track length and reprojection error.

Camera model: Karkoschka's G-images are exact gnomonic projections
(hdtm/camera.py). With x_colmap = col + 0.5 and y_colmap = 255.5 - row,
the camera is PINHOLE with f = 1/SC, cx = W/2, cy = 128, and the
camera-to-world rotation has columns B r, -B u, B a (r, u, a from
camera.axes, B the sensor-head basis from pose.head_basis).

Usage: colmap_sfm.py OUTDIR [--alt-max 20] [--ups 2] [--max-kp 2048] [--stages export,features,match,map]
Set HDTM_SYNTH to run on synthetic views (injection tests).
"""
import argparse, json, math, os, shutil, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2, torch
import pycolmap
from hdtm import camera
from hdtm.pose import head_basis
from hdtm.views import load_views

REGION = (-3.5, 1.5, 1.5, 6.5)
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--alt-max", type=float, default=20.0)
ap.add_argument("--alt-min", type=float, default=0.3)
ap.add_argument("--min-footprint", type=float, default=0.05)
ap.add_argument("--ups", type=int, default=2)
ap.add_argument("--max-kp", type=int, default=2048)
ap.add_argument("--min-pair-overlap", type=float, default=0.1)
ap.add_argument("--sig-pos-h", type=float, default=200.0)
ap.add_argument("--sig-pos-v", type=float, default=50.0)
ap.add_argument("--stages", default="export,features,match,map")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
imgdir = os.path.join(args.out, "images")
logf = open(os.path.join(args.out, "log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()


def cam_to_world(v):
    B = head_basis(v["az"], v["pitch"], v["roll"])
    a, r, u = camera.axes(v["imager"])
    return np.stack([B @ r, -(B @ u), B @ a], axis=1)


def footprint(v, n=9):
    """Ground points (ENU km) of a grid of pixels on the plane z = 0, prior pose."""
    B = head_basis(v["az"], v["pitch"], v["roll"])
    W = camera.WIDTH[v["imager"]]
    cc, rr = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, 255, 2 * n))
    d = camera.pix_to_ray(v["imager"], cc.ravel(), rr.ravel()) @ B.T
    ok = d[:, 2] < -0.3
    t = v["C"][2] / -d[ok, 2]
    return v["C"][:2] + t[:, None] * d[ok, :2]


def in_region(g):
    x0, x1, y0, y1 = REGION
    return (g[:, 0] > x0) & (g[:, 0] < x1) & (g[:, 1] > y0) & (g[:, 1] < y1)


stages = args.stages.split(",")
meta_path = os.path.join(args.out, "cameras.json")

# ------------------------------------------------------------------ export
if "export" in stages:
    os.makedirs(imgdir, exist_ok=True)
    views = [v for v in load_views(args.alt_min, args.alt_max) if in_region(footprint(v)).mean() > args.min_footprint]
    meta = []
    for v in views:
        img = v["img"][::-1]                          # flip rows -> right-handed camera
        lo, hi = np.percentile(img, [0.5, 99.5])
        im8 = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        name = "%s_%05d.png" % (v["imager"], v["num"])
        cv2.imwrite(os.path.join(imgdir, name), im8)
        W = camera.WIDTH[v["imager"]]
        meta.append(dict(name=name, num=v["num"], imager=v["imager"], mt=v["mt"], width=W, height=256,
                         f=1.0 / camera.SC[v["imager"]], cx=W / 2, cy=128.0,
                         C_m=(v["C"] * 1e3).tolist(), R_cw=cam_to_world(v).tolist(),
                         footprint=footprint(v, 5).tolist()))
    json.dump(meta, open(meta_path, "w"), indent=1)
    log("export: %d views (%s)" % (len(meta), {k: sum(m["imager"] == k for m in meta) for k in ("SLI", "MRI", "HRI")}))
meta = json.load(open(meta_path))

# ------------------------------------------------------------------ features
feat_path = os.path.join(args.out, "disk_features.npz")
if "features" in stages:
    import kornia.feature as KF
    torch.set_num_threads(4)
    disk = KF.DISK.from_pretrained("depth").eval()
    feats = {}
    t0 = time.time()
    for m in meta:
        im = cv2.imread(os.path.join(imgdir, m["name"]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        s = args.ups
        big = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        H, W = big.shape
        Hp, Wp = (H + 15) // 16 * 16, (W + 15) // 16 * 16
        pad = np.zeros((Hp, Wp), np.float32); pad[:H, :W] = big
        x = torch.tensor(pad)[None, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            f = disk(x, n=args.max_kp, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
        kp = f.keypoints.numpy()
        ok = (kp[:, 0] < W - 1) & (kp[:, 1] < H - 1)
        kp = kp[ok]
        # DISK pixel (x, y) in upsampled image -> COLMAP coordinates in the original image
        kpc = (kp + 0.5) / s
        feats[m["name"]] = dict(kp=kpc.astype(np.float32), desc=f.descriptors.numpy()[ok].astype(np.float32),
                                score=f.detection_scores.numpy()[ok].astype(np.float32))
    np.savez_compressed(feat_path, **{k + "|" + j: v[j] for k, v in feats.items() for j in v})
    log("features: DISK x%d upsampling, median %d keypoints/image, %.0f s" % (
        args.ups, np.median([len(v["kp"]) for v in feats.values()]), time.time() - t0))


def load_feats():
    z = np.load(feat_path)
    out = {}
    for k in z.files:
        n, j = k.split("|")
        out.setdefault(n, {})[j] = z[k]
    return out


# ------------------------------------------------------------------ match
db_path = os.path.join(args.out, "database.db")
if "match" in stages:
    import kornia.feature as KF
    feats = load_feats()
    lg = KF.LightGlue("disk").eval()
    # candidate pairs: footprint overlap on the plane from prior poses
    from matplotlib.path import Path
    polys = []
    for m in meta:
        g = np.array(m["footprint"])
        hull = cv2.convexHull(g.astype(np.float32)).reshape(-1, 2)
        polys.append(Path(hull))
    probe = np.stack(np.meshgrid(np.linspace(REGION[0] - 3, REGION[1] + 3, 120), np.linspace(REGION[2] - 3, REGION[3] + 3, 120)), -1).reshape(-1, 2)
    inside = np.array([p.contains_points(probe) for p in polys])
    pairs = []
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            ov = (inside[i] & inside[j]).sum() / max(1, min(inside[i].sum(), inside[j].sum()))
            if ov >= args.min_pair_overlap:
                pairs.append((i, j))
    log("match: %d candidate pairs of %d possible" % (len(pairs), len(meta) * (len(meta) - 1) // 2))
    if os.path.exists(db_path):
        os.remove(db_path)
    db = pycolmap.Database.open(db_path)
    cam_ids, img_ids = {}, []
    for k, m in enumerate(meta):
        key = m["imager"]
        if key not in cam_ids:
            c = pycolmap.Camera(model="PINHOLE", width=m["width"], height=m["height"], params=[m["f"], m["f"], m["cx"], m["cy"]])
            c.has_prior_focal_length = True
            cam_ids[key] = db.write_camera(c)
        im = pycolmap.Image(name=m["name"], camera_id=cam_ids[key])
        iid = db.write_image(im)
        img_ids.append(iid)
        db.write_keypoints(iid, feats[m["name"]]["kp"].astype(np.float32))
        pp = pycolmap.PosePrior(position=np.array(m["C_m"]),
                                position_covariance=np.diag([args.sig_pos_h ** 2, args.sig_pos_h ** 2, args.sig_pos_v ** 2]),
                                coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN)
        pp.corr_data_id = pycolmap.data_t(sensor_id=pycolmap.sensor_t(pycolmap.SensorType.CAMERA, cam_ids[key]), id=iid) \
            if hasattr(pycolmap, "data_t") else pp.corr_data_id
        try:
            db.write_pose_prior(pp)
        except TypeError:
            db.write_pose_prior(iid, pp)
    t0 = time.time()
    nm = []
    lines = []
    for i, j in pairs:
        a, b = feats[meta[i]["name"]], feats[meta[j]["name"]]
        if len(a["kp"]) < 8 or len(b["kp"]) < 8:
            continue
        def pack(f, m):
            return dict(keypoints=torch.tensor(f["kp"])[None], descriptors=torch.tensor(f["desc"])[None],
                        image_size=torch.tensor([[m["width"], m["height"]]], dtype=torch.float32))
        with torch.no_grad():
            out = lg({"image0": pack(a, meta[i]), "image1": pack(b, meta[j])})
        mt = out["matches"][0].numpy().astype(np.uint32)
        if len(mt) >= 8:
            db.write_matches(img_ids[i], img_ids[j], mt)
            lines.append("%s %s" % (meta[i]["name"], meta[j]["name"]))
        nm.append(len(mt))
    db.close()
    pairs_txt = os.path.join(args.out, "pairs.txt")
    open(pairs_txt, "w").write("\n".join(lines) + "\n")
    log("LightGlue: %d pairs, median %d matches, %d pairs with >= 8, %.0f s" % (len(nm), np.median(nm), len(lines), time.time() - t0))
    opts = pycolmap.TwoViewGeometryOptions()
    pycolmap.verify_matches(db_path, pairs_txt, opts)
    db = pycolmap.Database.open(db_path)
    log("geometric verification: %d verified pairs, %d inlier matches" % (db.num_verified_image_pairs(), db.num_inlier_matches()))
    db.close()


# ------------------------------------------------------------------ map
def enu_points(rec):
    P, err, tl = [], [], []
    for pid, p in rec.points3D.items():
        P.append(p.xyz); err.append(p.error); tl.append(p.track.length())
    return np.array(P), np.array(err), np.array(tl)


def image_centres(rec):
    out = {}
    for iid, im in rec.images.items():
        if im.has_pose:
            out[im.name] = im.projection_center()
    return out


if "map" in stages:
    by_name = {m["name"]: m for m in meta}
    # (a) incremental mapping with position priors
    outa = os.path.join(args.out, "sparse_incremental"); shutil.rmtree(outa, ignore_errors=True); os.makedirs(outa)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.use_prior_position = True
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.min_model_size = 3
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db_path, imgdir, outa, options=opts)
    log("incremental mapping with position priors: %d models, %.0f s" % (len(recs), time.time() - t0))
    for k, rec in recs.items():
        P, err, tl = enu_points(rec)
        cen = image_centres(rec)
        dev = np.array([np.linalg.norm(cen[n] - np.array(by_name[n]["C_m"])) for n in cen])
        log("  model %d: %d images, %d points, mean track %.2f, mean reproj %.2f px, camera-centre deviation from prior median %.0f m" % (
            k, rec.num_reg_images(), rec.num_points3D(), rec.compute_mean_track_length(), rec.compute_mean_reprojection_error(),
            np.median(dev) if len(dev) else float("nan")))
    # (b) triangulation from prior poses + bundle adjustment
    outb = os.path.join(args.out, "sparse_priorpose"); shutil.rmtree(outb, ignore_errors=True); os.makedirs(outb)
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.Reconstruction()
    cams = db.read_all_cameras()
    for c in cams:
        rec.add_camera_with_trivial_rig(c)
    for im in db.read_all_images():
        m = by_name[im.name]
        Rcw = np.array(m["R_cw"]); C = np.array(m["C_m"])
        Rwc = Rcw.T
        cfw = pycolmap.Rigid3d(pycolmap.Rotation3d(Rwc), -Rwc @ C)
        im2 = pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id)
        im2.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in db.read_keypoints(im.image_id)[:, :2].astype(np.float64)])
        rec.add_image_with_trivial_frame(im2, cfw)
    db.close()
    t0 = time.time()
    rec = pycolmap.triangulate_points(rec, db_path, imgdir, outb, clear_points=True)
    outt = os.path.join(args.out, "sparse_priorpose_tri"); shutil.rmtree(outt, ignore_errors=True); os.makedirs(outt)
    rec.write(outt)   # tracks triangulated at the prior poses, before any adjustment
    log("prior-pose triangulation: %d points, mean track %.2f, mean reproj %.2f px, %.0f s" % (
        rec.num_points3D(), rec.compute_mean_track_length(), rec.compute_mean_reprojection_error(), time.time() - t0))
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = False; ba.refine_principal_point = False; ba.refine_extra_params = False
    pycolmap.bundle_adjustment(rec, ba)
    cen = image_centres(rec)
    dev = np.array([np.linalg.norm(cen[n] - np.array(by_name[n]["C_m"])) for n in cen])
    log("  after BA: %d points, mean reproj %.2f px, camera-centre shift from prior median %.0f m (BA gauge free; see alignment)" % (
        rec.num_points3D(), rec.compute_mean_reprojection_error(), np.median(dev)))
    rec.write(outb)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"))
