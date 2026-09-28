#!/usr/bin/env python3
"""Register the IPGP orthomosaic to our frozen brightness map (products/v1).

1. Coarse search: IPGP ortho, optionally mirrored, scaled 18 m -> our grid,
   rotated in 3 deg steps; normalised template matching of a central crop
   against our band-passed brightness map.
2. The best coarse solution is converted to an affine map from our grid
   pixel to IPGP pixel and refined with ECC (MOTION_AFFINE).
3. Reports the implied rotation, scale and handedness between the IPGP
   world frame (tfw) and our east/north frame, and writes
   derived/ipgp_to_ours.json.
"""
import json, math, os
import numpy as np, cv2, tifffile
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
R = "data/reference/Guest-Storage-Facility/IPGP_Titan_Huygens_V1.0/"
o0 = tifffile.imread(R + "IPGP_Ortho.tif").astype(np.float32)
tfw = [float(x) for x in open(R + "IPGP_Ortho.tfw").read().split()]
A = tifffile.imread("products/v1/brightness.tif").astype(np.float32)   # row 0 = south
meta = json.load(open("products/v1/FREEZE.json"))["meta"]
g = meta["grid_m"] / 1e3
x0, y0 = meta["x_centres_km"][0], meta["y_centres_km"][0]


def bp(a, s1=1.5, s2=10):
    return cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)


Ab = bp(A)
sc = 18.0 / (g * 1e3)
H0, W0 = o0.shape
best = None
for flip in (False, True):
    o = o0[:, ::-1] if flip else o0
    o = cv2.resize(o, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    m = (o > 0).astype(np.uint8)
    c = (o.shape[1] / 2, o.shape[0] / 2)
    for th in range(0, 360, 3):
        sz = int(max(o.shape) * 1.5)
        Mr = cv2.getRotationMatrix2D(c, th, 1.0)
        Mr[:, 2] += (sz / 2 - c[0], sz / 2 - c[1])
        orr = cv2.warpAffine(o, Mr, (sz, sz))
        mr = cv2.erode(cv2.warpAffine(m, Mr, (sz, sz), flags=cv2.INTER_NEAREST), np.ones((9, 9)))
        rr, cc = np.where(mr > 0)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        h, w = r1 - r0, c1 - c0
        r0 += int(h * .2); r1 -= int(h * .2); c0 += int(w * .2); c1 -= int(w * .2)
        t = bp(np.where(mr > 0, orr, orr[mr > 0].mean()))[r0:r1, c0:c1]
        if t.shape[0] >= A.shape[0] or t.shape[1] >= A.shape[1]:
            continue
        res = cv2.matchTemplate(Ab, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (mx, flip, th, loc, (r0, c0), Mr)
mx, flip, th, loc, (r0, c0), Mr = best
print("coarse: NCC %.3f flip=%s rot=%d" % (mx, flip, th))


def ours_to_ipgp_px(p):
    """Our pixel (col, row) -> IPGP pixel (col, row) through the coarse chain."""
    s = np.array([p[0] - loc[0] + c0, p[1] - loc[1] + r0, 1.0])
    Mi = cv2.invertAffineTransform(Mr)
    u = Mi @ s                      # scaled (and mirrored) IPGP pixel
    v = u / sc
    return np.array([W0 - 1 - v[0] if flip else v[0], v[1]])


P = np.array([[20, 20], [200, 30], [40, 220], [180, 200]], float)
Q = np.array([ours_to_ipgp_px(p) for p in P])
Wc, *_ = np.linalg.lstsq(np.c_[P, np.ones(4)], Q, rcond=None)
warp = np.ascontiguousarray(Wc.T, dtype=np.float32)  # 2x3: ipgp_px = warp @ [c, r, 1]
ob = bp(np.where(o0 > 0, o0, o0[o0 > 0].mean()), 1.5 / sc, 10 / sc).astype(np.float32)
mask = cv2.erode((o0 > 0).astype(np.uint8), np.ones((15, 15)))
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
ecc, warp = cv2.findTransformECC(Ab.astype(np.float32), ob, warp, cv2.MOTION_AFFINE, crit, mask)
print("ECC correlation %.3f" % ecc)
# check agreement after warp
ip = cv2.warpAffine(o0, warp, (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
vm = cv2.warpAffine(mask, warp, (A.shape[1], A.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
a, b = bp(A)[vm], bp(ip)[vm]
a, b = a - a.mean(), b - b.mean()
ncc = float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
print("band-passed NCC over overlap %.3f, overlap %.1f km^2" % (ncc, vm.sum() * g * g))

# world-frame relation: our (E, N) km = x0 + g*c, y0 + g*r ; IPGP world (X, Y) m = tfw
L = warp[:, :2]
T = warp[:, 2]
# IPGP world = S_ipgp @ ipgp_px + o_ipgp ; ipgp_px = L @ ours_px + T ; ours_px = (EN - [x0, y0]) / g
S_ip = np.array([[tfw[0], tfw[2]], [tfw[1], tfw[3]]]) / 1e3
o_ip = np.array([tfw[4], tfw[5]]) / 1e3
Mw = S_ip @ L / g                    # d(IPGP world km) / d(our EN km)
bw = S_ip @ (T - L @ np.array([x0, y0]) / g) + o_ip
U, sv, Vt = np.linalg.svd(Mw)
det = np.linalg.det(Mw)
Rm = U @ Vt
ang = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
print("IPGP_world = M @ ours_EN + b; M =\n", Mw.round(4), "\nb (km) =", bw.round(3))
print("singular values (scale) %.4f %.4f, det %.4f (%s), rotation part %.2f deg" % (sv[0], sv[1], det,
      "mirror: axes swapped or flipped" if det < 0 else "proper rotation", ang))
origin_ours = np.linalg.solve(Mw, -bw)
print("IPGP world origin (nadir of #450 per their guide) in our frame: E %.3f N %.3f km" % tuple(origin_ours))
json.dump(dict(coarse=dict(ncc=mx, flip=flip, rot_deg=th), ecc=ecc, bandpass_ncc=ncc,
               ours_px_to_ipgp_px=warp.tolist(), ipgp_world_from_ours_EN=dict(M=Mw.tolist(), b_km=bw.tolist()),
               scale=sv.tolist(), det=det, rotation_deg=ang, ipgp_origin_in_ours_km=origin_ours.tolist()),
          open("derived/ipgp_to_ours.json", "w"), indent=1)
cv2.imwrite("work/reg_check.png", np.hstack([
    cv2.normalize(bp(A), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[::-1],
    cv2.normalize(np.where(vm, bp(ip), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[::-1]]))
