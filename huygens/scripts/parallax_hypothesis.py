#!/usr/bin/env python3
"""Does a given height map predict the measured pair shifts?

For each phase-correlation measurement from parallax.py (window w, pair i-j,
measured shift d, parallax vector e), a height map predicts d_pred = h(w) e.
Reports the fraction of along-parallax shift variance explained, with and
without per-pair affine detrending applied identically to measured and
predicted shifts, and a bootstrap CI over view pairs.
Usage: parallax_hypothesis.py PARALLAX_DIR HEIGHT_NPZ[:key] [label]
"""
import sys, numpy as np
pdir, hspec = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else hspec
path, key = (hspec.split(":") + ["h_m"])[:2]
H = np.load(path)[key] / 1e3                    # km
P = np.load(pdir + "/parallax.npz")
M, rows, cols = P["meas"], P["row"], P["col"]
wid = M[:, 0].astype(int)
hw = H[np.clip(rows[wid].round().astype(int), 0, H.shape[0] - 1), np.clip(cols[wid].round().astype(int), 0, H.shape[1] - 1)]
ok = np.isfinite(hw)
M, hw, wid = M[ok], hw[ok], wid[ok]
hw = hw - hw.mean()
d, e = M[:, 3:5], M[:, 5:7]
u = e / np.linalg.norm(e, axis=1)[:, None]
along = (d * u).sum(1)
pred = hw * np.linalg.norm(e, axis=1)
pairs = [tuple(p) for p in M[:, 1:3].astype(int)]


def detrend(v):
    v = v.copy()
    for p in set(pairs):
        m = np.array([q == p for q in pairs])
        if m.sum() < 4:
            v[m] = np.nan; continue
        G = np.c_[np.ones(m.sum()), rows[wid[m]], cols[wid[m]]]
        c, *_ = np.linalg.lstsq(G, v[m], rcond=None)
        v[m] -= G @ c
    return v


def explained(a, p, idx):
    a, p = a[idx], p[idx]
    k = np.isfinite(a) & np.isfinite(p)
    a, p = a[k], p[k]
    return 1 - np.var(a - p) / np.var(a), np.corrcoef(a, p)[0, 1]


up = sorted(set(pairs))
pid = np.array([up.index(p) for p in pairs])
rng = np.random.default_rng(0)
for name, a, p in (("raw", along, pred), ("per-pair affine removed", detrend(along), detrend(pred))):
    full = np.arange(len(a))
    ev, r = explained(a, p, full)
    bs = []
    for _ in range(1000):
        pick = rng.integers(0, len(up), len(up))
        idx = np.concatenate([np.where(pid == q)[0] for q in pick])
        bs.append(explained(a, p, idx)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("%-22s %-24s n=%d  corr %.3f  variance explained %+.3f  95%% CI [%+.3f, %+.3f]" % (label, name, len(a), r, ev, lo, hi))

# best-fit amplitude scale k (measured_along ~ k * predicted), bootstrap over view pairs
for name, a, p in (("raw", along, pred), ("per-pair affine removed", detrend(along), detrend(pred))):
    k_ = np.isfinite(a) & np.isfinite(p)
    kk = lambda idx: (a[idx] * p[idx]).sum() / (p[idx] ** 2).sum()
    base = np.where(k_)[0]
    ks = []
    for _ in range(1000):
        pick = rng.integers(0, len(up), len(up))
        idx = np.concatenate([np.where((pid == q) & k_)[0] for q in pick])
        ks.append(kk(idx))
    rr = []
    for _ in range(1000):
        pick = rng.integers(0, len(up), len(up))
        idx = np.concatenate([np.where((pid == q) & k_)[0] for q in pick])
        rr.append(np.corrcoef(a[idx], p[idx])[0, 1])
    print("%-22s %-24s amplitude scale k %.2f  95%% CI [%.2f, %.2f]; corr 95%% CI [%.3f, %.3f]" % (
        label, name, kk(base), *np.percentile(ks, [2.5, 97.5]), *np.percentile(rr, [2.5, 97.5])))
