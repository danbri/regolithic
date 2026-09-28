#!/bin/sh
# Injection test: synthetic views of a known surface through the same pipeline.
# Usage: injection.sh NAME HEIGHT_NPZ ROT_DEG POS_KM
set -e
n=$1; h=$2; r=$3; p=$4
d=work/inject/$n; mkdir -p $d
python3 scripts/synth_views.py $h $d/views.npz --rot $r --pos $p --noise 0.01 --seed 7
export HDTM_SYNTH=$d/views.npz
python3 scripts/reconstruct.py $d/flat --flat > /dev/null
python3 scripts/parallax.py $d/flat $d/parallax > /dev/null
python3 scripts/parallax.py $d/flat $d/parallax --detrend > /dev/null
python3 scripts/parallax_hypothesis.py $d/parallax $h truth > $d/parallax_truth.txt
python3 scripts/parallax_hypothesis.py $d/parallax_detrend $h truth >> $d/parallax_truth.txt
python3 scripts/reconstruct.py $d/s300 --w-smooth 300 > /dev/null
python3 scripts/reconstruct.py $d/s30 --w-smooth 30 > /dev/null
python3 - $d $h <<'PY'
import sys, numpy as np
d, hf = sys.argv[1], sys.argv[2]
t = np.load(hf)["h_m"]
for run in ("s300", "s30"):
    r = np.load(f"{d}/{run}/result.npz")["h"] * 1e3
    m = np.isfinite(t)
    ny, nx = t.shape; X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    G = np.c_[np.ones(m.sum()), X[m], Y[m]]
    ct, *_ = np.linalg.lstsq(G, t[m], rcond=None); cr, *_ = np.linalg.lstsq(G, r[m], rcond=None)
    rt, rr = t[m] - G @ ct, r[m] - G @ cr
    print("%s: truth plane slope (m/texel) %s recovered %s; relief std truth %.0f rec %.0f; corr after plane removal %.3f; raw corr %.3f" % (
        run, ct[1:].round(2), cr[1:].round(2), rt.std(), rr.std(), np.corrcoef(rt, rr)[0, 1], np.corrcoef(t[m], r[m])[0, 1]))
PY
