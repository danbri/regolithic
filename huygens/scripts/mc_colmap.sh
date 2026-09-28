#!/bin/sh
# Monte Carlo of the COLMAP-track + navigation-BA tilt profile over pose-error draws.
# Usage: mc_colmap.sh TRUTH_NPZ LABEL SEED...
t=$1; lab=$2; shift 2
for s in "$@"; do
  d=work/mcc_${lab}_s$s; mkdir -p $d
  [ -f $d/views.npz ] || python3 scripts/synth_views.py $t $d/views.npz --rot 0.5 --pos 0.05 --noise 0.01 --seed $s --mask-outside > /dev/null
  [ -d $d/sparse_priorpose_tri ] || HDTM_SYNTH=$d/views.npz python3 scripts/colmap_sfm.py $d > /dev/null 2>&1
  [ -f $d/priorba/profile.json ] || HDTM_SYNTH=$d/views.npz python3 scripts/colmap_prior_ba.py $d --tilt-scan=-120,-80,-40,0,40,80 > /dev/null 2>&1
  python3 - $d $lab $s <<'PY'
import json, sys, numpy as np
d, lab, s = sys.argv[1:]
p = json.load(open(d + "/priorba/profile.json")); g = np.array(p["grid"])
i = g[:, 2].argmin()
print("%s seed %s: quadratic min dE %.1f dN %.1f | grid argmin (%.0f, %.0f)" % (lab, s, p["min"][0], p["min"][1], g[i, 0], g[i, 1]))
PY
done
