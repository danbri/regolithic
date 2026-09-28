#!/bin/sh
# Monte Carlo of the SfM tilt estimate over pose-error draws (0.5 deg, 50 m, 1% noise).
# Usage: mc_sfm.sh TRUTH_NPZ LABEL SEED...
t=$1; lab=$2; shift 2
for s in "$@"; do
  d=work/inject/mc_${lab}_s$s; mkdir -p $d
  [ -f $d/views.npz ] || python3 scripts/synth_views.py $t $d/views.npz --rot 0.5 --pos 0.05 --noise 0.01 --seed $s > /dev/null
  [ -f $d/flat/result.npz ] || HDTM_SYNTH=$d/views.npz python3 scripts/reconstruct.py $d/flat --flat > /dev/null 2>&1
  [ -f $d/sfm/profile.json ] || HDTM_SYNTH=$d/views.npz python3 scripts/sfm.py $d/flat $d/sfm --iters 8 --px-sigma 2.5 --reject 6 --profile --tilt-scan=-120,-80,-40,0,40,80 > /dev/null 2>&1
  python3 -c "import json;p=json.load(open('$d/sfm/profile.json'));print('$lab seed $s', 'min dE %.1f dN %.1f' % tuple(p['min']))"
done
