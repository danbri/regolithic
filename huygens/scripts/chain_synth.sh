#!/bin/sh
# Full DTM chain on one synthetic scenario. Usage: chain_synth.sh DIR TRUTH_NPZ [synth_views options...]
set -e
d=$1; t=$2; shift 2
mkdir -p $d
[ -f $d/views.npz ] || python3 scripts/synth_views.py $t $d/views.npz "$@" > $d/synth.log
export HDTM_SYNTH=$d/views.npz
[ -d $d/sparse_priorpose_tri ] || python3 scripts/colmap_sfm.py $d > /dev/null 2>&1
[ -f $d/priorba/sfm.npz ] || python3 scripts/colmap_prior_ba.py $d --tilt-scan=-120,-80,-40,0,40,80 > /dev/null 2>&1
[ -f $d/priorba/grid.npz ] || python3 scripts/grid_points.py $d/priorba/sfm.npz $d/priorba/grid.npz > /dev/null
[ -f $d/dense_seed_s30/result.npz ] || python3 scripts/reconstruct.py $d/dense_seed_s30 --h-init $d/priorba/grid.npz --w-smooth 30 > /dev/null 2>&1
grep -E "LightGlue|verification" $d/log.txt | sed "s|^|$d: |"
grep -E "profile minimum|median sigma" $d/priorba/log.txt | sed "s|^|$d: |"
python3 scripts/eval_scales.py $d/dense_seed_s30/result.npz $t $d/priorba/grid.npz "$(basename $d)"
