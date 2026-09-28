#!/bin/sh
# DTM v3 chain on one synthetic draw (no dense stage, no tilt profile). Usage: chain_synth_v3.sh DIR TRUTH_NPZ SEED
set -e
d=$1; t=$2; s=$3
mkdir -p $d
[ -f $d/views.npz ] || python3 scripts/synth_views.py $t $d/views.npz --rot 0.5 --pos 0.05 --noise 0.005 --seed $s --mask-outside --texture work/texture_5m.npz --ss 3 --match-contrast > $d/synth.log
export HDTM_SYNTH=$d/views.npz
[ -d $d/sparse_priorpose_tri ] || python3 scripts/colmap_sfm.py $d > /dev/null 2>&1
[ -f $d/priorba/sfm.npz ] || python3 scripts/colmap_prior_ba.py $d > /dev/null 2>&1
python3 scripts/grid_points.py $d/priorba/sfm.npz $d/priorba/grid_k70.npz --sigma-m 70 --truth $t 2>/dev/null | tail -1 | sed "s|^|$(basename $d): |"
python3 scripts/eval_scales.py $d/priorba/grid_k70.npz $t $d/priorba/grid_k70.npz "$(basename $d)"
