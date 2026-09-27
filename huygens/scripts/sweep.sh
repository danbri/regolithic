#!/bin/sh
# 3-fold cross-validation over regularisation settings. Usage: sweep.sh OUTROOT "label|args" ...
root=$1; shift
for cfg in "$@"; do
  label=${cfg%%|*}; a=${cfg#*|}
  for f in 0 1 2; do
    d=$root/$label/f$f
    [ -f $d/heldout.json ] || python3 scripts/reconstruct.py $d --holdout 3 --fold $f $a > /dev/null 2>&1
  done
done
python3 - "$root" <<'PY'
import json,glob,sys,os,numpy as np
for d in sorted(glob.glob(sys.argv[1]+'/*/')):
    ev=[e for f in glob.glob(d+'f*/heldout.json') for e in json.load(open(f))]
    if ev: print("%-28s n=%3d median %.3f mean %.3f" % (os.path.basename(d.rstrip('/')), len(ev), np.median([e['ncc'] for e in ev]), np.mean([e['ncc'] for e in ev])))
PY
