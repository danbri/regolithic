"""Probe pose priors from Karkoschka et al. (2007), DISR Users' Guide App. 3.

World frame: local east-north-up (km) with origin at the landing site
(App. 3 X east, Y north, altitude above the surface).

Sensor-head basis for a pose (azimuth A, pitch p, roll r), before
refinement:
  F0 = horizontal look direction at azimuth A (clockwise from north),
  R0 = horizontal right, Dn0 = down.
  pitch (positive = DISR pointing down) rotates F toward Dn about R;
  roll (positive = clockwise seen along the SLI view direction) rotates
  R toward Dn about F.
The sign conventions are switchable (CONV) so that scripts/check_pose.py
can test them against image overlap.
"""
import csv, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONV = dict(az=1, pitch=1, roll=1)


def load_appx3():
    rows = []
    with open(os.path.join(ROOT, "derived/appx3_pointing.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k[:4] != "has_" else v == "True") for k, v in r.items()})
    return rows


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def head_basis(az_deg, pitch_deg, roll_deg, conv=None):
    """3x3 matrix whose columns are F, R, Dn in ENU."""
    c = CONV if conv is None else conv
    A = np.radians(az_deg) * c["az"]
    F = np.array([np.sin(A), np.cos(A), 0.0])
    Dn = np.array([0.0, 0.0, -1.0])
    R = np.cross(Dn, F)                      # F x R = Dn
    # pitch: rotating about R by -p moves F toward Dn
    P = rot(R, -np.radians(pitch_deg) * c["pitch"])
    F, Dn = P @ F, P @ Dn
    # roll: rotating about F by +r moves R toward Dn
    Q = rot(F, np.radians(roll_deg) * c["roll"])
    R, Dn = Q @ R, Q @ Dn
    return np.stack([F, R, Dn], axis=1)
