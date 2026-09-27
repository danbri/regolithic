"""DISR G-image camera model.

Karkoschka's G-images are gnomonic projections (G-IMAGE_PROCESSING.TXT):
optical axis at azimuth 0 and nadir angle NAc in the sensor-head frame,
central scale SC rad/pixel, centre (xc, yc) at the middle of the array.
This is an exact pinhole camera with focal length 1/SC pixels.

Sensor-head frame (right-handed): F = horizontal look direction
(azimuth 0), R = right, Dn = down (nadir). A ray with clockwise azimuth AZ
(positive to the right) and nadir angle NA is
    d = sin NA cos AZ F + sin NA sin AZ R + cos NA Dn.
"""
import numpy as np

NAC = {"SLI": 70.3, "MRI": 31.3, "HRI": 14.5}          # deg
SC = {"SLI": 0.0038397, "MRI": 0.0021817, "HRI": 0.0010821}  # rad/pixel
WIDTH = {"SLI": 128, "MRI": 176, "HRI": 160}

# Orientation of the stored PGM relative to Karkoschka's (x, y):
# FLIP_Y True means y = (H-1) - row.  Determined empirically in
# scripts/check_orientation.py.
FLIP_X = False
FLIP_Y = False


def axes(imager):
    """Return (a, r, u): optical axis, +x and +y unit vectors in (F, R, Dn) coords."""
    n = np.radians(NAC[imager])
    a = np.array([np.sin(n), 0.0, np.cos(n)])
    r = np.array([0.0, 1.0, 0.0])
    u = np.array([np.cos(n), 0.0, -np.sin(n)])   # direction of increasing NA
    return a, r, u


def pix_to_ray(imager, col, row, H=256, flip_x=None, flip_y=None):
    """Unit rays in the sensor-head frame (F, R, Dn) for pixel centres."""
    fx = FLIP_X if flip_x is None else flip_x
    fy = FLIP_Y if flip_y is None else flip_y
    W = WIDTH[imager]
    xc, yc = (W - 1) / 2, (H - 1) / 2
    x = (W - 1 - col) if fx else col
    y = (H - 1 - row) if fy else row
    a, r, u = axes(imager)
    s = SC[imager]
    d = a[None] + ((x - xc) * s)[..., None] * r + ((y - yc) * s)[..., None] * u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ray_to_pix(imager, d, H=256, flip_x=None, flip_y=None):
    fx = FLIP_X if flip_x is None else flip_x
    fy = FLIP_Y if flip_y is None else flip_y
    W = WIDTH[imager]
    xc, yc = (W - 1) / 2, (H - 1) / 2
    a, r, u = axes(imager)
    s = SC[imager]
    z = d @ a
    x = xc + (d @ r) / z / s
    y = yc + (d @ u) / z / s
    col = (W - 1 - x) if fx else x
    row = (H - 1 - y) if fy else y
    return col, row, z
