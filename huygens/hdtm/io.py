"""Loading DISR G-images (Karkoschka 2005 calibrated, gnomonic) and metadata."""
import glob, os, re
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
G_DIR = os.path.join(ROOT, "data/raw/CASSINI-HUYGENS/DISR/HP-SSA-DISR-2-3-EDR-RDR-V1.3/"
                     "EXTRAS/PROCESSED_IMAGES/DISRSOFT_G_IMAGES_13NOV2005/PGM")
IMAGER_BY_WIDTH = {128: "SLI", 176: "MRI", 160: "HRI"}


def read_pgm(path):
    b = open(path, "rb").read()
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", b)
    w, h, mx = map(int, m.groups())
    dt = ">u2" if mx > 255 else "u1"
    return np.frombuffer(b[m.end():m.end() + w * h * np.dtype(dt).itemsize], dtype=dt).reshape(h, w).astype(np.float32)


def parse_name(path):
    """V_00414DCS_hhmmss_ffff.PGM -> (image number, mission time in s)."""
    m = re.match(r"V_(\d{5})\w*?_(\d\d)(\d\d)(\d\d)_(\d{4})", os.path.basename(path))
    num = int(m.group(1))
    mt = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1e4
    return num, mt


def g_images():
    out = []
    for p in sorted(glob.glob(os.path.join(G_DIR, "*.PGM"))):
        num, mt = parse_name(p)
        out.append(dict(path=p, num=num, mt=mt))
    return out
