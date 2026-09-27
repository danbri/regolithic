# Lab notebook

Dated entries, newest last. Each entry states what was done, what was
found, and what is still unverified.

## 2026-09-27: data access and inputs

Network access to the ESA PSA works from this environment (the earlier
failures in `logs/network-retest-confirmation.json` were specific to that
session).

Inputs (all recorded in `data/MANIFEST.jsonl` with URL, time, SHA-256;
`scripts/fetch_all.sh` re-creates `data/`):

- `raw` tier, DISR archive HP-SSA-DISR-2-3-EDR-RDR-V1.3:
  - `EXTRAS/PROCESSED_IMAGES/DISRSOFT_G_IMAGES_13NOV2005/PGM`: 606 images
    processed by E. Karkoschka (Nov 2005): decompression artefact
    reduction, flat/dark correction, PSF equalisation, resampling to a
    gnomonic projection, and I/F scaling. These are the primary inputs.
    They are single-frame calibrated observations, not terrain products.
  - `EXTRAS/IMAGE_ELEMENTS`: raw DCT-decoded frames, flats, darks, square
    root tables, for a possible later re-processing.
  - Users' Guide Appendix 3 (Karkoschka et al. 2007, PSS 55): per-exposure
    probe position (east/north of landing site, altitude) and attitude
    (azimuth, pitch, roll). Parsed to `derived/appx3_pointing.csv`
    (226 exposures; the on-surface exposure at MT 9008.5 is excluded).
    This is a navigation prior derived by Karkoschka partly from the
    images themselves; it contains no terrain heights.
- `reference` tier: `IPGP_Titan_Huygens_V1.0` (Daudon et al. 2020). Only
  the user guide PDF has been opened. IPGP_DTM.tif, IPGP_Ortho.tif and
  the eight G TIFFs remain unopened until geometry is frozen.
  From the guide: frames #414, 420, 450, 462, 471 (HRI), 541, 553, 601
  (MRI); SPICE navigation from ESA (doi:10.5270/esa-ssem3np); local
  tangent frame origin at the nadir of #450 (167.64370 E, -10.577749 N);
  18 m grid.

## 2026-09-27: camera model and conventions

G-image geometry (G-IMAGE_PROCESSING.TXT): gnomonic, optical axis at
nadir angle 70.3 / 31.3 / 14.5 deg (SLI / MRI / HRI), scale 0.0038397 /
0.0021817 / 0.0010821 rad/pixel, centre at the array middle. Implemented
as a pinhole in `hdtm/camera.py`.

Orientation of stored rows/columns:
- Row index increases with nadir angle. Checked by resampling HRI into
  the MRI frame of the same exposure (`scripts/check_orientation.py` and
  visual comparison of overlap crops): features line up only without a
  row flip. A high-pass NCC was inconclusive because MRI is much blurrier
  than HRI; the visual check was decisive.
- Column orientation and App. 3 attitude sign conventions were tested
  together (`scripts/check_pose.py`, log in `logs/check_pose_1-8km.txt`):
  41 MRI/HRI views between 1 and 8 km, ortho-projected onto a flat plane,
  band-passed NCC over 52 overlapping pairs. The convention read from the
  documentation (azimuth clockwise from north, positive pitch = looking
  further down, positive roll = right side down, no column flip) scores
  median NCC 0.87; the best alternative scores 0.55 and the rest are
  below 0.16.

Consequence: App. 3 poses plus the G-image camera model already give
sub-footprint consistency on a flat-ground assumption. Relief must be
recovered from the residual parallax, which is the next step.
