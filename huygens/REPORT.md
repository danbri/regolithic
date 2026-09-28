# Huygens landing-site terrain from all DISR descent images: status report

Date: 2026-09-28. Everything in this report is reproducible from the
code at this commit and the archive files listed in
`data/MANIFEST.jsonl`. The dated record of every step is in `NOTEBOOK.md`.

## Question

Can the full set of public Huygens DISR descent images, processed with
modern multi-view methods, produce a better terrain model of the landing
site than the eight-image photogrammetric DTM of Daudon et al. (2020,
"IPGP DTM"), and can they resolve that model's tilt ambiguity?

## Data and method

- Inputs: Karkoschka's calibrated G-images (606 frames; exact gnomonic
  camera model, 0.03 deg stated accuracy) and his per-exposure positions
  and attitudes (DISR Users' Guide App. 3). All from the ESA PSA; SHA-256
  recorded.
- Region: 5 x 5 km containing the IPGP footprint. 40 views (7 SLI,
  14 MRI, 19 HRI) from 29 exposures below 20 km, against 8 in the IPGP DTM.
- Model: one joint differentiable fit of a height field, a shared surface
  brightness map, per-exposure pose corrections (priors: 1 deg attitude,
  200 m horizontal, 50 m vertical) and per-view photometric terms. Each
  grid cell is projected into every view and compared, coarse to fine
  (80, 40, 20 m). This is a surface-constrained differentiable-rendering
  method, the 2.5D counterpart of fitting surface-attached Gaussians.
  Free 3D Gaussians were not used; the images cannot constrain them in
  textureless areas.
- Blinding: the IPGP DTM and orthomosaic stayed unopened until our
  products were frozen and pushed (`products/v1`, commit bd2d108).

## Findings

1. Brightness map. The joint fit produces a consistent 20 m brightness
   mosaic of the region from all 40 views (`products/v1/overview.png`).
   It registers to the IPGP orthomosaic with band-passed NCC 0.70 over
   10.7 km^2.

2. Our height field is not a usable DTM. Cross-validation on held-out
   exposures chose strong smoothing. Against a proper control (flat ground
   with poses refined) no height field improved held-out prediction
   meaningfully (best +0.0006 NCC). Injection tests with synthetic views
   of known relief show why: the method recovers 10-60% of the relief
   amplitude (correlation 0.1-0.5 with truth) and almost none of a real
   4.5 deg tilt. The near-flat v1 surface reflects the method, not the
   terrain.

3. IPGP's window-scale relief is supported by the images. Pair-wise
   parallax measured directly between views (0.64 km windows, per-pair
   trends removed) correlates with IPGP heights at r = 0.15 [0.08, 0.23].
   The fitted amplitude, 0.38 [0.17, 0.54], lies within the range the same
   measurement returns when the truth is known exactly (0.27-0.46 in
   synthetic tests). The data are therefore consistent with IPGP's relief
   at its full amplitude, within wide limits. Our own heights do not
   correlate with the parallax (r = -0.03).

4. IPGP's regional tilt is not supported. Held fixed in our forward
   model, IPGP's best-fit plane (4.5 deg, falling to the east) predicts
   held-out views worse than level ground: -0.069 NCC [-0.125, -0.027],
   worse in 34 of 37 views. The penalty persists with the attitude prior
   loosened to 10 deg (-0.028 [-0.071, -0.005]). On synthetic views whose
   truth has that tilt, the same test prefers the tilt in 37 of 37 views,
   so the test has the power to detect it. A direct two-parameter plane
   fit, validated on synthetic data, gives an east-west slope of about
   0 to +14 m/km against IPGP's -75 m/km. The north-south slope is less
   certain (+24 to +63 m/km, prior-dependent). This conclusion rests on the
   DISR images together with the published descent altitudes and
   positions. It inherits any systematic error in them.

5. Frame. The IPGP world axes, as stored in its .tfw files, are rotated
   -109 deg from east/north in our frame (which reproduces the standard
   DISR mosaic orientation), with 1-5% scale differences. This is not yet
   explained; it should be checked against the figures of Daudon et al.
   (2020), whose full text could not be retrieved from this environment.

6. A new SfM pipeline (tie-point tracks, bundle adjustment, tilt
   profile; `scripts/sfm.py`, frozen real-data output in
   `products/sfm_v2`) was built and calibrated on synthetic data. On real
   data it gives a plane of +0.2 m/km east, +16 m/km north. A Monte Carlo
   over realistic pose errors shows its tilt estimate is not informative:
   six draws of a true -75 m/km slope give a mean of -7 m/km with a
   standard deviation of 39. Its relief correlates only 0.05-0.30 with
   synthetic truth. It does not change findings 2-4. The dense
   cross-validation test, in contrast, detects a true tilt on two
   independent pose-error draws (37/37 and 34/37 views).

7. Modern toolchain. DISK features + LightGlue matching + COLMAP
   (verification, incremental mapping with position priors) work on the
   real DISR frames: 37 of 40 views register automatically, 1,714 points,
   1.1 px reprojection. COLMAP's adjusters lack attitude priors, and on
   synthetic tests their geometry is unusable. COLMAP tracks with our
   navigation-constrained adjustment are usable: on 8 synthetic pose-error
   draws the east slope separates IPGP's tilt (mean -77 +- 33 m/km) from
   level ground (+15 +- 18). Real data give +1.9 m/km, consistent with
   level (likelihood ratio about 25 against IPGP's slope). Relief
   correlation with IPGP is 0.16, the best of the methods tried, but point
   heights remain too noisy for a DTM. Frozen output:
   `products/sfm_colmap_v1`.

## Novelty (per the project's A-D scale)

- (C) Additional observations and constraints: 40 views instead of 8, in
  one joint fit with navigation priors.
- (D) Materially new inferred geometry: limited to the regional-slope
  result (finding 4), which differs from the published DTM and comes with
  power checks. The height field itself is not new geometry of value.
- Closest methodological precedent: de Almeida et al., arXiv:2604.13235
  (2026), neural height fields for simulated lunar/Mars descent imagery.
  Not applied to Huygens. Karkoschka & Schröder (2016, Icarus 270) report
  topographic information from emission-angle comparisons across the DISR
  mosaic; their full text has not yet been read. Any novelty claim must
  wait until it has been.

## Limitations

- The height-field method under-recovers relief (finding 2). A better
  estimator is needed before this can be a DTM product.
- The slope result depends on App. 3 positions and attitudes and on the
  50 m altitude prior.
- Occlusion is ignored; views beyond 65 deg nadir angle are excluded.
- Only Karkoschka's 2005 G-images were used. The raw image elements
  (fetched) were not reprocessed.

## Next steps

00. Toolchain: add pairwise matching at native resolution between HRI
    frames (the IPGP stereo base), multi-scale LightGlue across the
    20:1 altitude range, and render synthetic frames at native
    resolution for calibration.

0. SfM: the limiting factor is tie-point precision (1-2 px from 320 m
   windows) against smooth pose-error shifts of similar size. Options are
   pairwise matching at native image resolution instead of orthos, and
   priors on pose correlation over time (the probe's swing is smooth
   between closely spaced exposures).

1. Fetch the ESA SPICE kernels used by IPGP. Compare their attitude,
   especially east-west tilt, with App. 3 for the eight IPGP frames. This
   tests the explanation for the slope difference.
2. Replace the per-texel height parameters with a multi-scale basis, and
   select smoothing by the injection tests (known truth) rather than only
   by held-out NCC. The held-out metric is blind to smooth relief.
3. Read Karkoschka & Schröder (2016) and the full Daudon et al. (2020).
4. Extend to the landing-site plain and to lower-altitude views with an
   occlusion-aware renderer.
