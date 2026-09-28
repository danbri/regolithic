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

## 2026-09-27: prior-art check (continued)

- de Almeida et al., arXiv:2604.13235 (Apr 2026), "Neural 3D
  Reconstruction of Planetary Surfaces from Descent-Phase Wide-Angle
  Imagery": explicit neural height field for descent imaging, tested on
  simulated lunar and Mars descents only. Not Huygens. Closest
  methodological precedent for this project's height-field formulation;
  any write-up must cite it.
- Karkoschka & Schröder 2016, Icarus 270, 307-325,
  doi:10.1016/j.icarus.2015.08.006, "The DISR imaging mosaic of Titan's
  surface and its dependence on emission angle": secondary summaries
  state that comparing images at different emission angles "yielded
  topographic information". Full text not yet read (publisher blocks
  access from here). Before claiming that using all images is new, this
  paper's topographic result (area, method, resolution) must be read
  and compared.
- No Huygens NeRF / Gaussian splatting / neural height-field work found.

## 2026-09-27: reconstruction method, first runs

Region: 5 x 5 km, x in [-3.5, 1.5], y in [1.5, 6.5] km relative to the
landing site; it contains the footprints of the eight IPGP frames
(located via App. 3 poses and frame numbers only).

Views: G-images with App. 3 pose, altitude 0.3-20 km, at least 5% of a
flat-ground footprint inside the region: 40 views (7 SLI, 14 MRI,
19 HRI) from 29 exposures.

Model (`hdtm/recon.py`): height field + shared brightness field +
per-exposure pose correction (rotation, position; Gaussian priors of 1 deg,
200 m horizontal, 50 m vertical) + per-view gain and linear offset.
Each grid texel is projected into each view and the sampled value is
compared to the view-blurred brightness (Charbonnier loss). Smoothness on
second differences of h, TV on brightness. Coarse to fine: 80, 40, 20 m.
Occlusion is ignored; rays beyond 65 deg nadir angle are excluded.

This is a surface-constrained differentiable-rendering reconstruction.
It is the 2.5D analogue of fitting surface-attached Gaussians; the
brightness grid plays the role of the splat colours. No free-floating
3D Gaussians are used, because the images constrain only the visible
surface and free Gaussians would be unconstrained in the textureless
plain.

First run (w_smooth 1, pose prior weight 50), held-out 5 exposures /
8 views: band-passed NCC median 0.785, mean 0.709, vs flat ground with
prior poses 0.753 / 0.607. Height field shows +-200-300 m blobs in the
textureless dark plain: regularisation too weak there. Brightness map
shows the known dendritic channels, the bright highland and the
shoreline-like boundary near y = 3.3 km.

Implementation note: running two torch processes at 4 threads each on
the 4-core machine slowed both by more than 10x (thread oversubscription).
Runs are now sequential. Per-view cropping to the footprint cut a full
three-level run from 22 min to 1.5 min.

## 2026-09-27: cross-validation of the height field

Protocol: 3 folds over the 29 exposures (every third exposure held out,
with all its SLI/MRI/HRI images). Fit on the rest. Score each held-out
view by band-passed NCC between the view resampled onto the recovered
surface and the model prediction.

First attempt scored held-out views at their App. 3 prior poses
(`logs/cv/sweep1_priorpose.txt`). No height-field setting beat flat
ground. The reason: a 0.5-1 deg attitude error at 10 km moves the
footprint by 90-170 m, far more than the terrain parallax under test.

Corrected protocol: with h and brightness frozen, refit only the
held-out exposure's pose (same priors) and the view's gain/offset,
coarse to fine, then score (`recon.fit_views_only`). Held-out NCC rises
from about 0.71 to 0.88 for all models, and differences between models
now reflect geometry.

Results (`logs/cv/sweep2_posefit.txt`, paired bootstrap against flat
ground in `logs/cv/sweep2_paired.txt`, 37 held-out views):

| config (w_smooth, w_prior) | mean diff vs flat | 95% CI | views better |
|---|---|---|---|
| 1, 50   | -0.019 | [-0.058, +0.008] | 18/37 |
| 10, 50  | +0.004 | [-0.014, +0.019] | 23/37 |
| 10, 5   | +0.011 | [-0.006, +0.026] | 26/37 |
| 100, 50 | +0.019 | [+0.009, +0.029] | 27/37 |

With enough smoothing, the recovered height field predicts unseen views
better than a flat surface does, with a confidence interval that excludes
zero. Weak smoothing overfits. The effect is small in NCC terms, which
is consistent with parallax of only a few pixels.

## 2026-09-28: controls: the CV gain came from pose refinement

Further configs (`logs/cv/sweep2_paired.txt`): w_smooth 300 and 1000
give the same +0.020 as 100. Using views up to 40 km
(`logs/cv/sweep3_alt40_paired.txt`) gives +0.013 over its own baseline.
It does not help, so the 20 km set is kept.

Missing control added: flat ground (h = 0) with pose refinement
(`--flat`). It scores +0.0193 [+0.010, +0.029] over the frozen-pose
baseline, the same as the height fields. Height fields against this
control (`logs/cv/sweep2_vs_flatpose.txt`):

| config | mean diff vs flat + pose refit | 95% CI |
|---|---|---|
| s10  | -0.015 | [-0.032, -0.003] |
| s100 | -0.0005 | [-0.0035, +0.0018] |
| s300 | +0.0006 | [-0.0002, +0.0014] |
| s1000 | +0.0006 | [+0.0003, +0.0010] |
| high-pass loss, s10 / s100 | -0.10 / -0.09 | both exclude 0 |

Conclusion: under this validation, no recovered height field predicts
unseen views better than flat ground by a meaningful margin. The earlier
+0.02 was pose refinement. Caveat: refitting each held-out view's pose
(6 dof) can absorb smooth, long-wavelength relief, so this test is blind
to it.

## 2026-09-28: direct parallax test

`scripts/parallax.py`: all 40 views ortho-projected on z = 0 with
refined flat-ground poses (`work/flat_all`), shifts between view pairs
measured by phase correlation (with one refinement step; validated on
synthetic shifts) in 640 m windows. Relief of height h must produce a
shift h * e_ij along a known direction per pair. Noise and most pose
errors do not prefer that direction.

- All pairs: robust shift RMS along e 22.3 m, across 19.6 m. Little
  excess.
- By parallax strength |e| (`logs/parallax/`): along/across ratio about
  1.0-1.2 for |e| < 0.7. For |e| >= 0.7 (mostly HRI-SLI and MRI-SLI
  pairs with the SLI at a median 2.7 km altitude) ratio 1.40, 95% CI
  [1.16, 1.72].
- After removing a per-pair affine shift field (which would absorb
  residual pose error, and also planar relief), the top-bin ratio is
  1.15 [0.97, 1.44], not significant.
- Split-half reproducibility of window heights, splitting by exposure so
  that no view is shared, is not distinguishable from a null with shifts
  rotated 90 deg (0.36 vs 0.34; detrended 0.41 vs 0.25; only about 24
  windows per split).

Reading: the only significant parallax signal is long-wavelength and
appears in pairs with low oblique SLI views, where a pitch error moves
the footprint along the same direction as parallax. At these scales,
relief and attitude error are not separable with the present priors.
This is the tilt ambiguity noted by Daudon et al. (2020), extended
beyond a single plane. Window-scale (about 0.3-1 km) relief is not
detected; the along-excess at |e| >= 0.7 bounds it at roughly 15-30 m
RMS for those windows.

Plan: freeze the geometry (config s300, all exposures, plus an 80%
exposure-subset ensemble), record hashes, then open the IPGP reference
and test its heights as a hypothesis in the same forward model and
parallax test.

## 2026-09-28: geometry frozen; reference opened

Frozen products: `products/v1` (commit bd2d108, code commit 8e8765b,
`products/v1/FREEZE.json` with SHA-256 of each file). Config s300,
all 29 exposures; 5-member ensemble on random 80% exposure subsets.
Height range p1/p99 -17/+12 m; ensemble std median 1.1 m. The spread
reflects the strong smoothing, not accuracy. The IPGP files were opened
only after this commit was pushed.

### Registration (`scripts/register_ipgp.py`, `logs/reference/ipgp_to_ours.json`)
IPGP_Ortho was matched to our brightness map by rotation/mirror search
plus ECC affine refinement. Band-passed NCC 0.70 over 10.7 km^2. The
IPGP world axes (from the .tfw files) are rotated -108.8 deg (proper
rotation) from our east/north frame, with scale factors 1.012 and 1.053.
The IPGP guide describes its frame as a local tangent plane with
north/east axes, so either the stored axes are not north-aligned, or the
SPICE attitude used by IPGP differs in azimuth from Karkoschka's. Our
frame reproduces the standard DISR mosaic orientation (channels NW of the
dark plain). This is unresolved and should be checked against Daudon
et al. (2020) figures before being reported as a finding.

### Heights (`scripts/ipgp_on_grid.py`)
Overlap 12.0 km^2. IPGP: std 91 m; best-fit plane slope -75 m/km east,
+20 m/km north (tilt 4.5 deg); residual std after plane 46 m. Ours: tilt
0.16 deg, residual std 3.8 m. Correlation 0.14 (0.21 after plane removal).

### IPGP heights as a hypothesis in our forward model
Same 3-fold CV, IPGP heights held fixed, poses and brightness refit;
held-out views scored after their own pose refit
(`logs/cv/ipgp_hypothesis_vs_flatpose.txt`), against flat + pose refit:

| height map held fixed | mean diff | 95% CI | views better |
|---|---|---|---|
| IPGP full | -0.068 | [-0.121, -0.028] | 4/37 |
| IPGP plane only | -0.069 | [-0.125, -0.027] | 3/37 |
| IPGP with plane removed | -0.0024 | [-0.0052, +0.0001] | 19/37 |
| ours (s300) | +0.0006 | [-0.0002, +0.0014] | 21/37 |

The IPGP 4.5 deg planar tilt makes held-out DISR views clearly worse to
predict. Its non-planar relief is neutral in this test. Caveat under
test: a common terrain tilt is degenerate with a common camera rotation,
so the rejection may come from the 1 deg attitude prior (Karkoschka's
attitudes, which use the SLI horizon) rather than from the images. Runs
with 3 and 10 deg priors are in progress.

### IPGP heights against measured parallax (`logs/reference/parallax_hypothesis.txt`)
After per-pair affine detrending (which removes planar terms), IPGP
window heights correlate with the measured along-parallax shifts:
r = 0.15, 95% CI [0.08, 0.23] (bootstrap over view pairs). Ours: r = -0.03.
Best-fit amplitude of IPGP relief: 0.38 [0.17, 0.54]. Taken at face
value, the images support IPGP's window-scale relief pattern at under
half its amplitude. Two effects could lower this estimate: random error
in IPGP heights (errors in the predictor), and bias in our shift
measurement. An injection test with synthetic views of a known surface
is set up (`scripts/synth_views.py`) to calibrate the measurement.

Current reading: IPGP captures some real window-scale relief that our
height-field method did not recover (ours is too smooth). The data do not
support IPGP's 4.5 deg tilt given Karkoschka's attitudes.

### Tilt: sensitivity to the attitude prior (`logs/cv/tilt_prior_sensitivity.txt`)

| attitude prior sigma | IPGP plane minus flat | 95% CI | plane better |
|---|---|---|---|
| 1 deg  | -0.069 | [-0.125, -0.027] | 3/37 |
| 3 deg  | -0.041 | [-0.089, -0.011] | 4/37 |
| 10 deg | -0.028 | [-0.071, -0.005] | 11/38 |

The penalty on IPGP's tilt shrinks as the attitude prior is loosened, but
it stays significant with a 10 deg prior, where the cameras are nearly
free to rotate. What still breaks the tilt/rotation degeneracy is the
position prior. Tilting terrain and cameras together by 4.5 deg would move
camera altitudes by 100s of m over the 5-15 km lever arms, against a 50 m
altitude prior (App. 3 altitudes, from the DTWG descent profile). So the
statement is: given the DISR images and the published descent altitudes
and positions, a 4.5 deg regional tilt of the IPGP area is disfavoured
relative to near-level ground. This conclusion inherits any systematic
error in those altitudes.

## 2026-09-28: injection tests (calibration of both tests and of the method)

`scripts/synth_views.py` renders every view by tracing its pixel rays to a
known height field and taking our frozen brightness map there (1% noise,
optional per-exposure pose perturbation; the pipeline is given the
unperturbed priors). `scripts/injection.sh` runs the same pipeline.
Truth: IPGP heights in our frame, either with the plane removed
(`detr`, relief std 34 m in the overlap) or complete (`full`, 4.5 deg tilt).
Logs in `logs/injection/`.

Parallax measurement calibration (per-pair affine removed, amplitude scale
k of measured shifts against the TRUE heights):

| scenario | k | corr |
|---|---|---|
| detr, no pose error | 0.46 [0.33, 0.59] | 0.22 [0.15, 0.29] |
| detr, 0.5 deg / 50 m pose error | 0.43 [0.28, 0.61] | 0.17 [0.12, 0.23] |
| full (tilted), 0.5 deg / 50 m | 0.27 [0.11, 0.50] | 0.12 [0.04, 0.19] |
| real data vs IPGP (for comparison) | 0.38 [0.17, 0.54] | 0.15 [0.08, 0.23] |

So the measurement recovers roughly 0.3-0.5 of the true window-scale
amplitude, even when the truth is known exactly. The real-data values
against IPGP fall inside the synthetic ranges. Correction to the earlier
reading: the images do not show that IPGP overstates its relief. They are
consistent with IPGP's window-scale relief at its full amplitude, within
wide limits.

Reconstruction recovery (our method, against truth):

| scenario | config | relief std truth / recovered | corr (plane removed) | plane slope truth / recovered (m per 20 m texel) |
|---|---|---|---|---|
| detr, no pose error | s300 | 34 / 3 m | 0.30 | 0 / 0 |
| detr, no pose error | s30 | 34 / 15 m | 0.47 | 0 / 0 |
| detr, pose error | s300 | 34 / 5 m | 0.11 | 0 / 0 |
| detr, pose error | s30 | 34 / 21 m | 0.21 | 0 / 0 |
| full, pose error | s300 | 34 / 4 m | 0.16 | (-1.51, 0.41) / (-0.04, -0.01) |
| full, pose error | s30 | 34 / 25 m | 0.21 | (-1.51, 0.41) / (-0.22, -0.04) |

Our height-field reconstruction strongly under-recovers relief and
recovers almost none of a real 4.5 deg tilt. Therefore:
- the near-flat frozen product v1 is a property of the method, not
  evidence that the terrain is flat;
- v1's near-zero tilt is not evidence against IPGP's tilt.
Whether the IPGP-plane-versus-flat CV test can detect a real tilt is being
checked on the tilted synthetic views (`logs/cv/injection_tilt_cv.txt`).

### Tilt test has power (`logs/cv/injection_tilt_cv.txt`)
On synthetic views whose truth carries IPGP's full heights (4.5 deg tilt),
with 0.5 deg / 50 m pose perturbations, the same CV test gives IPGP plane
minus flat = +0.092 [+0.048, +0.153], better in 37/37 views. On real data it
gives -0.069, better in 3/37. So the CV test detects a real tilt of this
size, and the real-data rejection is not an artefact of the test.

### Direct plane fit (`--plane`, `logs/reference/plane_fit_real.txt`)
Two plane parameters fitted jointly with poses and brightness (h = 0
otherwise). Validation: synthetic truth (-75.4, +20.4) m/km (east, north)
is recovered as (-67.4, +24.5). Synthetic level truth gives (+6.0, +1.1).

Real data:

| fit | dh/dE (m/km) | dh/dN (m/km) |
|---|---|---|
| all exposures | +12.2 | +28.2 |
| 80% subsets, seeds 1-5 | +3.8 to +13.8 | +23.9 to +40.8 |
| attitude prior 3 deg | +7.3 | +38.8 |
| attitude prior 10 deg | +1.4 | +63.3 |
| IPGP DTM, same area | -75.4 | +20.4 |

The east-west component is stable across subsets and priors at about
0 to +14 m/km, against IPGP's -75 m/km. The north-south component is less
certain (24 to 63 m/km) and depends on the attitude prior; IPGP's value
is at the low end of that range. The disagreement with IPGP is therefore
mainly the east-west slope.

Untested hypothesis: IPGP used SPICE attitude (ESA re-computation of the
DTWG solution), and the probe's east-west tilt is the least constrained
attitude component (the DISR archive lists it separately: Users' Guide
App. 4 and HUYGENS_DESCENT_PARAMETERS column 8, derived from radio Doppler
with rapid swings not modelled). A difference in east-west tilt between
the SPICE attitude and Karkoschka's image-based attitudes would map into
an east-west terrain slope. Checking this needs the SPICE kernels
(doi:10.5270/esa-ssem3np), which have not been fetched.

## 2026-09-28: new SfM pipeline (v2)

Goal: a structure-from-motion reconstruction with explicit tie points,
triangulation and bundle adjustment, as an alternative to the dense
height field (which the injection tests showed under-recovers relief).

Method (`scripts/sfm.py`):
- Multi-view tracks by guided matching. Each view is orthorectified onto
  the current surface with the current poses. At a grid of nodes (16-texel
  = 320 m windows, stride 8), each view's window is phase-correlated
  against a reference mosaic, the mean of the normalised orthos. The
  matched ortho location, lifted onto the surface and projected through
  the same view, is the observed image coordinate. About 500 tracks with
  5-6 views each.
- Bundle adjustment of track points and per-exposure pose corrections
  (App. 3 priors), Huber loss, L-BFGS, outlier rejection at 6 px.
- Iteration: grid the points (sigma-weighted Gaussian, 150 m), rebuild
  orthos and reference, re-match, re-adjust (8 cycles).
- Tilt: a profile over the point cloud's plane slope. Each grid value is
  held by a constraint while everything else is re-optimised; a quadratic
  fit to the cost gives the minimum. The free adjustment barely moves the
  plane because the cost is nearly flat along the tilt/pose degeneracy.
- Per-point height sigma from each point's 3x3 normal matrix.

Settings were chosen on synthetic injection data only (the IPGP reference
had already been opened, so tuning on real data was ruled out):
- Pixel sigma 0.5 px under-weighted the navigation priors (actual
  residuals were about 3x larger). With the tilted truth plus 0.5 deg /
  50 m pose error, the profile returned -13 m/km (truth -75).
- 1.5 px gave (-53, +18) for truth (-75, +19); 2.5 px gave (-63, +17),
  and (-7, -11) for level truth. 2.5 px adopted.
- Clean case (no pose error, no noise), 15 cycles: profile (-67 +- 6,
  +22 +- 5) m/km against truth (-75, +20).

Real data (frozen before comparison: `products/sfm_v2`, commit 164f496):
523 points, median 5 views, median sigma_h 102 m. Profile minimum
dh/dE +0.2, dh/dN +16.0 m/km (formal +-13). Against IPGP after
registration: plane (+0.3, +15.2) vs IPGP (-76.2, +16.2); relief
correlation after plane removal 0.09.

Monte Carlo over pose-error draws (0.5 deg, 50 m, 1% noise; `logs/sfm/`),
profile dh/dE:

| truth | draws | estimates (m/km) | mean +- sd |
|---|---|---|---|
| IPGP full, -75 | 6 | -62.6, +24.2, -40.6, -25.3, +33.1, +26.4 | -7.5 +- 39 |
| level, ~0 | 5 | -7.2, -21.7, +1.8, -5.6, -3.6 | -7.3 +- 9 |

With realistic navigation errors, the SfM tilt estimate does not
separate a 75 m/km slope from level ground. The formal +-13 m/km is far
too small. Relief accuracy on synthetic data (`logs/sfm/point_accuracy.txt`)
is also weak: correlation with truth after plane removal 0.05-0.30,
regression slope 0.13-0.56. The SfM v2 real-data output is therefore not
evidence about the tilt, and its heights are not a usable DTM.

In contrast, the dense photometric CV test (IPGP plane held fixed against
flat, poses refit) keeps its power on a second pose-error draw: +0.095
[+0.044, +0.164], tilt preferred in 34/37 held-out views
(`logs/cv/injection_tilt_cv_seed11.txt`; first draw: 37/37). The dense
test uses every textured pixel, not about 500 window matches. It remains
the basis for the statement that IPGP's regional tilt is disfavoured.

Why sparse SfM is weak here: the parallax the tilt produces between
views is a few pixels spread smoothly across the field. Per-exposure
pose errors of 0.5 deg produce smooth shifts of similar size and pattern.
With about 500 noisy tracks (1-2 px), the adjustment cannot tell them
apart, and the result depends on which local minimum the iteration
reaches.

## 2026-09-28: modern SfM toolchain (DISK + LightGlue + COLMAP)

Aim: replace the hand-built matcher with a standard current toolchain and
test it on the real descent images.

Toolchain (`scripts/colmap_sfm.py`, pycolmap 4.2, kornia 0.8.3):
- Export: the 40 region views (7 SLI, 14 MRI, 19 HRI) as 8-bit PNG. The
  G-image camera is exactly COLMAP PINHOLE (f = 1/SC, cx = W/2, cy = 128)
  once rows are flipped. As stored, the column, row and optical axes form
  a left-handed frame. Flipping the rows makes it right-handed and puts
  the SLI horizon at the top, as expected.
- Features: DISK (depth weights), images upsampled 2x, median 855
  keypoints per image. (ALIKED weights could not be downloaded: GitHub
  raw returned 403.)
- Matching: LightGlue (DISK weights) on 358 candidate pairs chosen by
  footprint overlap under the App. 3 poses. Median 30 matches per pair.
  COLMAP geometric verification: 337 pairs, 25,817 inlier matches.
- COLMAP incremental mapping with position priors registered 37 of 40
  views: 1,714 points, mean track length 6.2, reprojection 1.13 px.
  Automated matching of DISR images is therefore feasible with current
  learned features; the USGS work (as summarised by Daudon et al.) found
  automated matching unsuccessful.
- COLMAP's own adjusters, however, do not give usable geometry here. On
  synthetic views with known terrain, both the incremental model
  (aligned to the prior camera centres by a weighted similarity,
  `scripts/colmap_eval.py`) and the prior-pose model after COLMAP bundle
  adjustment came out tilted by hundreds of m/km, with 300-800 m
  camera-centre residuals. COLMAP's pose-prior adjustment supports only
  position priors. Attitude priors, which the tilt/attitude degeneracy
  needs, are not available.
- Therefore: COLMAP tracks (triangulated at the App. 3 poses, before any
  COLMAP adjustment) + our navigation-constrained bundle adjustment
  (`scripts/colmap_prior_ba.py`; 1 deg attitude, 200/50 m position
  priors, shared per exposure; Huber, 2.5 px) + the tilt profile.

Test fix: synthetic views previously kept real pixels outside the 5 x 5 km
region. Whole-frame matchers pick up those real features and mix real
and synthetic geometry. `synth_views.py --mask-outside` now blanks them.

Real data (frozen before comparison: `products/sfm_colmap_v1`, commit
10af208): 2,060 tracks (>= 3 views), 11,100 observations, 1.55 px RMS;
pose corrections 0.4 deg / 54 m RMS. Tilt profile grid minimum at (0, 0)
m/km (quadratic minimum +1.9, -3.8). Cells near IPGP's plane cost
100-130 units more. Points in region: 1,381.

Monte Carlo, masked synthetic views, 0.5 deg / 50 m pose error
(`logs/colmap/mcc_*.txt`), east slope of the profile minimum:

| truth | draws | dh/dE estimates (m/km) | mean +- sd |
|---|---|---|---|
| IPGP full, -80 | 4 | -92, -115, -40, -60 | -77 +- 33 |
| level, ~0 | 4 | +32, -8, +25, +10 | +15 +- 18 |
| real data | | +1.9 | |

The east component is recovered without bias for tilted truth, and it
separates the two cases in all 8 draws. The real-data value lies inside
the level distribution (z = -0.7) and outside the tilted one (z = +2.4).
Gaussian likelihood ratio, level versus IPGP's slope: about 25. This is
moderate evidence from 4 draws per class. The north component is not
reliable (estimates -34 to -183 m/km for truths near +10 and 0) and is not
reported.

Relief against IPGP (`logs/colmap/point_accuracy.txt`): correlation after
plane removal 0.16 (977 points), against 0.09 for the hand-built SfM v2
and -0.03 for the dense v1 height field. On synthetic truth: 0.31
(tilted) and 0.20 (level), regression slope 0.6-0.7. Point heights are
still too noisy (68-110 m scatter against 35 m relief) for a DTM product.

Caveat: synthetic frames are rendered from the 20 m brightness map and
are blurrier than real frames (median 2 LightGlue matches per pair
against 30 on real data). The Monte Carlo spread probably overstates the
real-data uncertainty. The best available real-data calibration would
come from rendering at native resolution, which needs a sharper texture
model than the 20 m mosaic.

Summary across methods: dense photometric CV (two draws, 37/37 and
34/37), and now the COLMAP-track SfM (east slope, 8/8 draws separated),
both indicate that the images plus the descent navigation favour
near-level ground over IPGP's 4.5 deg east-down slope. The hand-built
SfM v2 had no power for this question.

## 2026-09-28: figure archive

All generated images are now kept in `figures/` (index with provenance and
attribution in `figures/INDEX.md`, checksums in `figures/SHA256SUMS`).
Earlier figures that existed only in the ignored `work/` directory were
copied in; stages that had no figures got them via `scripts/make_figures.py`.
Rule from here on: every figure used to reach a conclusion is added to
`figures/` and the index in the same commit as the conclusion.

## 2026-09-28: literature read (full texts supplied by the user)

Papers (PDFs kept outside git, in `work/lit/`, for copyright reasons):
Karkoschka et al. 2007 (PSS 55, 1896-1935, doi:10.1016/j.pss.2007.04.019);
Karkoschka 2016 (Icarus 270, 326-338, doi:10.1016/j.icarus.2015.06.012);
Karkoschka & Schröder 2016 (Icarus 270, 307-325, doi:10.1016/j.icarus.2015.08.006,
accepted manuscript); Soderblom et al. 2007 (PSS 55, 2015-2024).

### How the App. 3 poses were made (Karkoschka et al. 2007, Sections 3-4)
- Altitude: taken from the Descent Trajectory Working Group (HASI
  pressure/temperature, GCMS molecular weight), independent of DISR.
  Image-scale checks confirm it to within a few percent (2% below 3 km).
  Radar altimeter altitudes (6% higher) were rejected.
- Position, azimuth, pitch, roll: fitted so that all images form one
  mosaic "projected onto Titan's surface, which is assumed here to be a
  perfect sphere of radius 2575 km". Iterative, largely manual.
- Average pitch/roll per altitude range: from sky-brightness mosaics
  (independent of terrain). The one component the sky cannot fix, tilt
  toward or away from the Sun (Sun azimuth about 113 deg, nearly east),
  was set from Doppler Wind Experiment zonal accelerations (parachute
  tilt). The last 20 SLI images (below 3 km) give pitch and roll from the
  horizon (0.2 deg and 1 deg), assuming a flat spherical surface.
- Zonal trajectory agrees with DWE above 12 km; between 12 and 5 km
  (where the IPGP frames lie, 6.5-16 km) it rests on DISR imaging alone.

Consequence for this project's tilt result: the attitude priors are
partly the product of a flat-sphere fit. A real regional tilt could
have been absorbed into per-exposure pitch and roll. What does not come
from that fit: the altitudes (pressure-based), the sky-derived average
attitudes, and the DWE-derived east-west parachute tilt. The dense CV
test still preferred level ground with the attitude prior loosened to
10 deg, so the rejection of IPGP's slope does not rest mainly on the
attitude priors. It rests on the pressure altitudes and on the
horizontal trajectory, and the horizontal trajectory below 12 km is
itself image-derived under the sphere assumption. The result is
therefore partly, not fully, independent of a flat-ground assumption.
This must be stated with the result.

### Karkoschka 2016 (orientation and swing)
- A re-derivation with the improved image calibration: 1600 manually
  measured image-pair offsets, 501 orientation angles plus 27 trajectory
  parameters, computational iteration. Stated to be four times more
  accurate than 2007. Tie points are still projected onto the sphere.
  Offsets are measured on flat regions where possible (the dark
  lakebed); hilly regions are down-weighted or used only for similar
  viewing directions. Horizon-detected tilts enter as constraints.
- Parameters are well determined at 5-20 km altitude (panoramas
  P10-P17), which includes the IPGP frames.
- The per-exposure angles are not tabulated in the paper, so the 2012
  archive table (App. 3, 2007 solution) remains the numerical source
  used here. Asking the PSA or the author for the 2016 angles would
  tighten the priors.
- The mosaic azimuth was rotated 1 +- 1 deg counterclockwise to match
  Cassini RADAR. This is far too small to explain the -109 deg rotation
  between our frame and the IPGP .tfw axes, which remains unexplained.
- The author reports small-scale distortions in bright ridges and
  highlands that must be due to topography, while lakebed features fit
  one offset. This agrees qualitatively with the COLMAP-track points.

### Karkoschka & Schröder 2016 (mosaic vs emission angle)
Their topography is qualitative: blinking side-looking against
down-looking mosaics (both projected on the sphere), with a detection
limit of about 50 m. Findings: the highland north of the landing site has
terrain elevated by more than 100 m, with a steep slope at the front
shoreline; the islands and most bright ridges in the lakebed are flat and
not elevated; no evidence for the USGS 180-200 m plateau. They produce no
DTM and no regional slope. So their work does not duplicate this
project's quantitative multi-view reconstruction, and their qualitative
findings are a check any DTM here should pass.

### Soderblom et al. 2007 (USGS DTMs)
- SOCET SET; Region 1 from 6 stereo pairs (HRI 414, 450; MRI 553, 601,
  634), about 3 x 5 km, 30 m stated accuracy; Region 2 from HRI 498 /
  MRI 595, 30-50 m. Automated matching was attempted and failed; posts
  were collected manually.
- "The integrated DTM was then arbitrarily leveled (residual tilt
  reduced); this was necessary as no surface datum yet exists." The
  USGS regional slope is therefore an assumption, not a measurement.
- Camera positions (Karkoschka 2006) agree with App. 3 to within tens
  of metres (e.g. #414: 16.61 km altitude, 1.96 km west, 0.13 km south).

### Effect on claims
- Novelty: no prior work gives a measured regional slope for this area
  (USGS levelled arbitrarily; IPGP chose one of an ambiguous family;
  K&S 2016 qualitative). Automated matching of DISR frames with learned
  features (37/40 registered) contrasts with the failure reported by
  USGS. Both are new, within the caveats above.
- The tilt statement must carry the partial-circularity caveat.

## 2026-09-28: Daudon et al. (2020), the IPGP DTM paper

Full text: open access (CC BY), University of Arizona repository copy;
reference and fetch script in `third_party/papers/`.

### How IPGP fixed position, orientation and tilt
- Camera positions and orientations from ESA's recomputed SPICE kernels,
  described as accurate only to "larger than 0.24 deg in longitude and
  0.17 deg in latitude" (about 10 and 8 km on Titan), "inadequate for
  photogrammetry".
- MicMac relative orientation, then an absolute orientation by a 3D
  similarity ("homothety") fitted to the SPICE camera *positions*, then
  ground control points on the shoreline constrained to equal elevation
  ("the shoreline being an equipotential"), then bundle adjustment.
- The result still had a global tilt of about 9.5 deg, with the lakebed
  above the hills. The authors attribute this to "an intrinsic
  mathematical indeterminacy" in the global orientation.
- Tilt chosen by river routing: the DTM was rotated about x and y from
  -20 to +20 deg in 1 deg steps; for each rotation, flow routing
  (TopoToolbox) was compared with a hand-drawn river mask; best score
  62% at -10 deg about y and +3 deg about x. The result is described as
  "close to horizontal, with only a remaining gentle global slope and
  rivers going down the slope". That remaining slope is the 4.5 deg plane
  measured here (after registration: -75 m/km east, +20 m/km north in
  our frame).
- The same routing test on the USGS DTM (score 35% as published) rose to
  56% after tilting it 4 deg about y and 3 deg about x. So the routing
  criterion tends to prefer a few degrees of regional slope in both DTMs.

### Explanation for the -109 deg frame rotation (new, checked here)
The eight IPGP cameras (App. 3 positions) are almost exactly coplanar
and nearly collinear: singular values of the centred positions
10.28, 0.63 and 0.024 km; the principal axis runs east and down at 74 deg
elevation; RMS distance from that line 224 m; all eight lie within 70 m of
one north-south position. A similarity fitted to camera positions alone
cannot determine rotation about a line through the cameras. With SPICE
position errors of several km, rotation about this near-vertical axis,
which is mostly an azimuth rotation, is effectively free. Levelled
shoreline points then fix tilt, but nothing fixes north. This accounts
for the -109 deg rotation and the 1-5% scale difference found by
registration (`derived/ipgp_to_ours.json`). The paper's Appendix A
defines the local tangent frame with north and east axes; the stored
data axes are not north-aligned, consistent with this degeneracy. (A
109 deg rotation about an axis 16 deg from vertical also tilts the
surface by about 24 deg; the shoreline levelling and the routing
rotation act on tilt only, not on azimuth.)

### Relation to this project's tilt result
IPGP's final slope comes from a landform criterion (rivers must flow
downhill along the mapped channels), not from navigation. This
project's result comes from the images plus pressure altitudes and
Karkoschka's trajectory. The two are different kinds of evidence. A
nearly level regional surface with an elevated highland (Karkoschka &
Schröder 2016: highland more than 100 m above the lakebed) is also
compatible with rivers flowing from the highland to the plain. The
routing score may reward a regional slope because it makes flow
directions consistent over a noisy DTM, as the USGS result (+21 points
after a 3-4 deg tilt) suggests. That is a hypothesis, not tested here.
Testing it would mean running the same routing score on the IPGP DTM
with its regional slope removed.

## 2026-09-28: DTM v2, a presentable model

Goal: a terrain model that can be shown, with an honest resolution and a
calibrated uncertainty.

Chain (settings fixed on synthetic data only; frozen before comparison):
1. COLMAP / LightGlue tracks + navigation-constrained BA
   (`colmap_prior_ba.py`, now also writes per-point height sigma and the
   max ray intersection angle). Real data: 1,382 points, median point
   sigma 83 m (synthetic: 114-126 m, whose texture is blurrier).
2. Robust, inverse-variance Gaussian gridding (`grid_points.py`, 200 m
   kernel, Huber reweighting; uncertainty = max of propagated and
   empirical scatter). Real: 20.0 km^2 supported.
3. Dense photometric fit seeded from that grid (`reconstruct.py
   --h-init`, smoothing 30).
4. Product (`make_product.py`): smoothing to 300 m within the support
   mask; uncertainty = gridding sigma x calibration factor.

Synthetic evaluation (masked views, 0.5 deg / 50 m pose error), scale
dependence for the seeded dense fit (`eval_scales.py`), tilted truth:
r = 0.34 / 0.48 / 0.60 / 0.78 at 0 / 200 / 400 / 800 m smoothing, amplitude
0.28 / 0.56 / 0.80 / 1.13; level truth: r = 0.44 / 0.61 / 0.68 / 0.81.
Seeding helps mainly the amplitude and the regional slope (seeded -60 vs
flat-start -10 m/km for truth -76).
At the 300 m product scale (`make_product.py calibrate`): relief r 0.54
and 0.65, amplitude 0.68 and 0.83, rms error 21-23 m against 20-21 m truth
relief, err/sigma rms 1.04 and 1.06 -> calibration factor 1.05. Regional
slope: truth (-73, 17) est (-58, 15); truth (1, -2) est (23, -2) m/km.

Real product `products/dtm_v2` (commit b43ed93): 20.0 km^2, relief after
plane 23 m rms, plane (+8, +10) m/km, median sigma 19 m,
p5/p95 -60/+35 m. Figure 16.

Checks after freezing (`logs/dtm_v2_checks.txt`):
- Bright highland (y > 3.4 km) minus dark lakebed (2.0-3.3 km): +27 m,
  conservative 1-sigma 36 m. The sign agrees with Karkoschka & Schröder
  2016 (highland elevated); the magnitude is not significant under the
  conservative error, and is likely underestimated (amplitude 0.7-0.8).
- The most prominent feature is a trough about 60 m deep along the
  dark lakebed just south of the shoreline, with the highland rising to
  the north. This matches the steep shoreline front described by K&S 2016.
- Against IPGP (both at 300 m, planes removed, 11.2 km^2): r = 0.52;
  our relief is 0.37 of IPGP's.

Viewer: `viewer/huygens_terrain.html` (built by `build_viewer.py` from
`viewer/template.html`), published as a private artifact. It shows the
model draped with the 20 m mosaic, height and uncertainty colourings, the
IPGP surface for comparison, and dims low-confidence cells.

What this model is and is not: a kilometre-scale terrain model with
calibrated uncertainty and a measured regional slope. It does not
resolve the channel-scale relief visible in the images; the IPGP DTM has
finer (if noisier) detail.

## Sharper synthetic tests and DTM v3 (70 m)

Motivation: DTM v2 was validated on synthetic frames rendered from the
20 m brightness map. Those frames are smoother than the real ones, so
the matcher saw an easier problem. The tests below make the synthetic
frames resemble the real ones before trusting any finer product.

Rendering changes (`scripts/make_texture.py`, `scripts/synth_views.py`):
- 5 m texture: low frequencies (above 150 m) from the 20 m mosaic, fine
  detail from the sharpest real view covering each texel (weight
  gsd^-4, feathered view edges).
- 3x3 supersampling per pixel.
- `--match-contrast`: the rendered 1-4 px band is scaled to the real
  frame's amplitude in the same footprint. Without it the synthetic
  frames looked blurrier than the real ones; the difference was contrast
  (haze-reduced real detail vs texture detail), not resolution.
- Noise 0.005 of median; pixels outside the region masked.

Result 1: the dense photometric stage used in v2 diverges on these
frames (relief grows, correlation with truth falls). v2's dense-stage
amplitude, including its 60 m lakebed trough, is therefore not supported.
v3 drops the dense stage.

Result 2: v3 chain (`scripts/chain_synth_v3.sh`): DISK + LightGlue
(1024 keypoints for speed; 2048 on draws 7 and 21), COLMAP verification
and triangulation, navigation-constrained adjustment
(`colmap_prior_ba.py`), inverse-variance gridding with a 70 m kernel
(`grid_points.py`). Truth: IPGP relief with its tilt. Five pose-error
draws (0.5 deg, 50 m), `logs/v3_montecarlo.txt`:

| draw | area km^2 | r 0 m | r 100 m | r 200 m | k 100 m |
|---|---|---|---|---|---|
| 7  | 10.0 | 0.69 | 0.84 | 0.90 | 1.01 |
| 21 |  -   | 0.27 | 0.33 | 0.34 | 0.61 |
| 22 |  1.7 | 0.29 | 0.32 | 0.29 | 0.52 |
| 23 |  3.0 | 0.70 | 0.88 | 0.94 | 0.92 |
| 24 |  4.8 | 0.44 | 0.47 | 0.42 | 0.75 |

Median r at 100 m 0.47. Outcome depends strongly on the draw, that is on
the pattern of navigation error, not on the terrain. On the poor draws
the stated sigma is too small by about 1.55x (error/sigma rms 1.54, 1.59);
on draw 23 it is conservative (0.79). The calibration factor 1.24 used for
v3 lies between these. Level truth on draw 7: r 0.61/0.73/0.80 at
0/100/200 m.

Result 3: split-half consistency (`scripts/split_half.py`, exposures
split in two, each half gridded separately, relaxed mask). Real data:
r 0.57 native, 0.66 at 100 m over 5.4 km^2. Synthetic draw 7 (good):
0.62/0.72. Draw 21 (poor): about 0. Draws 22-24: overlap too small to
test (`logs/v3_split_half.txt`). The real value sits with the good draw.
This is evidence, not proof, that the real navigation errors fall in
the favourable regime.

Real product `products/dtm_v3` (commit d39d5d6, frozen before
comparison): 7.0 km^2 of highland, median sigma 34 m, relief after
plane 23 m rms, plane (+5.0, +0.7) m/km. Figure 18. Checks after freezing
(`logs/dtm_v3_checks.txt`):
- vs IPGP (planes removed, 5.6 km^2): r 0.53 native, 0.68 at 100 m,
  0.72 at 200 m; v3 relief 0.41-0.52 of IPGP's.
- vs v2 on the v3 mask: r 0.63.
- Lakebed band minus highland band: v3 -21 m (1-sigma 46 m), v2 -59 m.
  The v2 trough is not reproduced at significance.

Amplitude: on synthetic draws v3 recovers 0.5-1.0 of true relief at
100 m. The real v3/IPGP ratio (about 0.5) is at the low end of that
range, so part of the difference may lie in the IPGP model; this is not
established.

Viewer updated to v3 (`viewer/huygens_terrain.html`, same artifact URL).
Unmapped areas show a smoothed, dimmed continuation labelled as such.
