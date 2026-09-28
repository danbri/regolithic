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
