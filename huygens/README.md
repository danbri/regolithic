# Huygens modern DTM experiment

Goal: independently reconstruct Titan terrain from public Huygens/DISR observations using modern matching, metric camera constraints, multi-view geometry, and geometry-regularised Gaussian/surface methods.

## Provenance rule
No archive terrain product is ever placed in `data/raw`. Existing DTMs/orthomosaics go only in `data/reference` and are withheld from primary reconstruction until validation. Every external input is recorded with URL, retrieval date and SHA-256.

## Novelty rule
A result is not called new merely because it was regenerated. We distinguish: (A) exact/near reproduction, (B) reprocessing with new software, (C) additional observations/constraints, (D) materially new inferred geometry. Claims require comparison against archived USGS/IPGP products and literature.

## Status
See `REPORT.md` for current findings and `NOTEBOOK.md` for the full record.

## SfM toolchain (DISK + LightGlue + COLMAP)

    python3 scripts/colmap_sfm.py work/colmap_real          # export, features, match, map
    python3 scripts/colmap_prior_ba.py work/colmap_real --tilt-scan=-120,-80,-40,0,40,80
    python3 scripts/eval_points.py work/colmap_real/priorba derived/ipgp_h_on_grid.npz

Requires `pip install pycolmap kornia` (pycolmap 4.2 used). Synthetic
injection runs: set `HDTM_SYNTH` to a `synth_views.py --mask-outside` output.
Monte Carlo: `scripts/mc_colmap.sh`.

## Layout
- `NOTEBOOK.md`: dated log of steps, findings and open questions.
- `scripts/fetch.py`, `scripts/fetch_all.sh`: download with provenance. `data/` itself is not committed; `data/MANIFEST.jsonl` is.
- `hdtm/`: Python package (image I/O, camera model, pose priors).
- `scripts/check_*.py`: convention and consistency tests.
- `derived/`: small tables derived from archive documents.
- `logs/`: outputs of checks and runs.
- `figures/`: archived images from every stage, indexed in `figures/INDEX.md`.

Requirements: Python 3.11, numpy, scipy, opencv-python-headless, torch (CPU), pypdf.
