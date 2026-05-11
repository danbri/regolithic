# Botanical Garden – Victoria House — Gaussian Splat

Run `./fetch.sh` to materialize the binaries (~305 MiB across 297 files).

## Source

- **Title:** Botanical Garden – Victoria House (VR Ready)
- **Author:** simonbethke ([SuperSplat profile](https://superspl.at/user/simonbethke))
- **Scene page:** https://superspl.at/scene/6f697c4d
- **Embed viewer:** https://superspl.at/s?id=6f697c4d
- **Description:** Virtual recreation of the Victoria House greenhouse at the
  Kiel Botanical Garden (Germany) — tropical aquatic plants, mangroves, and
  economically significant crops including coconut palms, sugar cane, rice,
  papayas, and taro.

## License

**CC BY 4.0** (Creative Commons Attribution 4.0 International), per the
SuperSplat download dialog. Permits redistribution and adaptation with
attribution.

Attribution line:

> "Botanical Garden – Victoria House (VR Ready)" by simonbethke —
> https://superspl.at/scene/6f697c4d — licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/)

## Files

Format: **streamed multi-tile SOG with LOD** (the format the
[PlayCanvas blog post](https://blog.playcanvas.com/turning-a-gaussian-splat-into-a-videogame/)
recommends for mobile and slow connections).

- 1 root manifest (`lod-meta.json`)
- 37 tile directories (`0_0/`, `0_1/`, …, `1_0/`, `1_1/`, `2_0/`), each with
  its own `meta.json` and a set of WebP textures (`means_l.webp`, `means_u.webp`,
  `scales.webp`, `quats.webp`, `sh0.webp`, `shN_centroids.webp`, `shN_labels.webp`)
- **Total: 297 files, 320,329,438 bytes (305.5 MiB)**

## Integrity

`fetch.sh` verifies two SHA-256 hashes:

| What | SHA-256 |
|---|---|
| `lod-meta.json` (root manifest) | `4c8497336285f1b643073e390e61ac47783edd58a27ca8b406851372e1f2c0a3` |
| Aggregate (sorted `path\0content` over the full 297-file bundle) | `ff327961a15bbf532c42c15cc901a9876cd265bc253aeaa1b8cb9bee26b6e7c2` |

Recompute the aggregate hash with the helper inside `fetch.sh` (it is the
same construction).

## Upstream URL pattern

`https://d28zzqy0iyovbz.cloudfront.net/6f697c4d/v1/<path>` — PlayCanvas
CloudFront CDN fronting the `splats.playcanvas.com` S3 bucket in `eu-west-1`.

Captured: 2026-05-11.
