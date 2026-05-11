# Kitchen Table — Gaussian Splat

Run `./fetch.sh` to materialize the binaries.

## Source

- **Title:** Kitchen Table
- **Author:** wrender ([SuperSplat profile](https://superspl.at/user/wrender))
- **Scene page:** https://superspl.at/scene/ffa72aa0
- **Embed viewer:** https://superspl.at/s?id=ffa72aa0
- **Description (verbatim):** demonstrates "reflections and transparency,"
  with attention to "reflections on the apple" and transparent objects like
  "the bowl and glasses" that "do NOT refract the light."

## License

Published as a **Downloadable** scene on SuperSplat. Per the SuperSplat
documentation, "Any splat on SuperSplat tagged Downloadable has been published
under Creative Commons by its author" — see
<https://blog.playcanvas.com/new-in-supersplat-downloadable-splats-licenses-and-social-links/>.

The exact CC variant (e.g. CC0 / CC-BY 4.0 / CC-BY-SA / …) is shown in the
download dialog on the live scene page. **Confirm the variant before
redistribution.**

When redistributing, credit at minimum:

> "Kitchen Table" by wrender — https://superspl.at/scene/ffa72aa0 (CC license
> per SuperSplat scene page)

## Files

Format: streamed SOG (sets of WebP textures + meta.json), produced by
`splat-transform v0.16.1`. Splat count: 2,737,026.

| File | Bytes | SHA-256 |
|---|---:|---|
| `meta.json` | 15,313 | `2f97954b56368e988547b7d6167ba10aab2c69b9a045b4db74da7e99cced5119` |
| `means_l.webp` | 8,051,984 | `628b75ff2346d73e29d67a9313f055435c0eceef08ce19e6d2f0919cbf97233c` |
| `means_u.webp` | 1,334,394 | `373a316648fa88d7e2e5357def3c94ae53d334c65f3ed26ba0b424ff2cff3f38` |
| `scales.webp` | 6,888,362 | `f01d6e8c0a663a34e1ab649eeca987a069687b066b89ab50af3d8ef78f796bc3` |
| `quats.webp` | 8,858,844 | `3c68f3173383fe60db75260eec8105ff0bd5fbd0b0fa2490dd828a96bb959986` |
| `sh0.webp` | 7,927,586 | `d34e0c7973647864a32fb0999e3326dabca5828b97050e2ff420fd62bf38340a` |
| `shN_centroids.webp` | 1,910,094 | `30218046a78dbfb9a12c74e1591ac23fd592b525846a789ce46659cd8b9f4563` |
| `shN_labels.webp` | 4,854,232 | `647355143c0449131b4a55a0db3d56c52491e8daa71d7d4975e4b2fdeef96904` |

## Upstream URL pattern

`https://d28zzqy0iyovbz.cloudfront.net/ffa72aa0/v1/<file>` (PlayCanvas CDN
fronting the `splats.playcanvas.com` S3 bucket in `eu-west-1`).

Captured: 2026-05-11.
