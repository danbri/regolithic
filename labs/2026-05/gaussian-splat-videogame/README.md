# gaussian-splat-videogame

Reproduce and probe the workflow from the PlayCanvas blog post
["Turning a Gaussian Splat into a Videogame"][post] — convert a Gaussian Splat
scene into a playable, navigable, lit environment in the browser.

[post]: https://blog.playcanvas.com/turning-a-gaussian-splat-into-a-videogame/

## Reference demo & project

- Playable demo: https://playcanv.as/p/qxGSuzYq/ (WASD + mouse, click to fire)
- Public PlayCanvas project: https://playcanvas.com/project/1480299

## Pipeline under test

1. **Source splat** — grab a `Downloadable` (CC-licensed) scene from
   [SuperSplat](https://superspl.at/).
2. **Convert** `.ply` → streamed SOG with LODs:
   ```
   npx @playcanvas/splat-transform input.ply output/ --sog --lod
   ```
3. **Collision mesh** — voxelize the splat to a watertight `.collision.glb`:
   ```
   npx @playcanvas/splat-transform input.ply collision.glb -K
   ```
4. **Lighting** — bake a lightness grid by sampling luminance at probe
   positions via offscreen rendering.
5. **Navmesh** — precompute `navmesh.bin`, load at runtime with
   [recast-navigation-js](https://github.com/isaac-mason/recast-navigation-js):
   ```js
   const recast = await import('https://esm.sh/recast-navigation');
   await recast.init();
   const navmesh = recast.importNavMesh(new Uint8Array(buf));
   ```
6. **AI** — behaviour trees over `sequence` / `selector` / `condition` /
   `action` primitives, parameterised by per-NPC personality traits.

## Test scenes

Vendored under [`third_party/splats/`](../../../third_party/splats/) (binaries
fetched on demand via `fetch.sh`):

| Scene | Author | License | Format | Use |
|---|---|---|---|---|
| [Botanical Garden – Victoria House](../../../third_party/splats/superspl.at-victoria-house-6f697c4d/) (`6f697c4d`) | simonbethke | CC BY 4.0 | streamed multi-tile SOG + LOD (~305 MiB, 297 files) | exterior-ish: tropical greenhouse with trees, plants |
| [Kitchen Table](../../../third_party/splats/superspl.at-kitchen-table-ffa72aa0/) (`ffa72aa0`) | wrender | CC (variant TBC) | SOG (~39 MiB, 8 files) | interior, reflections/transparency |

Victoria House is the format the blog post actually recommends — a streamed
SOG bundle with LOD tiles, so it's a useful realistic test for the
"mobile-friendly" pipeline rather than a single fat PLY. Materialize with:

```sh
third_party/splats/fetch-all.sh
```

## Tests to run

- [ ] Convert a SuperSplat scene and load it in a stock PlayCanvas viewer.
- [ ] Compare SOG-streamed vs. raw `.ply` load time on a throttled connection.
- [ ] Generate a `.collision.glb` at varying voxel sizes; eyeball where the
      collision shell stops matching the visible geometry.
- [ ] Drop a character controller in and confirm walkable surfaces.
- [ ] Bake a navmesh and visualise it overlaid on the splat.
- [ ] Swap in a non-PlayCanvas viewer (e.g. raw `gsplat.js` / three.js) and
      see which stages of the pipeline transfer.

## Open questions

- How portable is the SOG format outside PlayCanvas?
- Voxelised collision vs. an authored proxy mesh — which wins for cost/quality?
- Can the lightness grid be replaced by something queryable from the splat
  itself (per-Gaussian SH coefficients)?

## Log

_Add dated entries as tests run._
