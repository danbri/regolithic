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

## Web viewer (`web/`)

A custom-element viewer that loads any scene from `web/catalog.json` and
runs on mobile, desktop, and WebXR (Quest 3 target).

```sh
# from repo root, after fetching splats:
labs/2026-05/gaussian-splat-videogame/web/serve.sh
# opens at http://localhost:8000/labs/2026-05/gaussian-splat-videogame/web/
```

For WebXR on Quest 3 you need HTTPS or a localhost route — the simplest path
is `adb reverse tcp:8000 tcp:8000` from a USB-tethered Quest, or a
self-signed cert via `mkcert` if you're on the same LAN.

### Controls

| Platform | Look | Move | Other |
|---|---|---|---|
| Desktop | drag | WASD / arrows, space/ctrl for up/down, shift = fast | wheel = dolly along view |
| Mobile  | 1-finger drag | 2-finger pinch = dolly | — |
| Quest 3 | head | thumbsticks (engine default) | controllers attach to experiments |

### Components

- `<splat-scene catalog="./catalog.json">` — the whole app. Initialises
  PlayCanvas 2.x (loaded from jsDelivr via import map), fetches the
  catalog, renders a scene-picker, wires controls, manages WebXR entry,
  and exposes a registry for experiments. See `components/splat-scene.js`.
- `components/hamburger-menu.js` — side-effect module; auto-attaches a
  hamburger button + settings/config panel to the active `<splat-scene>`.

### Experiments

Experiments register themselves with the scene and appear as toggles in the
hamburger menu. Each implements `enable() / disable() / renderSettings(host)`.

#### Virtual blind cane (`experiments/blind-cane.js`)

A 1.2 m cane is attached to each WebXR controller (or to the camera in flat
mode for testing). Each frame, a ray is cast along the cane's length
against an approximated collision volume. On contact:

- the shaft is shortened to the hit distance (fake "bending"),
- a short filtered noise burst is played as a 3D-positioned tap,
- the controller's haptic actuator pulses
  (`gamepad.hapticActuators[0].pulse(...)`),
- while still in contact, motion-driven low-amp scrub audio + haptic.

**Known approximations** — the collision volume is currently a single
configurable AABB ("room"). For real scene geometry, generate a
`.collision.glb` via `splat-transform -K` (per the
[PlayCanvas blog](https://blog.playcanvas.com/turning-a-gaussian-splat-into-a-videogame/))
and swap `_raycastBox` for a triangle test. No rigid-body simulator is
wired — the cane is kinematic with length clamping. Adequate to *feel* a
wall; not enough to slide weight along a surface.

#### WubWub — acoustic chromatic aberrations (`experiments/wubwub.js`)

Audio source (mic by default, or upload a file) is run through an
`AnalyserNode`. Per frame:

- bass / mid / treble energy is extracted from the FFT bin distribution,
- an SVG `<filter>` with channel-split `feColorMatrix` + per-channel
  `feOffset` drives a **true chromatic aberration** on the splat canvas
  (CSS `filter: url(#wub-chromab)`),
- the splat entity's local scale is modulated non-uniformly per axis
  (bass = squash, treble = stretch, mid = depth) — **squash/stretch
  of the cloud as a whole**.

The "per-particle position deformation" interpretation would require
injecting a chunk into PlayCanvas's gsplat material vertex shader. The
chunk override API is invasive and not portable across engine minor
versions, so this first cut sticks to entity scaling. Upgrade path noted
in the source.

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
