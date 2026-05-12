// SplatWorld — derive a world model from the current Gaussian splat
// (mesh primitives + camera-attached cane behaviours).
//
// Mesh detection is gated: it doesn't run until the user explicitly
// asks. On enable, we read the splat's AABB and synthesise a small
// set of typed primitives ("ground", "wall", "obstacle") inside it.
// This is an honest approximation — a real solution would voxelise
// the splat density (à la `splat-transform -K`) into a watertight
// mesh; that's deferred. The primitives shape the audio palette used
// by the phone-sensor cane.

import * as pc from 'playcanvas';

export class SplatWorld extends EventTarget {
  constructor(scene) {
    super();
    this.scene = scene;
    this.enabled = false;
    this.primitives = [];      // [{ id, type, center: Vec3, halfExtents: Vec3 }]
    this._helperEntities = [];
    this._listenerKey = null;

    // Re-detect on scene swap if previously enabled.
    scene.addEventListener('scene-loaded', () => {
      this._removeHelpers();
      this.primitives = [];
      if (this.enabled) this.detect();
      this.dispatchEvent(new CustomEvent('changed'));
    });
  }

  // Make a best-effort scan of the current splat and emit primitives.
  // Repeatable. No-op if no scene loaded.
  detect() {
    const ent = this.scene.splatEntity;
    if (!ent) {
      this.primitives = [];
      this.enabled = true;
      this.dispatchEvent(new CustomEvent('changed'));
      return this.primitives;
    }
    const aabb = this._readSplatAABB(ent);
    this.primitives = this._synthesisePrimitives(aabb);
    this.enabled = true;
    this._removeHelpers();
    this._buildHelpers();
    this.dispatchEvent(new CustomEvent('changed'));
    return this.primitives;
  }

  disable() {
    this.enabled = false;
    this.primitives = [];
    this._removeHelpers();
    this.dispatchEvent(new CustomEvent('changed'));
  }

  setHelpersVisible(visible) {
    for (const e of this._helperEntities) e.enabled = !!visible;
  }

  // ── Internals ────────────────────────────────────────────────────────
  _readSplatAABB(ent) {
    // Try the engine's reported aabb first; fall back to a default room.
    const candidates = [
      ent.gsplat?.customAabb,
      ent.gsplat?.instance?.meshInstance?.aabb,
      ent.gsplat?.instance?.aabb,
    ];
    for (const a of candidates) {
      if (a?.center && a?.halfExtents) {
        // Convert to world space using the entity transform
        const c = new pc.Vec3(a.center.x, a.center.y, a.center.z);
        const h = new pc.Vec3(a.halfExtents.x, a.halfExtents.y, a.halfExtents.z);
        const m = ent.getWorldTransform();
        m.transformPoint(c, c);
        const s = ent.getLocalScale();
        h.set(h.x * Math.abs(s.x), h.y * Math.abs(s.y), h.z * Math.abs(s.z));
        return { center: c, halfExtents: h, source: 'splat' };
      }
    }
    return {
      center: ent.getPosition().clone(),
      halfExtents: new pc.Vec3(2, 1.5, 2),
      source: 'fallback',
    };
  }

  _synthesisePrimitives(aabb) {
    const c = aabb.center, h = aabb.halfExtents;
    // Clamp wildly oversized bounds (some splats have outlier gaussians
    // that blow the AABB to ridiculous values; cap at 50 m per axis so
    // the audio cone of interest stays sensible).
    const cap = (v) => Math.min(Math.abs(v), 50) * Math.sign(v || 1);
    const hx = cap(h.x), hy = cap(h.y), hz = cap(h.z);

    return [
      // Floor — broad flat surface; characteristic sound = "screech"
      {
        id: 'ground',
        type: 'ground',
        sound: 'screech',
        center: new pc.Vec3(c.x, c.y - hy * 0.95, c.z),
        halfExtents: new pc.Vec3(hx, Math.max(0.05, hy * 0.05), hz),
      },
      // A wall — flat tall; "screech" too, but mid-pitched
      {
        id: 'wall-x',
        type: 'wall',
        sound: 'screech-mid',
        center: new pc.Vec3(c.x + hx * 0.95, c.y, c.z),
        halfExtents: new pc.Vec3(Math.max(0.05, hx * 0.05), hy, hz),
      },
      // An obstacle — small cube near the camera; "thunk"
      {
        id: 'obstacle',
        type: 'obstacle',
        sound: 'thunk',
        center: new pc.Vec3(c.x, c.y, c.z + hz * 0.3),
        halfExtents: new pc.Vec3(
          Math.max(0.1, Math.min(0.5, hx * 0.1)),
          Math.max(0.1, Math.min(0.5, hy * 0.1)),
          Math.max(0.1, Math.min(0.5, hz * 0.1)),
        ),
      },
      // A tiny "pip" target — a small protrusion the cane can find
      {
        id: 'pip',
        type: 'pip',
        sound: 'pip',
        center: new pc.Vec3(c.x - hx * 0.3, c.y + hy * 0.4, c.z + hz * 0.2),
        halfExtents: new pc.Vec3(0.1, 0.1, 0.1),
      },
    ];
  }

  _buildHelpers() {
    const app = this.scene.app;
    for (const p of this.primitives) {
      const e = new pc.Entity(`world-${p.id}`);
      e.addComponent('render', { type: 'box' });
      e.setPosition(p.center);
      e.setLocalScale(p.halfExtents.x * 2, p.halfExtents.y * 2, p.halfExtents.z * 2);
      const mat = e.render?.material;
      if (mat?.diffuse) {
        const c = HELPER_COLORS[p.type] ?? [0.6, 0.6, 0.6];
        mat.diffuse.set(c[0], c[1], c[2]);
        if (mat.opacity != null) {
          mat.opacity = 0.18;
          mat.blendType = pc.BLEND_NORMAL;
        }
        mat.update?.();
      }
      e.enabled = false; // hidden by default — toggle via setHelpersVisible
      app.root.addChild(e);
      this._helperEntities.push(e);
    }
  }

  _removeHelpers() {
    for (const e of this._helperEntities) e.destroy();
    this._helperEntities = [];
  }

  // Ray-vs-primitive test (slab). Returns nearest hit { t, prim, point }.
  raycast(origin, direction, maxLength = 2) {
    let best = null;
    for (const p of this.primitives) {
      const t = this._intersectBox(origin, direction, p, maxLength);
      if (t != null && (!best || t < best.t)) {
        best = {
          t,
          prim: p,
          point: new pc.Vec3(
            origin.x + direction.x * t,
            origin.y + direction.y * t,
            origin.z + direction.z * t,
          ),
        };
      }
    }
    return best;
  }

  // Minimum distance from a point to any primitive surface (approx, via
  // each AABB). Used for proximity rumble before contact.
  nearestDistance(point) {
    let best = Infinity;
    for (const p of this.primitives) {
      const d = this._distanceToBox(point, p);
      if (d < best) best = d;
    }
    return best;
  }

  _intersectBox(o, d, prim, maxLength) {
    const min = [
      prim.center.x - prim.halfExtents.x,
      prim.center.y - prim.halfExtents.y,
      prim.center.z - prim.halfExtents.z,
    ];
    const max = [
      prim.center.x + prim.halfExtents.x,
      prim.center.y + prim.halfExtents.y,
      prim.center.z + prim.halfExtents.z,
    ];
    const oo = [o.x, o.y, o.z], dd = [d.x, d.y, d.z];
    let tNear = -Infinity, tFar = Infinity;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(dd[i]) < 1e-9) {
        if (oo[i] < min[i] || oo[i] > max[i]) return null;
      } else {
        let t1 = (min[i] - oo[i]) / dd[i];
        let t2 = (max[i] - oo[i]) / dd[i];
        if (t1 > t2) [t1, t2] = [t2, t1];
        tNear = Math.max(tNear, t1);
        tFar  = Math.min(tFar, t2);
        if (tNear > tFar) return null;
      }
    }
    if (tNear < 0) tNear = 0;
    if (tNear > maxLength) return null;
    return tNear;
  }

  _distanceToBox(p, prim) {
    const dx = Math.max(0, Math.abs(p.x - prim.center.x) - prim.halfExtents.x);
    const dy = Math.max(0, Math.abs(p.y - prim.center.y) - prim.halfExtents.y);
    const dz = Math.max(0, Math.abs(p.z - prim.center.z) - prim.halfExtents.z);
    return Math.hypot(dx, dy, dz);
  }
}

const HELPER_COLORS = {
  ground:   [0.4, 0.7, 0.4],
  wall:     [0.7, 0.5, 0.4],
  obstacle: [0.9, 0.7, 0.3],
  pip:      [0.4, 0.6, 0.9],
};
