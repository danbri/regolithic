// Tour — fly the camera through a spiral around the active splat,
// pointing at each major SplatWorld primitive in turn.
//
// Pure camera scripting; no image capture, no AI. Confirm-the-flight
// MVP — those hooks come once the trajectory is right.
//
// Waypoint generation: spiral outward from the splat's AABB centre,
// 1.5 turns, radius rising from 0.3×maxExtent → 2×maxExtent, height
// from -0.1× → +0.5× of maxExtent. Look-at targets cycle through
// SplatWorld primitives if any are detected, else just track the
// splat centre.

import * as pc from 'playcanvas';

const D2R = Math.PI / 180;

export class Tour extends EventTarget {
  constructor(scene, world) {
    super();
    this.scene = scene;
    this.world = world;
    this.running = false;
    this.waypoints = [];
    this.idx = 0;
    this.t = 0;                    // 0..1 progress within current step
    this.stepDuration = 2.5;       // seconds per waypoint segment
    this.dwellDuration = 0.4;      // seconds held at each waypoint
    this._handlers = {};
    this._dwellRemaining = 0;
    this._fromPos = null;
    this._fromLook = null;

    // Re-build the trajectory whenever the scene changes (if running).
    scene.addEventListener('scene-loaded', () => {
      if (this.running) {
        this.stop();
      }
    });
  }

  start() {
    if (this.running) return;
    this.waypoints = this._buildWaypoints();
    if (this.waypoints.length === 0) {
      throw new Error('Tour needs a loaded scene (load one from Catalog first).');
    }
    this.idx = 0;
    this.t = 0;
    this._dwellRemaining = 0;
    // Seed the "from" pose with where the camera is RIGHT now so the
    // first segment eases in from current view.
    const cam = this.scene.camera;
    const camPos = cam.getPosition();
    const fwd = cam.forward;
    this._fromPos = { x: camPos.x, y: camPos.y, z: camPos.z };
    // Project a look-at point ~3 m ahead of where the camera currently looks.
    this._fromLook = {
      x: camPos.x + fwd.x * 3,
      y: camPos.y + fwd.y * 3,
      z: camPos.z + fwd.z * 3,
    };

    this.scene.pushMode('tour');
    this._handlers.update = (dt) => this._tick(dt);
    this.scene.app.on('update', this._handlers.update);
    this.running = true;
    this.dispatchEvent(new CustomEvent('changed'));
  }

  stop() {
    if (!this.running) return;
    if (this._handlers.update) this.scene.app.off('update', this._handlers.update);
    this.scene.popMode();
    this.running = false;
    this._handlers = {};
    this.dispatchEvent(new CustomEvent('changed'));
  }

  // Jump to the next/previous waypoint immediately (no eased transition).
  next() {
    if (!this.running) return;
    this.idx = Math.min(this.waypoints.length - 1, this.idx + 1);
    this.t = 0;
    this._dwellRemaining = 0;
    this._snapToCurrentAsFrom();
    this.dispatchEvent(new CustomEvent('changed'));
  }
  prev() {
    if (!this.running) return;
    this.idx = Math.max(0, this.idx - 1);
    this.t = 0;
    this._dwellRemaining = 0;
    this._snapToCurrentAsFrom();
    this.dispatchEvent(new CustomEvent('changed'));
  }

  get currentWaypoint() { return this.waypoints[this.idx] ?? null; }

  // ── Internals ────────────────────────────────────────────────────────
  _snapToCurrentAsFrom() {
    const cam = this.scene.camera;
    const p = cam.getPosition();
    const fwd = cam.forward;
    this._fromPos = { x: p.x, y: p.y, z: p.z };
    this._fromLook = { x: p.x + fwd.x * 3, y: p.y + fwd.y * 3, z: p.z + fwd.z * 3 };
  }

  _tick(dt) {
    if (this.idx >= this.waypoints.length) { this.stop(); return; }

    if (this._dwellRemaining > 0) {
      this._dwellRemaining = Math.max(0, this._dwellRemaining - dt);
      const wp = this.waypoints[this.idx];
      this.scene.setCameraFromPose(wp.pos, wp.lookAt);
      return;
    }

    this.t += dt / Math.max(0.05, this.stepDuration);
    if (this.t >= 1) {
      // Reached the target waypoint. Hold for dwellDuration, then advance.
      this.t = 1;
      const wp = this.waypoints[this.idx];
      this.scene.setCameraFromPose(wp.pos, wp.lookAt);
      this._dwellRemaining = this.dwellDuration;
      this.dispatchEvent(new CustomEvent('waypoint-reached', { detail: { idx: this.idx, waypoint: wp } }));
      // Schedule advance on the next dwell-expiry tick.
      this._fromPos = { ...wp.pos };
      this._fromLook = { ...wp.lookAt };
      this.idx++;
      this.t = 0;
      if (this.idx >= this.waypoints.length) {
        // Stay on the final waypoint for the dwell, then stop.
        const finalize = () => this.stop();
        setTimeout(finalize, this.dwellDuration * 1000);
        return;
      }
      this.dispatchEvent(new CustomEvent('changed'));
      return;
    }

    const wp = this.waypoints[this.idx];
    const e = smoothstep(this.t);
    const pos = lerp3(this._fromPos, wp.pos, e);
    const look = lerp3(this._fromLook, wp.lookAt, e);
    this.scene.setCameraFromPose(pos, look);
  }

  _buildWaypoints() {
    const ent = this.scene.splatEntity;
    if (!ent) return [];

    const aabb = this._readAABB(ent);
    const c = aabb.center;
    const maxExtent = Math.max(aabb.halfExtents.x, aabb.halfExtents.y, aabb.halfExtents.z);

    const primitives = this.world?.primitives ?? [];
    const targets = primitives.length > 0
      ? primitives.map(p => ({ id: p.id ?? p.type ?? 'prim', center: p.center }))
      : [{ id: 'centre', center: c }];

    // Spiral parameters
    const turns = 1.5;
    const r0 = Math.max(0.5, maxExtent * 0.3);
    const r1 = Math.max(2,   maxExtent * 2);
    const h0 = -maxExtent * 0.1;
    const h1 =  maxExtent * 0.5;
    const N = Math.max(targets.length * 2, 6);

    const wps = [];
    for (let i = 0; i < N; i++) {
      const u = N > 1 ? i / (N - 1) : 0;
      const angle = u * Math.PI * 2 * turns;
      const r = r0 + (r1 - r0) * u;
      const h = h0 + (h1 - h0) * u;
      const target = targets[i % targets.length];
      const tc = target.center;
      wps.push({
        idx: i,
        pos: {
          x: c.x + r * Math.cos(angle),
          y: c.y + h,
          z: c.z + r * Math.sin(angle),
        },
        lookAt: { x: tc.x, y: tc.y, z: tc.z },
        label: target.id,
      });
    }
    return wps;
  }

  _readAABB(ent) {
    const candidates = [
      ent.gsplat?.customAabb,
      ent.gsplat?.instance?.meshInstance?.aabb,
      ent.gsplat?.instance?.aabb,
    ];
    for (const a of candidates) {
      if (a?.center && a?.halfExtents) {
        const c = new pc.Vec3(a.center.x, a.center.y, a.center.z);
        const h = new pc.Vec3(a.halfExtents.x, a.halfExtents.y, a.halfExtents.z);
        ent.getWorldTransform().transformPoint(c, c);
        const s = ent.getLocalScale();
        h.set(h.x * Math.abs(s.x), h.y * Math.abs(s.y), h.z * Math.abs(s.z));
        // Clamp outlier-blown AABBs (rare in scrubby splats)
        const cap = (v) => Math.min(Math.abs(v), 50) * Math.sign(v || 1);
        return { center: { x: c.x, y: c.y, z: c.z },
                 halfExtents: { x: cap(h.x), y: cap(h.y), z: cap(h.z) } };
      }
    }
    return {
      center: ent.getPosition().clone(),
      halfExtents: { x: 2, y: 1.5, z: 2 },
    };
  }
}

function lerp(a, b, t) { return a + (b - a) * t; }
function lerp3(a, b, t) { return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t) }; }
function smoothstep(t) { return t * t * (3 - 2 * t); }
