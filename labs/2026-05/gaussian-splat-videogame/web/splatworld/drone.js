// Drone — autonomous wander + object-detection sweep of a Gaussian
// splat scene.
//
// Flight model: a virtual hoverdrone with continuous forward motion in
// the direction of (heading, pitch). Smooth heading-drift gives it a
// "curious" wander. Short-range raycasts against SplatWorld primitives
// pull it away from obstacles; AABB clamp keeps it within the scene.
//
// Survey model: every `surveyInterval` seconds, snapshot the current
// frame and feed it to a TransformersJSModel in mode 'object-detection'
// (YOLOS Tiny, DETR, OWL-ViT). Detected labels join a growing
// `discovered` map with their best score. New labels fire a
// 'detection' event so the UI can announce them.

import * as pc from 'playcanvas';

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

export class Drone extends EventTarget {
  constructor(scene, world) {
    super();
    this.scene = scene;
    this.world = world;

    // Tunables (also exposed to the menu sliders)
    this.flightSpeed = 0.8;        // m/s
    this.turnRate = 60;            // deg/s — max yaw change per second
    this.lookAhead = 0.8;          // m — forward raycast distance
    this.surveyInterval = 2.5;     // s between detections
    this.wanderJitter = 1.0;       // 0..1 — how curiously it drifts

    // State
    this.running = false;
    this.discovered = new Map();   // label → { count, bestScore, lastSeenMs }
    this.model = null;
    this._pos = new pc.Vec3();
    this._heading = 0;             // degrees, world-frame yaw
    this._pitch = 0;
    this._targetHeadingOffset = 0;
    this._aabb = null;
    this._lastSurveyAt = 0;
    this._wanderResetAt = 0;
    this._surveyBusy = false;
    this._handlers = {};
  }

  async start(model) {
    if (this.running) return;
    if (!model || model.mode !== 'object-detection') {
      throw new Error('Drone needs an object-detection model (e.g. YOLOS Tiny)');
    }
    const ent = this.scene.splatEntity;
    if (!ent) throw new Error('Load a scene first');

    this.model = model;
    this._aabb = this._readAABB(ent);
    // Spawn the drone near the AABB centre, looking down the +Z axis.
    this._pos.set(this._aabb.center.x, this._aabb.center.y, this._aabb.center.z);
    this._heading = Math.random() * 360;
    this._pitch = 0;
    this._targetHeadingOffset = 0;
    this._lastSurveyAt = 0;
    this._wanderResetAt = 0;
    this._surveyBusy = false;

    // Make sure the model is loaded before starting; tell listeners.
    this.dispatchEvent(new CustomEvent('status', { detail: { text: 'Initialising detector…' } }));
    await this.model.ensureReady((p) => {
      this.dispatchEvent(new CustomEvent('status', {
        detail: { text: `Loading model: ${p.text || ''}` },
      }));
    });

    this.scene.pushMode('drone');
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

  resetSurvey() {
    this.discovered.clear();
    this.dispatchEvent(new CustomEvent('changed'));
  }

  // ── Flight ───────────────────────────────────────────────────────────
  _tick(dt) {
    // 1. Wander: every couple seconds pick a new heading offset target.
    const now = performance.now() / 1000;
    if (now - this._wanderResetAt > 2.5) {
      this._wanderResetAt = now;
      this._targetHeadingOffset = (Math.random() - 0.5) * 90 * this.wanderJitter;
      this._pitch = (Math.random() - 0.5) * 20 * this.wanderJitter; // mild bob
    }
    // Smooth-steer toward target offset
    const desired = this._heading + this._targetHeadingOffset * dt * 0.5;
    this._heading = smoothApproach(this._heading, desired, this.turnRate * dt);
    this._pitch = Math.max(-25, Math.min(25, this._pitch));

    // 2. Compute forward vector
    const fwd = this._forward();

    // 3. Avoidance: short forward raycast against SplatWorld primitives.
    if (this.world?.enabled && this.world.primitives.length) {
      const hit = this.world.raycast(this._pos, fwd, this.lookAhead);
      if (hit && hit.t < this.lookAhead * 0.7) {
        // Pick a turn direction away from the obstacle's centre.
        const dx = hit.prim.center.x - this._pos.x;
        const dz = hit.prim.center.z - this._pos.z;
        const obstAngle = Math.atan2(dx, dz) * R2D;
        const headDelta = normaliseAngle(this._heading - obstAngle);
        // Turn whichever way takes us further from the obstacle
        this._heading += (headDelta >= 0 ? 1 : -1) * 90 * dt * 4;
        this._targetHeadingOffset = 0;
      }
    }

    // 4. AABB containment: if drifting outside, point back toward centre.
    const margin = 0.5;
    const a = this._aabb;
    const out = (
      this._pos.x < a.center.x - a.halfExtents.x + margin ||
      this._pos.x > a.center.x + a.halfExtents.x - margin ||
      this._pos.y < a.center.y - a.halfExtents.y + margin ||
      this._pos.y > a.center.y + a.halfExtents.y - margin ||
      this._pos.z < a.center.z - a.halfExtents.z + margin ||
      this._pos.z > a.center.z + a.halfExtents.z - margin
    );
    if (out) {
      const dx = a.center.x - this._pos.x;
      const dz = a.center.z - this._pos.z;
      const back = Math.atan2(dx, dz) * R2D;
      this._heading = smoothApproach(this._heading, back, this.turnRate * 1.5 * dt);
      this._pitch = (a.center.y - this._pos.y) * 5; // tilt back toward centre
    }

    // 5. Translate
    const f = this._forward();
    this._pos.x += f.x * this.flightSpeed * dt;
    this._pos.y += f.y * this.flightSpeed * dt;
    this._pos.z += f.z * this.flightSpeed * dt;

    // 6. Apply camera pose (look-at = position + forward)
    this.scene.setCameraFromPose(
      { x: this._pos.x, y: this._pos.y, z: this._pos.z },
      { x: this._pos.x + f.x, y: this._pos.y + f.y, z: this._pos.z + f.z },
    );

    // 7. Periodic detection sweep
    if (!this._surveyBusy && now - this._lastSurveyAt > this.surveyInterval) {
      this._lastSurveyAt = now;
      this._runSurvey();
    }
  }

  _forward() {
    const h = this._heading * D2R, p = this._pitch * D2R;
    return new pc.Vec3(
      Math.sin(h) * Math.cos(p),
      Math.sin(p),
      Math.cos(h) * Math.cos(p),
    );
  }

  // ── Survey ───────────────────────────────────────────────────────────
  async _runSurvey() {
    if (this._surveyBusy) return;
    this._surveyBusy = true;
    try {
      const blob = await this.scene.captureSnapshot();
      const bitmap = await createImageBitmap(blob);
      const detections = await this.model.detect(bitmap);
      const now = Date.now();
      const newLabels = [];
      for (const d of detections) {
        const cur = this.discovered.get(d.label);
        if (!cur) {
          this.discovered.set(d.label, { count: 1, bestScore: d.score, lastSeenMs: now });
          newLabels.push(d.label);
        } else {
          cur.count += 1;
          cur.bestScore = Math.max(cur.bestScore, d.score);
          cur.lastSeenMs = now;
        }
      }
      this.dispatchEvent(new CustomEvent('survey', {
        detail: { detections, newLabels, total: this.discovered.size },
      }));
      if (newLabels.length) {
        this.dispatchEvent(new CustomEvent('detection', { detail: { newLabels } }));
      }
      this.dispatchEvent(new CustomEvent('changed'));
    } catch (err) {
      // Don't break the flight loop on a transient inference error
      console.warn('[drone] survey failed:', err);
    } finally {
      this._surveyBusy = false;
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────
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
        const cap = (v) => Math.min(Math.abs(v), 50) * Math.sign(v || 1);
        return {
          center: { x: c.x, y: c.y, z: c.z },
          halfExtents: {
            x: cap(h.x * Math.abs(s.x)),
            y: cap(h.y * Math.abs(s.y)),
            z: cap(h.z * Math.abs(s.z)),
          },
        };
      }
    }
    const p = ent.getPosition();
    return {
      center: { x: p.x, y: p.y, z: p.z },
      halfExtents: { x: 3, y: 2, z: 3 },
    };
  }

  surveySummary() {
    if (this.discovered.size === 0) return '';
    const entries = [...this.discovered.entries()]
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 12);
    return entries.map(([l, d]) => d.count > 1 ? `${l} ×${d.count}` : l).join(' · ');
  }
}

function smoothApproach(cur, target, maxStep) {
  const diff = normaliseAngle(target - cur);
  if (Math.abs(diff) <= maxStep) return cur + diff;
  return cur + Math.sign(diff) * maxStep;
}

function normaliseAngle(deg) {
  let a = deg % 360;
  if (a > 180) a -= 360;
  if (a < -180) a += 360;
  return a;
}
