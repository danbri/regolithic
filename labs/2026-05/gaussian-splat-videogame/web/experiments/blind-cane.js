// Virtual blind cane.
//
// On WebXR, attach a 1.2m "cane" to each controller. Cast a ray along the
// cane's length each frame against an approximated collision mesh (currently
// a configurable AABB "room" around the splat); on contact, shorten the
// cane to the hit point, pulse the controller's haptic actuator, and play
// a 3D tap sound at the tip.
//
// On flat (mobile/desktop), simulate one cane forward from the camera so
// the UI/audio can be exercised without a headset.
//
// Approximations & known gaps:
//   • Collision geometry is a single axis-aligned bounding box. To swap in
//     a real mesh, load the splat's `.collision.glb` sibling (produced by
//     `splat-transform -K`) and replace `_raycastBox` with a triangle test.
//   • No "true" rigid body: the cane is kinematic. Bending under load is
//     faked via length clamping. Good enough to feel a wall; not enough
//     to slide along a surface with weight.

import * as pc from 'playcanvas';

const CANE_LENGTH = 1.2; // metres
const CANE_RADIUS = 0.008;
const ROOM = { min: [-5, -1, -5], max: [5, 3, 5] }; // metres, AABB fallback
const TAP_COOLDOWN = 0.08; // s, debounce contact onset

export class BlindCane {
  constructor(scene) {
    this.scene = scene;
    this._enabled = false;
    this._canes = new Map(); // inputSource -> { entity, tip, lastHit, lastTapAt }
    this._flatCane = null;
    this._audioCtx = null;
    this._listenerHooked = false;
    this._handlers = {};
  }

  enable() {
    const app = this.scene.app;
    if (!app) return;
    this._ensureAudio();

    // Hook XR controllers
    const xr = app.xr;
    if (xr) {
      this._handlers.add = (input) => this._spawnCane(input);
      this._handlers.remove = (input) => this._removeCane(input);
      xr.input.on('add', this._handlers.add);
      xr.input.on('remove', this._handlers.remove);
      for (const input of xr.input.inputSources) this._spawnCane(input);
    }

    // Flat-mode fallback: one cane forward of camera
    if (!xr?.active) this._spawnFlatCane();

    // Per-frame update
    this._handlers.update = (dt) => this._tick(dt);
    app.on('update', this._handlers.update);

    // Resume audio on the next user gesture (browsers gate it)
    this._handlers.resumeAudio = () => this._audioCtx?.resume();
    window.addEventListener('pointerdown', this._handlers.resumeAudio, { once: true });
  }

  disable() {
    const app = this.scene.app;
    if (!app) return;
    if (this._handlers.add)    app.xr?.input.off('add', this._handlers.add);
    if (this._handlers.remove) app.xr?.input.off('remove', this._handlers.remove);
    if (this._handlers.update) app.off('update', this._handlers.update);
    if (this._handlers.resumeAudio) window.removeEventListener('pointerdown', this._handlers.resumeAudio);
    for (const { entity } of this._canes.values()) entity.destroy();
    this._canes.clear();
    if (this._flatCane) { this._flatCane.entity.destroy(); this._flatCane = null; }
    this._handlers = {};
  }

  renderSettings(host) {
    const note = document.createElement('div');
    note.className = 'credit';
    note.innerHTML = `Approximated as raycast against an AABB room
      (${ROOM.min.join(', ')}) → (${ROOM.max.join(', ')}). Swap in a real
      mesh by loading a sibling <code>.collision.glb</code>.`;
    host.appendChild(note);
  }

  // ── Cane lifecycle ───────────────────────────────────────────────────
  _spawnCane(inputSource) {
    if (this._canes.has(inputSource)) return;
    const entity = this._buildCaneEntity(`cane-${inputSource.handedness || 'unknown'}`);
    this.scene.app.root.addChild(entity);
    this._canes.set(inputSource, {
      entity,
      lastHit: null,
      lastTapAt: 0,
      contactPhase: 0,
    });
  }

  _removeCane(inputSource) {
    const c = this._canes.get(inputSource);
    if (!c) return;
    c.entity.destroy();
    this._canes.delete(inputSource);
  }

  _spawnFlatCane() {
    if (this._flatCane) return;
    const entity = this._buildCaneEntity('cane-flat');
    this.scene.app.root.addChild(entity);
    this._flatCane = { entity, lastHit: null, lastTapAt: 0, contactPhase: 0 };
  }

  _buildCaneEntity(name) {
    // Parent entity = controller pose; child cylinder = visible cane,
    // oriented along -Z (controller forward in WebXR convention).
    const root = new pc.Entity(name);
    const shaft = new pc.Entity('shaft');
    shaft.addComponent('render', { type: 'cylinder' });
    shaft.setLocalScale(CANE_RADIUS * 2, CANE_LENGTH, CANE_RADIUS * 2);
    // Cylinder is Y-up; rotate so it lies along controller -Z, with grip at parent origin.
    shaft.setLocalEulerAngles(90, 0, 0);
    shaft.setLocalPosition(0, 0, -CANE_LENGTH / 2);

    // Tint white-ish; PlayCanvas primitives use the default material.
    const m = shaft.render.material;
    if (m && m.diffuse) {
      m.diffuse.set(0.95, 0.95, 0.95);
      m.update?.();
    }

    root.addChild(shaft);
    root.__caneShaft = shaft;
    root.__caneLength = CANE_LENGTH;
    return root;
  }

  // ── Per-frame ────────────────────────────────────────────────────────
  _tick(dt) {
    const app = this.scene.app;
    const xrActive = !!app.xr?.active;

    // Sync XR cane poses
    for (const [inputSource, state] of this._canes) {
      if (inputSource.grip) {
        state.entity.setPosition(inputSource.getPosition());
        state.entity.setRotation(inputSource.getRotation());
        this._updateCane(state, inputSource, dt);
      }
    }

    // Flat-mode cane follows camera, offset down-right
    if (!xrActive && this._flatCane) {
      const cam = this.scene.camera;
      const camRot = cam.getRotation();
      const camPos = cam.getPosition();
      const offset = new pc.Vec3(0.2, -0.25, 0);
      camRot.transformVector(offset, offset);
      this._flatCane.entity.setPosition(camPos.x + offset.x, camPos.y + offset.y, camPos.z + offset.z);
      this._flatCane.entity.setRotation(camRot);
      this._updateCane(this._flatCane, null, dt);
    }
  }

  _updateCane(state, inputSource, dt) {
    // Cast ray from grip along entity's local -Z (cane forward)
    const ent = state.entity;
    const origin = ent.getPosition();
    const forward = ent.forward; // entity-forward is local -Z in PC convention
    const len = CANE_LENGTH;

    const hitT = this._raycastBox(origin, forward, len);
    if (hitT != null) {
      // Shorten the visual shaft to the hit distance
      ent.__caneShaft.setLocalScale(CANE_RADIUS * 2, hitT, CANE_RADIUS * 2);
      ent.__caneShaft.setLocalPosition(0, 0, -hitT / 2);

      const now = performance.now() / 1000;
      const wasContact = state.contactPhase > 0;
      state.contactPhase = Math.min(1, state.contactPhase + dt * 12);

      // Onset = sharp tap
      if (!wasContact && now - state.lastTapAt > TAP_COOLDOWN) {
        state.lastTapAt = now;
        this._tap(origin.clone().add(forward.clone().mulScalar(hitT)), 1.0);
        this._pulse(inputSource, 0.85, 35);
      } else if (wasContact) {
        // Continuous low-amplitude rumble while in contact (motion-driven)
        const speedSq = (state._prev ?? origin).distance(origin) ** 2 / Math.max(dt, 1e-3);
        state._prev = origin.clone();
        if (speedSq > 0.0005) {
          this._scrub(origin.clone().add(forward.clone().mulScalar(hitT)), Math.min(1, speedSq * 50));
          this._pulse(inputSource, Math.min(0.5, speedSq * 200), 16);
        }
      }
    } else {
      // No contact — restore full length, decay phase
      ent.__caneShaft.setLocalScale(CANE_RADIUS * 2, len, CANE_RADIUS * 2);
      ent.__caneShaft.setLocalPosition(0, 0, -len / 2);
      state.contactPhase = Math.max(0, state.contactPhase - dt * 8);
      state._prev = null;
    }
  }

  // ── Geometry ─────────────────────────────────────────────────────────
  // Slab/box raycast against ROOM AABB (interior hits — i.e. ray from
  // inside the box hitting the inner surface). Returns t in metres or null.
  _raycastBox(origin, dir, maxLen) {
    let tNear = -Infinity, tFar = Infinity;
    const o = [origin.x, origin.y, origin.z];
    const d = [dir.x, dir.y, dir.z];
    for (let i = 0; i < 3; i++) {
      if (Math.abs(d[i]) < 1e-8) {
        if (o[i] < ROOM.min[i] || o[i] > ROOM.max[i]) return null;
        continue;
      }
      let t1 = (ROOM.min[i] - o[i]) / d[i];
      let t2 = (ROOM.max[i] - o[i]) / d[i];
      if (t1 > t2) [t1, t2] = [t2, t1];
      tNear = Math.max(tNear, t1);
      tFar  = Math.min(tFar,  t2);
      if (tNear > tFar) return null;
    }
    const t = tFar > 0 ? tFar : null; // first positive exit = inner surface
    if (t == null || t > maxLen) return null;
    return t;
  }

  // ── Audio ────────────────────────────────────────────────────────────
  _ensureAudio() {
    if (this._audioCtx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this._audioCtx = new AC();
  }

  _tap(worldPos, gain) {
    const ctx = this._audioCtx;
    if (!ctx) return;
    const dur = 0.06;
    const buffer = ctx.createBuffer(1, Math.ceil(dur * ctx.sampleRate), ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      const t = i / data.length;
      data[i] = (Math.random() * 2 - 1) * Math.exp(-t * 30);
    }
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 1500;
    const g = ctx.createGain();
    g.gain.value = gain;
    const panner = ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.positionX.value = worldPos.x;
    panner.positionY.value = worldPos.y;
    panner.positionZ.value = worldPos.z;
    src.connect(hp).connect(g).connect(panner).connect(ctx.destination);
    src.start();
  }

  _scrub(worldPos, gain) {
    const ctx = this._audioCtx;
    if (!ctx) return;
    const dur = 0.05;
    const buffer = ctx.createBuffer(1, Math.ceil(dur * ctx.sampleRate), ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.3;
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 800;
    bp.Q.value = 0.7;
    const g = ctx.createGain();
    g.gain.value = gain * 0.4;
    const panner = ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.positionX.value = worldPos.x;
    panner.positionY.value = worldPos.y;
    panner.positionZ.value = worldPos.z;
    src.connect(bp).connect(g).connect(panner).connect(ctx.destination);
    src.start();
  }

  _pulse(inputSource, intensity, durationMs) {
    const gp = inputSource?.gamepad;
    const act = gp?.hapticActuators?.[0];
    if (!act) return;
    if (typeof act.pulse === 'function') {
      try { act.pulse(intensity, durationMs); } catch {}
    } else if (typeof act.playEffect === 'function') {
      try { act.playEffect('dual-rumble', { duration: durationMs, strongMagnitude: intensity, weakMagnitude: intensity }); } catch {}
    }
  }
}
