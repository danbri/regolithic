// Phone-sensor cane.
//
// Maps DeviceOrientation (alpha/beta/gamma) to a cane attached to the
// scene camera. Each frame: raycasts along the cane against
// SplatWorld primitives; on contact, fires a sound chosen by the
// primitive's `sound` tag (screech / screech-mid / thunk / pip) plus
// navigator.vibrate haptics. A continuous low rumble swells as the
// cane tip approaches *any* primitive surface — proximity feedback
// before contact.
//
// iOS 13+ requires DeviceOrientationEvent.requestPermission() from a
// user gesture. Caller funnels Set-up clicks through start().

import * as pc from 'playcanvas';

const CANE_LENGTH = 1.0;
const CANE_RADIUS = 0.008;
const PROXIMITY_RANGE = 0.6; // metres; rumble fades over this distance
const TAP_REFRACTORY_MS = 90;

export class PhoneCane {
  constructor(scene, world) {
    this.scene = scene;
    this.world = world;
    this._running = false;
    this._orient = { alpha: 0, beta: 90, gamma: 0 };  // phone held upright
    this._entity = null;
    this._shaft = null;
    this._audioCtx = null;
    this._rumble = null;
    this._rumbleGain = null;
    this._lastTapAt = 0;
    this._inContactPrim = null;
    this._handlers = {};
  }

  async start() {
    if (this._running) return;
    // iOS gesture-gated permission
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') {
      const r = await DeviceOrientationEvent.requestPermission();
      if (r !== 'granted') throw new Error('Orientation permission denied');
    }
    // Use 'deviceorientationabsolute' if available for compass-aligned alpha
    const hasAbs = typeof DeviceOrientationEvent !== 'undefined' &&
      ('ondeviceorientationabsolute' in window);
    this._handlers.orient = (e) => {
      this._orient = { alpha: e.alpha ?? 0, beta: e.beta ?? 0, gamma: e.gamma ?? 0 };
    };
    window.addEventListener(hasAbs ? 'deviceorientationabsolute' : 'deviceorientation', this._handlers.orient);
    this._ensureAudio();
    this._buildCane();
    this._handlers.update = (dt) => this._tick(dt);
    this.scene.app.on('update', this._handlers.update);
    this._running = true;
  }

  stop() {
    if (!this._running) return;
    if (this._handlers.orient) {
      window.removeEventListener('deviceorientationabsolute', this._handlers.orient);
      window.removeEventListener('deviceorientation', this._handlers.orient);
    }
    if (this._handlers.update) this.scene.app.off('update', this._handlers.update);
    this._entity?.destroy();
    this._entity = this._shaft = null;
    try { this._rumble?.stop(); } catch {}
    this._rumble = this._rumbleGain = null;
    try { this._audioCtx?.close(); } catch {}
    this._audioCtx = null;
    this._handlers = {};
    this._running = false;
  }

  get running() { return this._running; }

  // ── Internals ────────────────────────────────────────────────────────
  _buildCane() {
    const root = new pc.Entity('phone-cane');
    const shaft = new pc.Entity('shaft');
    shaft.addComponent('render', { type: 'cylinder' });
    shaft.setLocalScale(CANE_RADIUS * 2, CANE_LENGTH, CANE_RADIUS * 2);
    shaft.setLocalEulerAngles(90, 0, 0); // align cylinder Y-up → -Z forward
    shaft.setLocalPosition(0, 0, -CANE_LENGTH / 2);
    const mat = shaft.render?.material;
    if (mat?.diffuse) {
      mat.diffuse.set(0.95, 0.85, 0.4);
      mat.update?.();
    }
    root.addChild(shaft);
    this.scene.app.root.addChild(root);
    this._entity = root;
    this._shaft = shaft;
  }

  _ensureAudio() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this._audioCtx = new AC();
    // Continuous proximity rumble — gain starts at 0, modulated each frame
    const osc = this._audioCtx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = 55;
    const filter = this._audioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 220;
    const gain = this._audioCtx.createGain();
    gain.gain.value = 0;
    osc.connect(filter).connect(gain).connect(this._audioCtx.destination);
    osc.start();
    this._rumble = osc;
    this._rumbleGain = gain;
  }

  _tick(dt) {
    // 1. Phone orientation → cane pose. We anchor the cane in the camera's
    //    local frame: in front of the head, slightly down-right, then
    //    twist by the device tilt.
    const cam = this.scene.camera;
    const camPos = cam.getPosition();
    const camRot = cam.getRotation();

    // Hand offset in camera space
    const handOffset = new pc.Vec3(0.18, -0.25, -0.3);
    camRot.transformVector(handOffset, handOffset);
    const grip = new pc.Vec3(camPos.x + handOffset.x, camPos.y + handOffset.y, camPos.z + handOffset.z);
    this._entity.setPosition(grip);

    // Convert phone orientation to a local rotation relative to camera.
    // beta ~ pitch (front-back), gamma ~ roll (left-right), alpha ~ yaw.
    // Phone-flat reads beta≈0; phone-upright reads beta≈90. We subtract
    // 90 so "phone tilted toward you" = cane pointing forward.
    const pitchDeg = (this._orient.beta - 90);
    const yawDeg   = this._orient.gamma * 1.5;
    const rollDeg  = 0;
    const local = new pc.Quat().setFromEulerAngles(pitchDeg, yawDeg, rollDeg);
    const combined = new pc.Quat().mul2(camRot, local);
    this._entity.setRotation(combined);

    if (!this.world?.enabled) return;

    // 2. Raycast along cane forward
    const origin = this._entity.getPosition();
    const dir = this._entity.forward;
    const hit = this.world.raycast(origin, dir, CANE_LENGTH);
    if (hit) {
      // Shrink visible shaft to hit
      this._shaft.setLocalScale(CANE_RADIUS * 2, hit.t, CANE_RADIUS * 2);
      this._shaft.setLocalPosition(0, 0, -hit.t / 2);

      const now = performance.now();
      if (this._inContactPrim !== hit.prim.id && now - this._lastTapAt > TAP_REFRACTORY_MS) {
        this._lastTapAt = now;
        this._fireHitSound(hit.prim);
        this._haptic(hit.prim);
      }
      this._inContactPrim = hit.prim.id;
    } else {
      this._shaft.setLocalScale(CANE_RADIUS * 2, CANE_LENGTH, CANE_RADIUS * 2);
      this._shaft.setLocalPosition(0, 0, -CANE_LENGTH / 2);
      this._inContactPrim = null;
    }

    // 3. Proximity rumble — gain rises as nearest surface comes within
    //    PROXIMITY_RANGE of the cane tip.
    if (this._rumbleGain) {
      const tip = new pc.Vec3(
        origin.x + dir.x * CANE_LENGTH,
        origin.y + dir.y * CANE_LENGTH,
        origin.z + dir.z * CANE_LENGTH,
      );
      const d = this.world.nearestDistance(tip);
      const target = Math.max(0, 1 - d / PROXIMITY_RANGE);
      const cur = this._rumbleGain.gain.value;
      // smoothly approach target
      this._rumbleGain.gain.value = cur + (target * 0.15 - cur) * 0.2;
    }
  }

  _fireHitSound(prim) {
    const ctx = this._audioCtx;
    if (!ctx) return;
    const now = ctx.currentTime;
    if (prim.sound === 'pip') {
      // High sine burst, very short
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = 1800;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.45, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
      o.connect(g).connect(ctx.destination);
      o.start(now); o.stop(now + 0.07);
    } else if (prim.sound === 'thunk') {
      // Low-freq sine + filtered noise burst
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = 110;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.7, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
      o.connect(g).connect(ctx.destination);
      o.start(now); o.stop(now + 0.25);
      this._noiseBurst(0.12, 600, 0.35);
    } else if (prim.sound === 'screech') {
      // Bandpass sweep up — flat surface scrape
      this._sweep(180, 1400, 0.35, 0.55);
    } else if (prim.sound === 'screech-mid') {
      this._sweep(400, 2200, 0.28, 0.5);
    } else {
      // Generic tap
      this._noiseBurst(0.05, 1200, 0.5);
    }
  }

  _noiseBurst(dur, centerHz, gain) {
    const ctx = this._audioCtx;
    const buf = ctx.createBuffer(1, Math.ceil(dur * ctx.sampleRate), ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      const t = i / data.length;
      data[i] = (Math.random() * 2 - 1) * Math.exp(-t * 18);
    }
    const src = ctx.createBufferSource(); src.buffer = buf;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.value = centerHz; bp.Q.value = 0.8;
    const g = ctx.createGain(); g.gain.value = gain;
    src.connect(bp).connect(g).connect(ctx.destination);
    src.start();
  }

  _sweep(startHz, endHz, dur, gain) {
    const ctx = this._audioCtx;
    const now = ctx.currentTime;
    const buf = ctx.createBuffer(1, Math.ceil(dur * ctx.sampleRate), ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.7;
    const src = ctx.createBufferSource(); src.buffer = buf;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 2.5;
    bp.frequency.setValueAtTime(startHz, now);
    bp.frequency.exponentialRampToValueAtTime(endHz, now + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + dur);
    src.connect(bp).connect(g).connect(ctx.destination);
    src.start();
  }

  _haptic(prim) {
    if (!navigator.vibrate) return;
    const pat = {
      pip:          [8],
      thunk:        [40],
      screech:      [25, 15, 25],
      'screech-mid':[20, 10, 20],
    }[prim.sound] ?? [15];
    try { navigator.vibrate(pat); } catch {}
  }
}
