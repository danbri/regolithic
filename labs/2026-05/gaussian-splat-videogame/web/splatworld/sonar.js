// Sonar — submarine-style ping that maps the distance to the nearest
// SplatWorld primitive in front of the camera into:
//   • ping frequency  (near → high, far → low)
//   • ping interval   (near → fast, far → slow)
//   • amplitude       (slightly softer at range)
//
// Out of range (no hit within maxRange m) → silent, polled at ~2 s.
//
// Needs SplatWorld mesh detection ON to have anything to hit.

const MIN_FREQ = 500;          // Hz at maxRange
const MAX_FREQ = 1500;         // Hz at 0 m
const MIN_INTERVAL = 0.10;     // s between pings at 0 m
const MAX_INTERVAL = 1.6;      // s between pings at maxRange

export class Sonar {
  constructor(scene, world) {
    this.scene = scene;
    this.world = world;
    this.maxRange = 5;
    this.volume = 0.45;
    this.running = false;
    this._audioCtx = null;
    this._nextPingAt = 0;
    this._handlers = {};
  }

  async start() {
    if (this.running) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('Web Audio not supported in this browser');
    this._audioCtx = new AC();
    // iOS often suspends AudioContext until a user gesture; resume()
    // after the click that flipped this toggle on.
    if (this._audioCtx.state === 'suspended') this._audioCtx.resume();
    this._nextPingAt = 0;
    this._handlers.update = () => this._tick();
    this.scene.app.on('update', this._handlers.update);
    this.running = true;
  }

  stop() {
    if (!this.running) return;
    if (this._handlers.update) this.scene.app.off('update', this._handlers.update);
    try { this._audioCtx?.close(); } catch {}
    this._audioCtx = null;
    this._handlers = {};
    this.running = false;
  }

  _tick() {
    const ctx = this._audioCtx;
    if (!ctx) return;
    const now = ctx.currentTime;
    if (now < this._nextPingAt) return;

    if (!this.world?.enabled || !this.world.primitives?.length) {
      // No mesh to hit; poll slowly.
      this._nextPingAt = now + 2;
      return;
    }

    const cam = this.scene.camera;
    const origin = cam.getPosition();
    const fwd = cam.forward;
    const hit = this.world.raycast(origin, fwd, this.maxRange);
    if (!hit) {
      this._nextPingAt = now + 2;
      return;
    }

    const t = Math.min(1, hit.t / this.maxRange);
    const interval = MIN_INTERVAL + (MAX_INTERVAL - MIN_INTERVAL) * t * t;
    const freq = MAX_FREQ + (MIN_FREQ - MAX_FREQ) * t;
    const vol = this.volume * (1 - t * 0.5);
    this._ping(freq, vol);
    this._nextPingAt = now + interval;
  }

  _ping(freq, vol) {
    const ctx = this._audioCtx;
    const dur = 0.08;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    // Brief downward chirp gives the classic sonar timbre.
    osc.frequency.exponentialRampToValueAtTime(freq * 0.7, t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.006);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.connect(g).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + dur);
  }
}
