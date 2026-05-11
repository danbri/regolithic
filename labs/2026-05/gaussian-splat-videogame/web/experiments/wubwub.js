// WubWub — acoustic chromatic aberrations.
//
// FFT of an audio input (mic or uploaded file) drives:
//   • a chromatic-aberration SVG filter applied to the splat canvas
//     (R and B channels offset proportionally to bass + treble)
//   • a per-axis non-uniform scale on the splat entity (squash/stretch
//     in time with bass / mid / treble bands)
//
// Implementation choices:
//   • SVG filter is the cheapest way to get true per-channel pixel offset
//     in the browser without writing a postprocess shader.
//   • Whole-entity scaling stands in for per-particle position deformation.
//     A more faithful version would inject a chunk into PlayCanvas's
//     gsplat material vertex shader, but the chunk override API is invasive
//     and not portable across engine minor versions; whole-entity scale
//     is a robust first cut.

import * as pc from 'playcanvas';

const FFT_SIZE = 1024;
const SMOOTHING = 0.6;

export class WubWub {
  constructor(scene) {
    this.scene = scene;
    this._enabled = false;
    this._audioCtx = null;
    this._analyser = null;
    this._source = null;
    this._buf = null;
    this._sourceLabel = 'idle';
    this._gain = 1.0;
    this._intensity = 1.0;
    this._chromAmount = 6; // max channel offset in CSS pixels
    this._squashAmount = 0.15; // max per-axis scale deviation
    this._handlers = {};
    this._filterEl = null;
    this._baseScale = new pc.Vec3(1, 1, 1);
  }

  enable() {
    this._installFilter();
    this._handlers.update = () => this._tick();
    this.scene.app.on('update', this._handlers.update);
    if (this.scene.splatEntity) this._baseScale.copy(this.scene.splatEntity.getLocalScale());
    this.scene.addEventListener('scene-loaded', this._onSceneLoaded = () => {
      if (this.scene.splatEntity) this._baseScale.copy(this.scene.splatEntity.getLocalScale());
    });
    // If no source picked yet, default to mic on first user gesture
    this._handlers.kick = async () => {
      if (this._source) return;
      await this._useMic().catch(err => console.warn('[wubwub] mic denied:', err));
    };
    window.addEventListener('pointerdown', this._handlers.kick, { once: true });
  }

  disable() {
    if (this._handlers.update) this.scene.app.off('update', this._handlers.update);
    if (this._onSceneLoaded) this.scene.removeEventListener('scene-loaded', this._onSceneLoaded);
    if (this._handlers.kick) window.removeEventListener('pointerdown', this._handlers.kick);
    this._handlers = {};
    document.documentElement.style.setProperty('--wub-filter', 'none');
    if (this.scene.splatEntity) this.scene.splatEntity.setLocalScale(this._baseScale);
    this._teardownSource();
    if (this._audioCtx) { this._audioCtx.close().catch(() => {}); this._audioCtx = null; }
  }

  renderSettings(host) {
    const sourceRow = document.createElement('div');
    sourceRow.className = 'row';
    const lbl = document.createElement('span');
    lbl.textContent = `Source: ${this._sourceLabel}`;
    sourceRow.appendChild(lbl);

    const micBtn = document.createElement('button');
    micBtn.textContent = 'Mic';
    micBtn.addEventListener('click', () => this._useMic().then(() => lbl.textContent = `Source: ${this._sourceLabel}`));
    sourceRow.appendChild(micBtn);

    const fileBtn = document.createElement('button');
    fileBtn.textContent = 'File…';
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    // iOS Safari's Files picker greys out audio files when `accept` is
    // just `audio/*`. Combining explicit extensions with the wildcard
    // restores normal behaviour without breaking desktop browsers.
    fileInput.accept = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus,audio/*';
    fileInput.style.display = 'none';
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files?.[0];
      if (f) {
        await this._useFile(f);
        lbl.textContent = `Source: ${this._sourceLabel}`;
      }
    });
    fileBtn.addEventListener('click', () => fileInput.click());
    sourceRow.appendChild(fileBtn);
    sourceRow.appendChild(fileInput);
    host.appendChild(sourceRow);

    host.appendChild(slider('Intensity', 0, 2, 0.01, this._intensity, v => this._intensity = v));
    host.appendChild(slider('Chrom. abb. px', 0, 24, 0.5, this._chromAmount, v => this._chromAmount = v));
    host.appendChild(slider('Squash/stretch', 0, 0.6, 0.01, this._squashAmount, v => this._squashAmount = v));
  }

  // ── Sources ──────────────────────────────────────────────────────────
  async _ensureAudio() {
    if (this._audioCtx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('Web Audio not supported');
    this._audioCtx = new AC();
    this._analyser = this._audioCtx.createAnalyser();
    this._analyser.fftSize = FFT_SIZE;
    this._analyser.smoothingTimeConstant = SMOOTHING;
    this._buf = new Uint8Array(this._analyser.frequencyBinCount);
  }

  async _useMic() {
    await this._ensureAudio();
    this._teardownSource();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const src = this._audioCtx.createMediaStreamSource(stream);
    src.connect(this._analyser);
    this._source = { node: src, stop: () => stream.getTracks().forEach(t => t.stop()) };
    this._sourceLabel = 'mic';
  }

  async _useFile(file) {
    await this._ensureAudio();
    this._teardownSource();
    const buf = await file.arrayBuffer();
    const audioBuf = await this._audioCtx.decodeAudioData(buf);
    const src = this._audioCtx.createBufferSource();
    src.buffer = audioBuf;
    src.loop = true;
    src.connect(this._analyser);
    this._analyser.connect(this._audioCtx.destination); // monitor playback
    src.start();
    this._source = { node: src, stop: () => { try { src.stop(); } catch {} } };
    this._sourceLabel = file.name;
  }

  _teardownSource() {
    if (!this._source) return;
    try { this._source.node.disconnect(); } catch {}
    try { this._source.stop(); } catch {}
    this._source = null;
    this._sourceLabel = 'idle';
  }

  // ── Per-frame ────────────────────────────────────────────────────────
  _tick() {
    if (!this._analyser || !this._buf) return;
    this._analyser.getByteFrequencyData(this._buf);

    const bins = this._buf.length;
    const bass   = avg(this._buf, 0,            Math.floor(bins * 0.05)) / 255;
    const mid    = avg(this._buf, Math.floor(bins * 0.05), Math.floor(bins * 0.3)) / 255;
    const treble = avg(this._buf, Math.floor(bins * 0.3),  bins) / 255;

    const k = this._intensity;
    const chrom = (bass + treble) * 0.5 * this._chromAmount * k;
    const offR = chrom;
    const offB = -chrom * 0.9;
    if (this._filterRChan && this._filterBChan) {
      this._filterRChan.setAttribute('dx', String(offR.toFixed(2)));
      this._filterBChan.setAttribute('dx', String(offB.toFixed(2)));
    }
    document.documentElement.style.setProperty('--wub-filter', 'url(#wub-chromab)');

    const ent = this.scene.splatEntity;
    if (ent) {
      const s = this._baseScale;
      const sq = this._squashAmount * k;
      const sx = s.x * (1 + bass * sq);
      const sy = s.y * (1 - bass * sq * 0.6 + treble * sq * 0.6);
      const sz = s.z * (1 + mid * sq * 0.4);
      ent.setLocalScale(sx, sy, sz);
    }
  }

  // ── SVG filter ───────────────────────────────────────────────────────
  _installFilter() {
    if (this._filterEl) return;
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('style', 'position:absolute;width:0;height:0;pointer-events:none');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = `
      <filter id="wub-chromab" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
        <feColorMatrix type="matrix" values="
          1 0 0 0 0
          0 0 0 0 0
          0 0 0 0 0
          0 0 0 1 0" result="r"/>
        <feOffset id="wub-r-off" in="r" dx="0" dy="0" result="ro"/>
        <feColorMatrix type="matrix" values="
          0 0 0 0 0
          0 1 0 0 0
          0 0 0 0 0
          0 0 0 1 0" result="g"/>
        <feColorMatrix type="matrix" values="
          0 0 0 0 0
          0 0 0 0 0
          0 0 1 0 0
          0 0 0 1 0" result="b"/>
        <feOffset id="wub-b-off" in="b" dx="0" dy="0" result="bo"/>
        <feBlend in="ro" in2="g" mode="screen" result="rg"/>
        <feBlend in="rg" in2="bo" mode="screen"/>
      </filter>`;
    document.body.appendChild(svg);
    this._filterEl = svg;
    this._filterRChan = svg.querySelector('#wub-r-off');
    this._filterBChan = svg.querySelector('#wub-b-off');
  }
}

function avg(buf, lo, hi) {
  if (hi <= lo) return 0;
  let s = 0;
  for (let i = lo; i < hi; i++) s += buf[i];
  return s / (hi - lo);
}

function slider(label, min, max, step, value, onChange) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('span');
  lbl.textContent = `${label}: ${value.toFixed(2)}`;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    lbl.textContent = `${label}: ${v.toFixed(2)}`;
    onChange(v);
  });
  row.appendChild(lbl);
  row.appendChild(input);
  return row;
}
