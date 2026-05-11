// WubWub — acoustic chromatic aberrations.
//
// FFT of an audio input (mic or uploaded file) drives:
//   • a 3-channel chromatic-aberration SVG filter where R, G, B each
//     offset in their own direction (default 120° apart) with amplitude
//     driven by bass / mid / treble respectively,
//   • stacked CSS filters (hue-rotate, saturate, contrast) on the splat
//     canvas, so peaks shift the entire colour space too,
//   • per-axis non-uniform scale on the splat entity (squash/stretch).
//
// Implementation choices:
//   • SVG filter is the cheapest way to get per-channel pixel offset in
//     the browser without writing a postprocess shader.
//   • Whole-entity scaling stands in for per-particle position
//     deformation; an upgrade path is a GSplat material chunk override.

import * as pc from 'playcanvas';

const FFT_SIZE = 1024;
const SMOOTHING = 0.55;
const TAU = Math.PI * 2;

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
    this._chromAmount = 32;    // max per-channel offset in CSS px
    this._hueAmount = 180;     // max hue rotation in deg
    this._satAmount = 1.5;     // max +saturation (1 + this)
    this._contrastAmount = 0.6;// max +contrast (1 + this)
    this._squashAmount = 0.25; // max per-axis scale deviation
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
    // just `audio/*`. Explicit extensions + wildcard fixes it.
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

    host.appendChild(slider('Intensity',       0, 3,   0.01, this._intensity,       v => this._intensity = v));
    host.appendChild(slider('Chrom. abb. px',  0, 150, 1,    this._chromAmount,     v => this._chromAmount = v));
    host.appendChild(slider('Hue rotate deg',  0, 360, 1,    this._hueAmount,       v => this._hueAmount = v));
    host.appendChild(slider('Saturation +',    0, 4,   0.05, this._satAmount,       v => this._satAmount = v));
    host.appendChild(slider('Contrast +',      0, 2,   0.05, this._contrastAmount,  v => this._contrastAmount = v));
    host.appendChild(slider('Squash/stretch',  0, 0.8, 0.01, this._squashAmount,    v => this._squashAmount = v));
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
    this._analyser.connect(this._audioCtx.destination);
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
    const bass   = avg(this._buf, 0,                           Math.floor(bins * 0.05)) / 255;
    const mid    = avg(this._buf, Math.floor(bins * 0.05),     Math.floor(bins * 0.3))  / 255;
    const treble = avg(this._buf, Math.floor(bins * 0.3),      bins) / 255;

    const k = this._intensity;
    const A = this._chromAmount * k;

    // Per-channel offsets, each in its own static direction (120° apart),
    // with amplitude driven by that channel's band.
    const ampR = bass   * A;
    const ampG = mid    * A;
    const ampB = treble * A;
    // 0°, 120°, 240° relative to +X. Negate Y for screen-space (down positive).
    const aR = 0;
    const aG = TAU / 3;
    const aB = 2 * TAU / 3;
    if (this._filterR && this._filterG && this._filterB) {
      this._filterR.setAttribute('dx', (ampR * Math.cos(aR)).toFixed(2));
      this._filterR.setAttribute('dy', (ampR * Math.sin(aR)).toFixed(2));
      this._filterG.setAttribute('dx', (ampG * Math.cos(aG)).toFixed(2));
      this._filterG.setAttribute('dy', (ampG * Math.sin(aG)).toFixed(2));
      this._filterB.setAttribute('dx', (ampB * Math.cos(aB)).toFixed(2));
      this._filterB.setAttribute('dy', (ampB * Math.sin(aB)).toFixed(2));
    }

    // Stack CSS filters on top of the SVG channel offset.
    const hue = mid    * this._hueAmount      * k;
    const sat = 1 + bass * this._satAmount    * k;
    const con = 1 + treble * this._contrastAmount * k;
    document.documentElement.style.setProperty(
      '--wub-filter',
      `hue-rotate(${hue.toFixed(1)}deg) saturate(${sat.toFixed(2)}) contrast(${con.toFixed(2)}) url(#wub-chromab)`,
    );

    const ent = this.scene.splatEntity;
    if (ent) {
      const s = this._baseScale;
      const sq = this._squashAmount * k;
      const sx = s.x * (1 + bass   * sq);
      const sy = s.y * (1 - bass   * sq * 0.6 + treble * sq * 0.6);
      const sz = s.z * (1 + mid    * sq * 0.4);
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
    // Per-channel split via feColorMatrix preserving alpha, each channel
    // independently offset, then re-composited with screen (additive on
    // disjoint channels). The wide filter region accommodates large
    // chromatic offsets without clipping at the canvas edge.
    svg.innerHTML = `
      <filter id="wub-chromab" x="-25%" y="-25%" width="150%" height="150%" color-interpolation-filters="sRGB">
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
        <feOffset id="wub-g-off" in="g" dx="0" dy="0" result="go"/>
        <feColorMatrix type="matrix" values="
          0 0 0 0 0
          0 0 0 0 0
          0 0 1 0 0
          0 0 0 1 0" result="b"/>
        <feOffset id="wub-b-off" in="b" dx="0" dy="0" result="bo"/>
        <feBlend in="ro" in2="go" mode="screen" result="rg"/>
        <feBlend in="rg" in2="bo" mode="screen"/>
      </filter>`;
    document.body.appendChild(svg);
    this._filterEl = svg;
    this._filterR = svg.querySelector('#wub-r-off');
    this._filterG = svg.querySelector('#wub-g-off');
    this._filterB = svg.querySelector('#wub-b-off');
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
