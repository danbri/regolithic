// WubWub — acoustic chromatic aberrations.
//
// Audio features extracted per frame (via AnalyserNode FFT):
//   • band energies         — bass / low-mid / mid / high
//   • spectral flux         — sum of positive bin-deltas; "is the
//                              spectrum changing?" (good for chromatic
//                              shimmer intensity)
//   • spectral centroid     — energy-weighted mean frequency bin;
//                              "is the sound bright or dark?" (drives hue)
//   • beat detection        — running mean + std of sub-bass energy;
//                              a beat is `bass > μ + kσ` with refractory
//                              period. Drives a 1.0 → 0 exponential
//                              "beat-energy" envelope.
//   • BPM estimate          — median interval between the last N beats.
//
// These drive: per-channel chromatic offsets, hue rotation, saturation,
// contrast, and a non-uniform splat-entity scale that pulses on beats.

import * as pc from 'playcanvas';

const FFT_SIZE = 2048;
const SMOOTHING = 0.4;                // more reactive than 0.6
const HIST_LEN = 64;                  // ~1s @ 60Hz; for running stats
const BEAT_REFRACTORY_MS = 220;       // min gap between beats (≈270 BPM ceiling)
const BEAT_THRESHOLD_K = 1.3;         // beat = bass > mean + k * stddev
const BEAT_MIN_LEVEL = 0.12;          // ignore quiet floor
const BEAT_DECAY = 0.055;             // per-frame exponential decay
const TAU = Math.PI * 2;

export class WubWub {
  constructor(scene) {
    this.scene = scene;
    this._enabled = false;
    this._audioCtx = null;
    this._analyser = null;
    this._source = null;
    this._buf = null;
    this._prevBuf = null;
    this._sourceLabel = 'idle';

    // Tunables
    this._intensity = 1.0;
    this._chromAmount = 40;
    this._hueAmount = 220;
    this._satAmount = 1.6;
    this._contrastAmount = 0.7;
    this._squashAmount = 0.6;          // bumped: was 0.28
    this._beatPulseAmount = 0.7;       // bumped: was 0.45
    this._jitterAmount = 0.04;         // entity-position shake on beats (m)
    this._flashAmount = 0.7;           // clearColor flash on beats (0..1)
    this._fovPunch = 6;                // degrees of FOV punch on beats

    // Features (smoothed)
    this._flux = 0;
    this._centroid = 0;
    this._loud = 0;
    this._bassHist = new Float32Array(HIST_LEN);
    this._bassHistN = 0;
    this._bassHistIdx = 0;
    this._beatEnergy = 0;
    this._lastBeatAt = 0;
    this._beatTimes = [];
    this._bpm = 0;
    this._beatCount = 0;

    this._handlers = {};
    this._filterEl = null;
    this._beatDot = null;
    this._bpmLabel = null;
    this._baseScale = new pc.Vec3(1, 1, 1);
    this._basePosition = new pc.Vec3(0, 0, 0);
    this._baseClearColor = null;
    this._baseFov = 70;
  }

  enable() {
    this._installFilter();
    this._handlers.update = () => this._tick();
    this.scene.app.on('update', this._handlers.update);
    this._captureBase();
    this.scene.addEventListener('scene-loaded', this._onSceneLoaded = () => {
      this._captureBase();
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
    if (this.scene.splatEntity) {
      this.scene.splatEntity.setLocalScale(this._baseScale);
      this.scene.splatEntity.setLocalPosition(this._basePosition);
    }
    // Restore camera clearColor + fov
    const cam = this.scene.camera?.camera;
    if (cam) {
      if (this._baseClearColor) cam.clearColor = this._baseClearColor;
      cam.fov = this._baseFov;
    }
    this._teardownSource();
    if (this._audioCtx) { this._audioCtx.close().catch(() => {}); this._audioCtx = null; }
  }

  _captureBase() {
    const ent = this.scene.splatEntity;
    if (ent) {
      this._baseScale.copy(ent.getLocalScale());
      this._basePosition.copy(ent.getLocalPosition());
    }
    const cam = this.scene.camera?.camera;
    if (cam) {
      this._baseClearColor = cam.clearColor.clone();
      this._baseFov = cam.fov;
    }
  }

  renderSettings(host) {
    // Source row
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
    fileInput.accept = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus,audio/*';
    fileInput.style.display = 'none';
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files?.[0];
      if (f) { await this._useFile(f); lbl.textContent = `Source: ${this._sourceLabel}`; }
    });
    fileBtn.addEventListener('click', () => fileInput.click());
    sourceRow.appendChild(fileBtn);
    sourceRow.appendChild(fileInput);
    host.appendChild(sourceRow);

    // Beat indicator + BPM readout
    const beatRow = document.createElement('div');
    beatRow.className = 'row';
    const dot = document.createElement('span');
    dot.textContent = '●';
    dot.style.cssText = 'color:#666; font-size:18px; transition:color 100ms ease-out, transform 100ms ease-out; display:inline-block;';
    const bpm = document.createElement('span');
    bpm.textContent = 'beat — BPM —';
    bpm.style.flex = '1';
    beatRow.appendChild(dot);
    beatRow.appendChild(bpm);
    host.appendChild(beatRow);
    this._beatDot = dot;
    this._bpmLabel = bpm;

    host.appendChild(slider('Intensity',        0, 3,   0.01, this._intensity,       v => this._intensity = v));
    host.appendChild(slider('Chrom. abb. px',   0, 150, 1,    this._chromAmount,     v => this._chromAmount = v));
    host.appendChild(slider('Hue rotate deg',   0, 360, 1,    this._hueAmount,       v => this._hueAmount = v));
    host.appendChild(slider('Saturation +',     0, 4,   0.05, this._satAmount,       v => this._satAmount = v));
    host.appendChild(slider('Contrast +',       0, 2,   0.05, this._contrastAmount,  v => this._contrastAmount = v));
    host.appendChild(slider('Squash/stretch',   0, 0.8, 0.01, this._squashAmount,    v => this._squashAmount = v));
    host.appendChild(slider('Beat pulse',       0, 1.5, 0.01, this._beatPulseAmount, v => this._beatPulseAmount = v));
    host.appendChild(slider('Jitter (m)',       0, 0.3, 0.005, this._jitterAmount,   v => this._jitterAmount = v));
    host.appendChild(slider('Flash',            0, 1,   0.01, this._flashAmount,     v => this._flashAmount = v));
    host.appendChild(slider('FOV punch (deg)',  0, 30,  0.5,  this._fovPunch,        v => this._fovPunch = v));
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
    this._prevBuf = new Uint8Array(this._analyser.frequencyBinCount);
  }

  async _useMic() {
    await this._ensureAudio();
    this._teardownSource();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
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

  // ── Feature extraction ───────────────────────────────────────────────
  _extractFeatures() {
    const buf = this._buf;
    const bins = buf.length;
    const sr = this._audioCtx.sampleRate;
    const binHz = sr / FFT_SIZE;

    // Bin ranges for musically-meaningful bands (Hz → bin)
    const binsFor = (loHz, hiHz) => [
      Math.max(0, Math.floor(loHz / binHz)),
      Math.min(bins, Math.ceil(hiHz / binHz)),
    ];
    const [sb0, sb1] = binsFor(40, 180);    // sub-bass / kick
    const [b0, b1]   = binsFor(40, 250);    // bass
    const [m0, m1]   = binsFor(250, 2000);  // mid
    const [t0, t1]   = binsFor(2000, 8000); // treble

    const subbass = avg(buf, sb0, sb1) / 255;
    const bass    = avg(buf, b0, b1)   / 255;
    const mid     = avg(buf, m0, m1)   / 255;
    const treble  = avg(buf, t0, t1)   / 255;
    const loud    = avg(buf, 0, bins)  / 255;
    this._loud = this._loud * 0.6 + loud * 0.4;

    // Spectral flux: Σ max(0, buf[i] - prev[i]) / bins, normalised 0..1
    let flux = 0;
    for (let i = 0; i < bins; i++) {
      const d = buf[i] - this._prevBuf[i];
      if (d > 0) flux += d;
    }
    flux = (flux / bins) / 255;
    this._flux = this._flux * 0.65 + flux * 0.35;
    this._prevBuf.set(buf);

    // Spectral centroid: Σ i·buf[i] / Σ buf[i], normalised 0..1
    let total = 0, weighted = 0;
    for (let i = 0; i < bins; i++) {
      total += buf[i];
      weighted += i * buf[i];
    }
    const centroid = total > 0 ? (weighted / total) / bins : 0;
    this._centroid = this._centroid * 0.5 + centroid * 0.5;

    // Beat detection on sub-bass: rolling mean & stddev over HIST_LEN
    // frames; beat = subbass > μ + k·σ, debounced.
    this._bassHist[this._bassHistIdx] = subbass;
    this._bassHistIdx = (this._bassHistIdx + 1) % HIST_LEN;
    if (this._bassHistN < HIST_LEN) this._bassHistN++;

    let mean = 0;
    for (let i = 0; i < this._bassHistN; i++) mean += this._bassHist[i];
    mean /= this._bassHistN;
    let varSum = 0;
    for (let i = 0; i < this._bassHistN; i++) {
      const d = this._bassHist[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / this._bassHistN);

    const now = performance.now();
    let beatFired = false;
    if (
      this._bassHistN > 20 &&
      subbass > BEAT_MIN_LEVEL &&
      subbass > mean + BEAT_THRESHOLD_K * std &&
      now - this._lastBeatAt > BEAT_REFRACTORY_MS
    ) {
      this._lastBeatAt = now;
      this._beatEnergy = 1.0;
      this._beatCount++;
      this._beatTimes.push(now);
      if (this._beatTimes.length > 8) this._beatTimes.shift();
      // BPM from median inter-beat interval
      if (this._beatTimes.length >= 4) {
        const dt = [];
        for (let i = 1; i < this._beatTimes.length; i++) dt.push(this._beatTimes[i] - this._beatTimes[i - 1]);
        dt.sort((a, b) => a - b);
        const med = dt[Math.floor(dt.length / 2)];
        if (med > 0) this._bpm = 60000 / med;
      }
      beatFired = true;
    }
    this._beatEnergy = Math.max(0, this._beatEnergy - BEAT_DECAY);
    // Drop BPM if no beats for >3s
    if (now - this._lastBeatAt > 3000) { this._bpm = 0; this._beatTimes.length = 0; }

    return { subbass, bass, mid, treble, beatFired };
  }

  // ── Per-frame ────────────────────────────────────────────────────────
  _tick() {
    if (!this._analyser || !this._buf) return;
    this._analyser.getByteFrequencyData(this._buf);
    const f = this._extractFeatures();
    const k = this._intensity;
    const A = this._chromAmount * k;
    const be = this._beatEnergy;

    // Per-channel chromatic offset, in directions 120° apart. Amplitude
    // mixes the channel's natural band with shared signals so even quiet
    // mids/treble pick up motion from flux + beats.
    const ampR = (f.bass   + be * 0.7) * A;
    const ampG = (f.mid    + this._flux * 1.5) * A * 0.8;
    const ampB = (f.treble + be * 0.4 + this._flux * 0.6) * A;
    const aR = 0, aG = TAU / 3, aB = 2 * TAU / 3;
    if (this._filterR && this._filterG && this._filterB) {
      this._filterR.setAttribute('dx', (ampR * Math.cos(aR)).toFixed(2));
      this._filterR.setAttribute('dy', (ampR * Math.sin(aR)).toFixed(2));
      this._filterG.setAttribute('dx', (ampG * Math.cos(aG)).toFixed(2));
      this._filterG.setAttribute('dy', (ampG * Math.sin(aG)).toFixed(2));
      this._filterB.setAttribute('dx', (ampB * Math.cos(aB)).toFixed(2));
      this._filterB.setAttribute('dy', (ampB * Math.sin(aB)).toFixed(2));
    }

    // Hue from centroid (brightness of sound), with a kick on every beat.
    const hue = (this._centroid * this._hueAmount + be * 60) * k;
    const sat = 1 + (this._loud + be * 0.4) * this._satAmount    * k;
    const con = 1 + (f.treble + be * 0.3)   * this._contrastAmount * k;
    document.documentElement.style.setProperty(
      '--wub-filter',
      `hue-rotate(${hue.toFixed(1)}deg) saturate(${sat.toFixed(2)}) contrast(${con.toFixed(2)}) url(#wub-chromab)`,
    );

    // Splat entity scale: clean band → axis mapping.
    //   bass   → X widens
    //   mid    → Z deepens
    //   treble → Y stretches (per user request: "high notes should
    //            stretch vertically")
    // Beat punches all three outward, plus jitters position to shake
    // the cloud and reveal its granular structure.
    const ent = this.scene.splatEntity;
    if (ent) {
      const s = this._baseScale;
      const sq = this._squashAmount * k;
      const pulse = be * this._beatPulseAmount * k;
      const sx = s.x * (1 + f.bass   * sq * 1.5 + pulse);
      const sy = s.y * (1 + f.treble * sq * 2.5 + pulse);
      const sz = s.z * (1 + f.mid    * sq * 1.2 + pulse);
      ent.setLocalScale(sx, sy, sz);

      // Position jitter on beats — exposes the particle nature by
      // motion-parallax against the dark background.
      const j = be * this._jitterAmount * k;
      const b = this._basePosition;
      ent.setLocalPosition(
        b.x + (Math.random() - 0.5) * j,
        b.y + (Math.random() - 0.5) * j,
        b.z + (Math.random() - 0.5) * j,
      );
    }

    // Camera clear-color flash on beats — even when the splat is dim,
    // the empty background flashes, so you can see the audio is being
    // heard. Hue follows centroid.
    const camComp = this.scene.camera?.camera;
    if (camComp && this._baseClearColor) {
      const flash = be * this._flashAmount * k;
      // Map centroid 0..1 → hue 0..360
      const h = this._centroid * 360;
      const [fr, fg, fb] = hslToRgb(h, 0.9, 0.55);
      const cc = camComp.clearColor;
      cc.r = this._baseClearColor.r + (fr - this._baseClearColor.r) * flash;
      cc.g = this._baseClearColor.g + (fg - this._baseClearColor.g) * flash;
      cc.b = this._baseClearColor.b + (fb - this._baseClearColor.b) * flash;
      // FOV punch on beats — gives a "thwack" zoom
      camComp.fov = this._baseFov - be * this._fovPunch * k;
    }

    // UI feedback for beat detection
    if (this._beatDot) {
      const heat = Math.min(1, be + (f.beatFired ? 0.5 : 0));
      this._beatDot.style.color = heat > 0.05 ? `rgb(${77 + heat * 178}, ${166 - heat * 90}, ${255 - heat * 200})` : '#666';
      this._beatDot.style.transform = `scale(${1 + heat * 0.6})`;
    }
    if (this._bpmLabel) {
      const bpmTxt = this._bpm > 0 ? Math.round(this._bpm) : '—';
      this._bpmLabel.textContent = `beats ${this._beatCount}  BPM ${bpmTxt}`;
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

function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360 / 360;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
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
