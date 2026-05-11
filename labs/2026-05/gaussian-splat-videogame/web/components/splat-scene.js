// <splat-scene catalog="./catalog.json" [scene-id="..."]>
//
// Custom element that initialises PlayCanvas, loads a Gaussian Splat scene
// from a catalog.json listing, and exposes controls + experiment hooks.
//
// Controls
// ────────
// Orbit mode (default): one-finger drag orbits the target; two-finger
// pinch zooms; two-finger drag (or shift+drag, or right-drag, or middle
// drag, or mouse wheel) pans/zooms; WASD pans the target along the
// horizon. Good for object/scene viewing.
//
// Fly mode (toggle via the hamburger menu): one-finger drag looks around;
// WASD flies through the scene. Good for environment exploration.
//
// Emits: 'ready', 'catalog-loaded', 'scene-loading', 'scene-loaded',
//        'xr-supported', 'xr-state', 'mode-changed'.

import * as pc from 'playcanvas';

const XR_TYPE = pc.XRTYPE_VR;
const XR_SPACE = pc.XRSPACE_LOCALFLOOR;
const D2R = Math.PI / 180;

class SplatScene extends HTMLElement {
  constructor() {
    super();
    this._experiments = new Map();
    this._currentSceneId = null;
    this._splatEntity = null;
    this._catalog = null;
    this._catalogUrl = null;
    this._input = {
      mode: 'orbit',          // 'orbit' | 'fly'
      yaw: 0,                 // degrees
      pitch: -15,             // degrees
      target: new pc.Vec3(),  // orbit target
      distance: 3,            // orbit radius
      flyPos: new pc.Vec3(0, 1.6, 3),
      moveSpeed: 1.5,
      keys: new Set(),
    };
  }

  // ── Public API ────────────────────────────────────────────────────────
  get app() { return this._app; }
  get camera() { return this._cameraEntity; }
  get splatEntity() { return this._splatEntity; }
  get mode() { return this._input.mode; }

  setMode(mode) {
    if (mode !== 'orbit' && mode !== 'fly') return;
    if (this._input.mode === mode) return;
    // Convert state between representations so the camera doesn't jump.
    const cam = this._cameraEntity;
    if (mode === 'fly') {
      this._input.flyPos.copy(cam.getPosition());
      const e = cam.getEulerAngles();
      this._input.yaw = e.y;
      this._input.pitch = e.x;
    } else {
      // pick a target straight ahead, at the current distance
      const fwd = cam.forward;
      const camPos = cam.getPosition();
      const t = this._input.target;
      t.set(
        camPos.x + fwd.x * this._input.distance,
        camPos.y + fwd.y * this._input.distance,
        camPos.z + fwd.z * this._input.distance,
      );
      // yaw/pitch from offset (camera - target)
      const ox = camPos.x - t.x, oy = camPos.y - t.y, oz = camPos.z - t.z;
      const r = Math.hypot(ox, oy, oz) || 1;
      this._input.distance = r;
      this._input.pitch = Math.asin(oy / r) / D2R;
      this._input.yaw = Math.atan2(ox, oz) / D2R;
    }
    this._input.mode = mode;
    this._updateHint();
    this.dispatchEvent(new CustomEvent('mode-changed', { detail: { mode } }));
  }

  registerExperiment(name, instance) {
    this._experiments.set(name, instance);
    this.dispatchEvent(new CustomEvent('experiment-registered', { detail: { name } }));
  }
  experiments() { return Array.from(this._experiments.keys()); }
  isExperimentEnabled(name) {
    const ex = this._experiments.get(name);
    return !!(ex && ex._enabled);
  }
  toggleExperiment(name, enabled) {
    const ex = this._experiments.get(name);
    if (!ex) return;
    const want = !!enabled;
    if (ex._enabled === want) return;
    if (want) ex.enable?.(); else ex.disable?.();
    ex._enabled = want;
    this.dispatchEvent(new CustomEvent('experiment-toggled', { detail: { name, enabled: want } }));
  }

  async loadScene(id) {
    const scene = this._catalog?.scenes.find(s => s.id === id);
    if (!scene) throw new Error(`Unknown scene: ${id}`);
    this._currentSceneId = id;
    this._setStatus(`Loading ${scene.title}…`);
    this.dispatchEvent(new CustomEvent('scene-loading', { detail: { scene } }));

    if (this._splatEntity) {
      this._splatEntity.destroy();
      this._splatEntity = null;
    }

    const url = new URL(scene.url, this._catalogUrl).href;
    const tryLoad = async (u) => {
      console.log(`[splat-scene] tryLoad ${scene.id} ← ${u}`);
      const asset = new pc.Asset(scene.id, 'gsplat', { url: u });
      this._app.assets.add(asset);
      const t0 = performance.now();
      let timer;
      await new Promise((res, rej) => {
        timer = setTimeout(() => {
          rej(new Error(`load timeout after 30s (loaded=${asset.loaded}, loading=${asset.loading})`));
        }, 30000);
        asset.on('error',    (err) => { console.error(`[splat-scene] asset error ${scene.id}:`, err); rej(err); });
        asset.on('progress', (a, b) => console.log(`[splat-scene] asset progress ${scene.id}: ${a}/${b}`));
        asset.on('load',     () => console.log(`[splat-scene] asset 'load' fired ${scene.id} after ${(performance.now()-t0).toFixed(0)}ms`));
        asset.ready(() => {
          clearTimeout(timer);
          console.log(`[splat-scene] asset ready ${scene.id} after ${(performance.now()-t0).toFixed(0)}ms`);
          res(asset);
        });
        this._app.assets.load(asset);
      });
      return asset;
    };

    let asset;
    try {
      asset = await tryLoad(url);
    } catch (e) {
      if (scene.format === 'sog-lod') {
        console.warn(`[splat-scene] LOD root failed, falling back to tile 0_0: ${e.message}`);
        try { asset = await tryLoad(url.replace(/lod-meta\.json$/, '0_0/meta.json')); }
        catch (e2) {
          this._setStatus(`Failed to load ${scene.title}: ${e2.message}`);
          throw e2;
        }
      } else {
        this._setStatus(`Failed to load ${scene.title}: ${e.message}`);
        throw e;
      }
    }

    const entity = new pc.Entity(`splat-${scene.id}`);
    entity.addComponent('gsplat', { asset });
    this._app.root.addChild(entity);
    this._splatEntity = entity;

    const t = scene.transform ?? {};
    const r = t.rotation ?? [0, 0, 0];
    const p = t.position ?? [0, 0, 0];
    const s = t.scale ?? [1, 1, 1];
    entity.setLocalEulerAngles(r[0], r[1], r[2]);
    entity.setLocalPosition(p[0], p[1], p[2]);
    entity.setLocalScale(s[0], s[1], s[2]);

    // Set up controls from catalog hints. Orbit: target = lookAt,
    // distance/yaw/pitch derived from camera offset. Fly: just place.
    const camPos = scene.camera?.position ?? [0, 1.6, 3];
    const look   = scene.camera?.lookAt   ?? [0, 1, 0];
    if (this._input.mode === 'orbit') {
      this._input.target.set(look[0], look[1], look[2]);
      const ox = camPos[0] - look[0];
      const oy = camPos[1] - look[1];
      const oz = camPos[2] - look[2];
      const dist = Math.hypot(ox, oy, oz) || 3;
      this._input.distance = dist;
      this._input.pitch = Math.asin(oy / dist) / D2R;
      this._input.yaw   = Math.atan2(ox, oz) / D2R;
    } else {
      this._input.flyPos.set(camPos[0], camPos[1], camPos[2]);
      const cam = this._cameraEntity;
      cam.setPosition(camPos[0], camPos[1], camPos[2]);
      cam.lookAt(look[0], look[1], look[2]);
      const e = cam.getEulerAngles();
      this._input.yaw = e.y;
      this._input.pitch = e.x;
    }

    this._setStatus(null);
    this.dispatchEvent(new CustomEvent('scene-loaded', { detail: { scene, asset } }));
    return asset;
  }

  async enterXR() {
    if (!this._app?.xr?.supported) return;
    if (this._app.xr.active) return;
    return new Promise((resolve, reject) => {
      this._app.xr.start(this._cameraEntity.camera, XR_TYPE, XR_SPACE, {
        callback: (err) => err ? reject(err) : resolve(),
      });
    });
  }
  exitXR() { if (this._app?.xr?.active) this._app.xr.end(); }

  // ── Lifecycle ─────────────────────────────────────────────────────────
  async connectedCallback() {
    this._buildLayout();
    try {
      await this._initEngine();
      await this._loadCatalog();
      this._renderScenePicker();
      this._wireInput();
      this._wireXR();
      this._updateHint();
      this.dispatchEvent(new CustomEvent('ready'));
      const initial = this.getAttribute('scene-id') || this._catalog.scenes[0]?.id;
      if (initial) await this.loadScene(initial);
    } catch (err) {
      console.error('[splat-scene] init failed', err);
      this._setStatus(`Init failed: ${err.message}`);
    }
  }
  disconnectedCallback() { this._app?.destroy(); }

  // ── Internals ─────────────────────────────────────────────────────────
  _buildLayout() {
    this._canvas = document.createElement('canvas');
    this.appendChild(this._canvas);

    this._statusEl = document.createElement('div');
    this._statusEl.id = 'status';
    this._statusEl.textContent = 'Booting…';
    this.appendChild(this._statusEl);

    this._scenePickerEl = document.createElement('div');
    this._scenePickerEl.className = 'overlay scene-picker';
    this.appendChild(this._scenePickerEl);

    this._xrBtn = document.createElement('button');
    this._xrBtn.className = 'overlay xr-btn';
    this._xrBtn.textContent = 'Checking XR…';
    this._xrBtn.disabled = true;
    this._xrBtn.addEventListener('click', () => {
      if (this._app?.xr?.active) this.exitXR();
      else this.enterXR().catch(e => this._setStatus(`XR failed: ${e.message ?? e}`));
    });
    this.appendChild(this._xrBtn);

    this._hint = document.createElement('div');
    this._hint.className = 'overlay hint';
    this.appendChild(this._hint);
  }

  _updateHint() {
    if (!this._hint) return;
    this._hint.innerHTML = this._input.mode === 'orbit'
      ? 'drag = orbit • pinch = zoom • two-finger drag / shift+drag = pan • WASD pans target'
      : 'WASD / arrows fly • drag = look • shift = fast • wheel / pinch = dolly';
  }

  async _initEngine() {
    const gd = await pc.createGraphicsDevice(this._canvas, {
      deviceTypes: ['webgl2'], antialias: false, alpha: false,
    });
    const opts = new pc.AppOptions();
    opts.graphicsDevice = gd;
    opts.componentSystems = [
      pc.RenderComponentSystem, pc.CameraComponentSystem, pc.LightComponentSystem,
      pc.ScriptComponentSystem, pc.GSplatComponentSystem,
    ];
    opts.resourceHandlers = [
      pc.TextureHandler, pc.ContainerHandler, pc.ScriptHandler, pc.GSplatHandler,
    ];
    opts.mouse = new pc.Mouse(this._canvas);
    opts.touch = new pc.TouchDevice(this._canvas);
    opts.keyboard = new pc.Keyboard(window);
    opts.xr = pc.XrManager;

    const app = new pc.AppBase(this._canvas);
    app.init(opts);
    app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    window.addEventListener('resize', () => app.resizeCanvas());
    app.scene.ambientLight = new pc.Color(0.45, 0.45, 0.5);

    const cam = new pc.Entity('camera');
    cam.addComponent('camera', {
      clearColor: new pc.Color(0.04, 0.04, 0.05),
      farClip: 2000, nearClip: 0.05, fov: 70,
    });
    app.root.addChild(cam);
    this._cameraEntity = cam;

    this._app = app;
    app.start();
    app.on('update', (dt) => this._tick(dt));
  }

  async _loadCatalog() {
    const ref = this.getAttribute('catalog') || './catalog.json';
    this._catalogUrl = new URL(ref, document.baseURI).href;
    const res = await fetch(this._catalogUrl);
    if (!res.ok) throw new Error(`catalog ${ref}: HTTP ${res.status}`);
    this._catalog = await res.json();
    this.dispatchEvent(new CustomEvent('catalog-loaded', { detail: { catalog: this._catalog } }));
  }

  _renderScenePicker() {
    this._scenePickerEl.innerHTML = '';
    for (const scene of this._catalog.scenes) {
      const btn = document.createElement('button');
      btn.className = 'scene-chip';
      btn.textContent = scene.title;
      btn.title = `${scene.author} — ${scene.license}${scene.warning ? ' • ' + scene.warning : ''}`;
      btn.setAttribute('aria-pressed', String(scene.id === this._currentSceneId));
      btn.addEventListener('click', () => {
        this.loadScene(scene.id).then(() => {
          for (const el of this._scenePickerEl.querySelectorAll('.scene-chip'))
            el.setAttribute('aria-pressed', String(el.textContent === scene.title));
        });
      });
      this._scenePickerEl.appendChild(btn);
    }
  }

  _wireInput() {
    const c = this._canvas;
    const pointers = new Map();
    let lastTwo = null;

    const onDown = (e) => {
      c.setPointerCapture?.(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
    };
    const onMove = (e) => {
      const prev = pointers.get(e.pointerId);
      if (!prev) return;
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: prev.button, shift: prev.shift });

      if (pointers.size === 1) {
        // Modifier-pan: shift+drag, right-button drag, or middle-button drag → pan
        const wantsPan = prev.shift || prev.button === 1 || prev.button === 2;
        if (wantsPan && this._input.mode === 'orbit') {
          this._pan(dx, dy);
        } else {
          this._input.yaw   -= dx * 0.3;
          this._input.pitch -= dy * 0.3;
          this._input.pitch = Math.max(-89, Math.min(89, this._input.pitch));
        }
      } else if (pointers.size === 2) {
        const pts = [...pointers.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        const cx = (pts[0].x + pts[1].x) / 2;
        const cy = (pts[0].y + pts[1].y) / 2;
        if (lastTwo) {
          const ddist = dist - lastTwo.dist;
          const dcx = cx - lastTwo.cx;
          const dcy = cy - lastTwo.cy;
          this._zoom(ddist * 0.005);
          this._pan(dcx, dcy);
        }
        lastTwo = { dist, cx, cy };
      }
    };
    const onUp = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) lastTwo = null;
    };
    c.addEventListener('pointerdown', onDown);
    c.addEventListener('pointermove', onMove);
    c.addEventListener('pointerup', onUp);
    c.addEventListener('pointercancel', onUp);
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._zoom(-e.deltaY * 0.0015);
    }, { passive: false });

    window.addEventListener('keydown', (e) => this._input.keys.add(e.code));
    window.addEventListener('keyup',   (e) => this._input.keys.delete(e.code));
  }

  // ── Camera math ──────────────────────────────────────────────────────
  _zoom(amount) {
    // amount > 0 = zoom in (closer)
    if (this._input.mode === 'orbit') {
      this._input.distance = Math.max(0.05, this._input.distance * Math.exp(-amount));
    } else {
      const fwd = this._cameraEntity.forward;
      const k = amount * 1.5;
      this._input.flyPos.add(new pc.Vec3(fwd.x * k, fwd.y * k, fwd.z * k));
    }
  }

  _pan(dxPx, dyPx) {
    const cam = this._cameraEntity;
    const scale = this._input.mode === 'orbit'
      ? this._input.distance * 0.0015
      : 0.002;
    const right = cam.right;
    const up = cam.up;
    const px = -dxPx * scale * right.x +  dyPx * scale * up.x;
    const py = -dxPx * scale * right.y +  dyPx * scale * up.y;
    const pz = -dxPx * scale * right.z +  dyPx * scale * up.z;
    if (this._input.mode === 'orbit') {
      this._input.target.x += px;
      this._input.target.y += py;
      this._input.target.z += pz;
    } else {
      this._input.flyPos.x += px;
      this._input.flyPos.y += py;
      this._input.flyPos.z += pz;
    }
  }

  _tick(dt) {
    if (this._app.xr?.active) return; // XR has its own pose handling
    this._applyKeys(dt);
    if (this._input.mode === 'orbit') this._applyOrbit();
    else this._applyFly();
  }

  _applyOrbit() {
    const { target, yaw, pitch, distance } = this._input;
    const yawR = yaw * D2R, pitchR = pitch * D2R;
    const cp = Math.cos(pitchR), sp = Math.sin(pitchR);
    const sy = Math.sin(yawR),   cy = Math.cos(yawR);
    const x = target.x + distance * cp * sy;
    const y = target.y + distance * sp;
    const z = target.z + distance * cp * cy;
    this._cameraEntity.setPosition(x, y, z);
    this._cameraEntity.lookAt(target.x, target.y, target.z);
  }

  _applyFly() {
    const cam = this._cameraEntity;
    cam.setPosition(this._input.flyPos);
    cam.setEulerAngles(this._input.pitch, this._input.yaw, 0);
  }

  _applyKeys(dt) {
    const { keys, moveSpeed, mode } = this._input;
    if (!keys.size) return;
    const fast = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 4 : 1;
    const cam = this._cameraEntity;
    const fwdRaw = cam.forward;
    const rgtRaw = cam.right;

    if (mode === 'orbit') {
      // Horizon-locked pan along camera-right and camera-forward (XZ-flat).
      const fwd = new pc.Vec3(fwdRaw.x, 0, fwdRaw.z); fwd.normalize();
      const right = new pc.Vec3(rgtRaw.x, 0, rgtRaw.z); right.normalize();
      const speed = moveSpeed * fast * dt * this._input.distance * 0.4;
      const v = new pc.Vec3();
      if (keys.has('KeyW') || keys.has('ArrowUp'))    v.add(fwd);
      if (keys.has('KeyS') || keys.has('ArrowDown'))  v.sub(fwd);
      if (keys.has('KeyA') || keys.has('ArrowLeft'))  v.sub(right);
      if (keys.has('KeyD') || keys.has('ArrowRight')) v.add(right);
      if (keys.has('Space'))       v.y += 1;
      if (keys.has('ControlLeft')) v.y -= 1;
      if (v.lengthSq() > 0) {
        v.normalize().mulScalar(speed);
        this._input.target.x += v.x;
        this._input.target.y += v.y;
        this._input.target.z += v.z;
      }
      // Q/E rotate yaw, R/F adjust pitch — keyboard orbit
      if (keys.has('KeyQ')) this._input.yaw -= 60 * dt;
      if (keys.has('KeyE')) this._input.yaw += 60 * dt;
    } else {
      const speed = moveSpeed * fast * dt;
      const v = new pc.Vec3();
      if (keys.has('KeyW') || keys.has('ArrowUp'))    v.add(fwdRaw);
      if (keys.has('KeyS') || keys.has('ArrowDown'))  v.sub(fwdRaw);
      if (keys.has('KeyA') || keys.has('ArrowLeft'))  v.sub(rgtRaw);
      if (keys.has('KeyD') || keys.has('ArrowRight')) v.add(rgtRaw);
      if (keys.has('Space'))       v.y += 1;
      if (keys.has('ControlLeft')) v.y -= 1;
      if (v.lengthSq() > 0) {
        v.normalize().mulScalar(speed);
        this._input.flyPos.x += v.x;
        this._input.flyPos.y += v.y;
        this._input.flyPos.z += v.z;
      }
    }
  }

  _wireXR() {
    const xr = this._app.xr;
    if (!xr) { this._xrBtn.textContent = 'XR unsupported'; return; }
    const updateBtn = () => {
      if (!xr.supported) {
        this._xrBtn.textContent = 'XR unsupported';
        this._xrBtn.disabled = true;
        return;
      }
      const available = xr.isAvailable(XR_TYPE);
      if (xr.active) {
        this._xrBtn.textContent = 'Exit VR';
        this._xrBtn.disabled = false;
      } else if (available) {
        this._xrBtn.textContent = 'Enter VR';
        this._xrBtn.disabled = false;
      } else {
        this._xrBtn.textContent = 'XR unavailable';
        this._xrBtn.disabled = true;
      }
    };
    xr.on('available', updateBtn);
    xr.on('unavailable', updateBtn);
    xr.on('start', () => {
      updateBtn();
      this.dispatchEvent(new CustomEvent('xr-state', { detail: { active: true } }));
    });
    xr.on('end', () => {
      updateBtn();
      this.dispatchEvent(new CustomEvent('xr-state', { detail: { active: false } }));
    });
    updateBtn();
    if (xr.supported) this.dispatchEvent(new CustomEvent('xr-supported'));
  }

  _setStatus(msg) {
    if (!this._statusEl) return;
    if (msg) {
      this._statusEl.textContent = msg;
      this._statusEl.classList.remove('hidden');
    } else {
      this._statusEl.classList.add('hidden');
    }
  }
}

customElements.define('splat-scene', SplatScene);
