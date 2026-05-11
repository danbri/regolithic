// <splat-scene catalog="./catalog.json" [scene-id="..."]>
//
// Custom element that initialises PlayCanvas, loads a Gaussian Splat scene
// from a catalog.json listing, and exposes controls + experiment hooks
// to its children (e.g. the hamburger menu, blind-cane, wubwub).
//
// Emits: 'ready', 'catalog-loaded', 'scene-loading', 'scene-loaded',
//        'xr-supported', 'xr-state' (detail: { active: bool }).

import * as pc from 'playcanvas';

const XR_TYPE = pc.XRTYPE_VR;
const XR_SPACE = pc.XRSPACE_LOCALFLOOR;

class SplatScene extends HTMLElement {
  constructor() {
    super();
    this._experiments = new Map();
    this._currentSceneId = null;
    this._splatEntity = null;
    this._inputState = { yaw: 0, pitch: 0, keys: new Set(), moveSpeed: 1.5 };
    this._catalog = null;
    this._catalogUrl = null;
  }

  // ── Public API ────────────────────────────────────────────────────────
  get app() { return this._app; }
  get camera() { return this._cameraEntity; }
  get splatEntity() { return this._splatEntity; }

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

    // Remove previous splat entity
    if (this._splatEntity) {
      this._splatEntity.destroy();
      this._splatEntity = null;
    }

    const url = new URL(scene.url, this._catalogUrl).href;
    let assetUrl = url;
    // LOD bundles use lod-meta.json at the root. Engine support varies;
    // if loading the LOD root fails, fall back to the first tile.
    const tryLoad = async (u) => {
      const asset = new pc.Asset(scene.id, 'gsplat', { url: u });
      this._app.assets.add(asset);
      await new Promise((res, rej) => {
        asset.on('error', rej);
        asset.ready(() => res(asset));
        this._app.assets.load(asset);
      });
      return asset;
    };

    let asset;
    try {
      asset = await tryLoad(assetUrl);
    } catch (e) {
      if (scene.format === 'sog-lod') {
        console.warn(`[splat-scene] LOD root failed, falling back to tile 0_0: ${e.message}`);
        assetUrl = url.replace(/lod-meta\.json$/, '0_0/meta.json');
        try {
          asset = await tryLoad(assetUrl);
        } catch (e2) {
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

    // Place the camera per the catalog hint.
    const cam = this._cameraEntity;
    const p = scene.camera?.position ?? [0, 1.6, 3];
    const look = scene.camera?.lookAt ?? [0, 1, 0];
    cam.setPosition(p[0], p[1], p[2]);
    cam.lookAt(look[0], look[1], look[2]);
    // Sync stored Euler angles from the actual transform so drag-look continues smoothly.
    const e = cam.getEulerAngles();
    this._inputState.yaw = e.y;
    this._inputState.pitch = e.x;

    this._setStatus(null);
    this.dispatchEvent(new CustomEvent('scene-loaded', { detail: { scene, asset } }));
    return asset;
  }

  async enterXR() {
    if (!this._app?.xr?.supported) return;
    if (this._app.xr.active) return;
    return new Promise((resolve, reject) => {
      this._app.xr.start(this._cameraEntity.camera, XR_TYPE, XR_SPACE, {
        callback: (err) => {
          if (err) reject(err); else resolve();
        }
      });
    });
  }

  exitXR() {
    if (this._app?.xr?.active) this._app.xr.end();
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────
  async connectedCallback() {
    this._buildLayout();
    try {
      await this._initEngine();
      await this._loadCatalog();
      this._renderScenePicker();
      this._wireInput();
      this._wireXR();
      this.dispatchEvent(new CustomEvent('ready'));
      const initial = this.getAttribute('scene-id') || this._catalog.scenes[0]?.id;
      if (initial) await this.loadScene(initial);
    } catch (err) {
      console.error('[splat-scene] init failed', err);
      this._setStatus(`Init failed: ${err.message}`);
    }
  }

  disconnectedCallback() {
    this._app?.destroy();
  }

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
    this._hint.innerHTML = 'WASD / arrows to move • drag to look • pinch to zoom';
    this.appendChild(this._hint);
  }

  async _initEngine() {
    const gd = await pc.createGraphicsDevice(this._canvas, {
      deviceTypes: ['webgl2'],
      antialias: false,
      alpha: false,
    });

    const opts = new pc.AppOptions();
    opts.graphicsDevice = gd;
    opts.componentSystems = [
      pc.RenderComponentSystem,
      pc.CameraComponentSystem,
      pc.LightComponentSystem,
      pc.ScriptComponentSystem,
      pc.GSplatComponentSystem,
    ];
    opts.resourceHandlers = [
      pc.TextureHandler,
      pc.ContainerHandler,
      pc.ScriptHandler,
      pc.GSplatHandler,
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

    // Subtle ambient so non-emissive helper meshes (cane, debug aabb) read.
    app.scene.ambientLight = new pc.Color(0.45, 0.45, 0.5);

    // Camera
    const cam = new pc.Entity('camera');
    cam.addComponent('camera', {
      clearColor: new pc.Color(0.04, 0.04, 0.05),
      farClip: 2000,
      nearClip: 0.05,
      fov: 70,
    });
    cam.setPosition(0, 1.6, 3);
    app.root.addChild(cam);
    this._cameraEntity = cam;

    this._app = app;
    app.start();
    this._tickHandle = (dt) => this._tick(dt);
    app.on('update', this._tickHandle);
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
      btn.title = `${scene.author} — ${scene.license}`;
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
    let pointers = new Map();
    let lastTouchDist = null;

    const onDown = (e) => {
      c.setPointerCapture?.(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    };
    const onMove = (e) => {
      if (!pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId);
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.size === 1) {
        this._inputState.yaw   -= dx * 0.2;
        this._inputState.pitch -= dy * 0.2;
        this._inputState.pitch = Math.max(-89, Math.min(89, this._inputState.pitch));
      } else if (pointers.size === 2) {
        const pts = [...pointers.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (lastTouchDist != null) {
          const ddist = dist - lastTouchDist;
          this._moveAlongView(ddist * 0.005);
        }
        lastTouchDist = dist;
      }
    };
    const onUp = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) lastTouchDist = null;
    };
    c.addEventListener('pointerdown', onDown);
    c.addEventListener('pointermove', onMove);
    c.addEventListener('pointerup', onUp);
    c.addEventListener('pointercancel', onUp);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._moveAlongView(-e.deltaY * 0.002);
    }, { passive: false });

    window.addEventListener('keydown', (e) => this._inputState.keys.add(e.code));
    window.addEventListener('keyup',   (e) => this._inputState.keys.delete(e.code));
  }

  _moveAlongView(amount) {
    const cam = this._cameraEntity;
    const fwd = cam.forward;
    cam.translate(fwd.x * amount, fwd.y * amount, fwd.z * amount);
  }

  _tick(dt) {
    if (this._app.xr?.active) return; // XR has its own pose handling
    // Apply yaw/pitch from drag to camera rotation
    const { yaw, pitch, keys, moveSpeed } = this._inputState;
    this._cameraEntity.setEulerAngles(pitch, yaw, 0);

    // WASD / arrows fly
    const cam = this._cameraEntity;
    const speed = (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 4 : 1) * moveSpeed * dt;
    const fwd = cam.forward, right = cam.right, up = pc.Vec3.UP;
    const move = new pc.Vec3();
    if (keys.has('KeyW') || keys.has('ArrowUp'))    move.add(fwd);
    if (keys.has('KeyS') || keys.has('ArrowDown'))  move.sub(fwd);
    if (keys.has('KeyA') || keys.has('ArrowLeft'))  move.sub(right);
    if (keys.has('KeyD') || keys.has('ArrowRight')) move.add(right);
    if (keys.has('Space'))      move.add(up);
    if (keys.has('ControlLeft')) move.sub(up);
    if (move.lengthSq() > 0) {
      move.normalize().mulScalar(speed);
      cam.translate(move.x, move.y, move.z);
    }
  }

  _wireXR() {
    const xr = this._app.xr;
    if (!xr) {
      this._xrBtn.textContent = 'XR unsupported';
      return;
    }
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
