// Hamburger menu with collapsible sections (using native <details>).
//
// Sections (top-level, expandable):
//   • Catalog     — scenes grouped Category → Subcategory → Scene.
//   • Scene       — current scene credit (author, license, link).
//   • Render      — move speed, FOV, orbit/fly mode.
//   • AI          — Chrome Prompt API (Nano) + WebLLM stubs; one-shot
//                    scene analysis that snapshots the canvas and shows
//                    the response in a bottom banner.
//   • Experiments — toggle each registered experiment; per-experiment
//                    settings are nested <details> closed by default.
//
// Side-effect module: auto-attaches to the first <splat-scene> on the
// page once it emits 'ready'.

import { buildRegistry, pickDefaultModelId, isIOS } from '../ai/models.js';
import { SplatWorld } from '../splatworld/world.js';
import { PhoneCane } from '../splatworld/phone-cane.js';
import { Tour } from '../splatworld/tour.js';
import { Drone } from '../splatworld/drone.js';
import { Sonar } from '../splatworld/sonar.js';

function attach(scene) {
  const root = scene;

  // Hamburger button
  const btn = document.createElement('button');
  btn.className = 'overlay menu-btn';
  btn.setAttribute('aria-label', 'Open settings');
  btn.textContent = '☰';
  root.appendChild(btn);

  // Panel
  const panel = document.createElement('div');
  panel.className = 'overlay menu-panel hidden';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Settings');
  root.appendChild(panel);

  btn.addEventListener('click', () => {
    panel.classList.toggle('hidden');
    btn.setAttribute('aria-expanded', String(!panel.classList.contains('hidden')));
  });

  // ── Scenes (catalog + active scene credit at top) ─────────────────────
  const scenesSec = makeSection('Scenes', false, 'scenes');
  panel.appendChild(scenesSec.el);
  const credit = document.createElement('div');
  credit.className = 'credit';
  credit.innerHTML = '<span class="meta">Loading…</span>';
  scenesSec.body.appendChild(credit);
  const updateCredit = (s) => {
    credit.innerHTML = s
      ? `<strong>Now showing:</strong> ${esc(s.title)} <span class="meta">·
         by ${esc(s.author)} ·
         <a href="${escAttr(s.license_url)}" target="_blank" rel="noopener">${esc(s.license)}</a>
         ${s.size_mb ? ` · ${s.size_mb} MB` : ''}</span>`
      : '<span class="meta">No scene loaded.</span>';
  };
  scene.addEventListener('scene-loading', e => updateCredit(e.detail.scene));
  scene.addEventListener('scene-loaded',  e => updateCredit(e.detail.scene));
  buildCatalog(scenesSec.body, scene);

  // ── View (camera + render) ────────────────────────────────────────────
  const viewSec = makeSection('View', true, 'view');
  panel.appendChild(viewSec.el);
  viewSec.body.appendChild(slider('Move speed', 0.2, 8, 0.1, scene._input.moveSpeed, v => scene._input.moveSpeed = v));
  viewSec.body.appendChild(slider('FOV', 40, 110, 1, scene.camera?.camera?.fov ?? 70, v => {
    if (scene.camera?.camera) scene.camera.camera.fov = v;
  }));
  const modeRow = row();
  modeRow.appendChild(span('Mode'));
  const modeSel = document.createElement('select');
  for (const m of ['orbit', 'fly']) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m === 'orbit' ? 'Orbit (around target)' : 'Fly (free camera)';
    modeSel.appendChild(opt);
  }
  modeSel.value = scene.mode;
  modeSel.addEventListener('change', () => scene.setMode(modeSel.value));
  scene.addEventListener('mode-changed', () => { modeSel.value = scene.mode; });
  modeRow.appendChild(modeSel);
  viewSec.body.appendChild(modeRow);

  // ── World (mesh detection + sonar + cane) ─────────────────────────────
  const worldSec = makeSection('World', false, 'world');
  panel.appendChild(worldSec.el);
  buildSplatWorldSection(worldSec.body, scene);

  // ── Experiments (umbrella for everything else) ────────────────────────
  // Holds Explore (Tour + Drone), AI, and each registered experiment
  // (Cane, WubWub) — each as its own collapsible sub-folder.
  const xpSec = makeSection('Experiments', true, 'experiments');
  panel.appendChild(xpSec.el);

  // Explore sub-folder
  const exploreDet = document.createElement('details');
  exploreDet.className = 'tree-group tree-sub';
  const exploreSum = document.createElement('summary');
  exploreSum.textContent = 'Explore — Tour & Drone';
  exploreDet.appendChild(exploreSum);
  const exploreBody = document.createElement('div');
  exploreBody.className = 'menu-body';
  exploreDet.appendChild(exploreBody);
  xpSec.body.appendChild(exploreDet);

  const tourSub = document.createElement('details');
  tourSub.className = 'tree-group tree-sub';
  const tourSum = document.createElement('summary');
  tourSum.textContent = 'Tour — spiral fly-through';
  tourSub.appendChild(tourSum);
  const tourBody = document.createElement('div');
  tourBody.className = 'menu-body';
  tourSub.appendChild(tourBody);
  exploreBody.appendChild(tourSub);
  buildTourSection(tourBody, scene);

  const droneSub = document.createElement('details');
  droneSub.className = 'tree-group tree-sub';
  const droneSum = document.createElement('summary');
  droneSum.textContent = 'Drone — autonomous survey';
  droneSub.appendChild(droneSum);
  const droneBody = document.createElement('div');
  droneBody.className = 'menu-body';
  droneSub.appendChild(droneBody);
  exploreBody.appendChild(droneSub);
  buildDroneSection(droneBody, scene);

  // AI sub-folder
  const aiDet = document.createElement('details');
  aiDet.className = 'tree-group tree-sub';
  const aiSum = document.createElement('summary');
  aiSum.textContent = 'AI — vision models & speech';
  aiDet.appendChild(aiSum);
  const aiBody = document.createElement('div');
  aiBody.className = 'menu-body';
  aiDet.appendChild(aiBody);
  xpSec.body.appendChild(aiDet);
  buildAISection(aiBody, scene);

  // Each registered experiment (Cane, WubWub, …) gets its own sub-folder
  const expsContainer = document.createElement('div');
  expsContainer.className = 'experiments-container';
  xpSec.body.appendChild(expsContainer);

  const rebuildExperiments = () => {
    expsContainer.innerHTML = '';
    const names = scene.experiments();
    if (names.length === 0) {
      expsContainer.appendChild(emptyHint('(no other experiments registered)'));
      return;
    }
    for (const name of names) {
      const det = document.createElement('details');
      det.className = 'tree-group tree-sub';
      const sum = document.createElement('summary');
      sum.textContent = humanise(name);
      det.appendChild(sum);
      const body = document.createElement('div');
      body.className = 'menu-body';
      det.appendChild(body);

      // Toggle row inside the sub-folder
      const toggleRow = document.createElement('label');
      toggleRow.className = 'experiment-toggle';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = scene.isExperimentEnabled(name);
      cb.addEventListener('change', () => scene.toggleExperiment(name, cb.checked));
      toggleRow.appendChild(cb);
      const lbl = document.createElement('strong');
      lbl.textContent = 'Enabled';
      toggleRow.appendChild(lbl);
      body.appendChild(toggleRow);

      // Settings inline (no extra "Settings" sub-details)
      const ex = scene._experiments.get(name);
      if (typeof ex?.renderSettings === 'function') {
        ex.renderSettings(body);
      }
      expsContainer.appendChild(det);
    }
  };
  scene.addEventListener('experiment-registered', rebuildExperiments);
  rebuildExperiments();

  // Trigger initial credit render if a scene is already loaded
  if (scene._currentSceneId) {
    const s = scene._catalog?.scenes.find(x => x.id === scene._currentSceneId);
    if (s) updateCredit(s);
  }

  // Slide-open/close animation for every <details> in the panel.
  // We wrap each details' non-summary children in a `.collapse` div
  // (styled overflow:hidden + height transition) and on each toggle
  // animate height between 0 and the measured scrollHeight.
  // Works in every browser with CSS transitions — no reliance on
  // interpolate-size / ::details-content which iOS Safari hasn't
  // landed reliably yet.
  setupSlideAnimations(panel);
}

function setupSlideAnimations(panel) {
  for (const d of panel.querySelectorAll('details')) hookSlide(d);
  // Re-hook anything added later (rebuildExperiments etc).
  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        if (n.tagName === 'DETAILS') hookSlide(n);
        const inner = n.querySelectorAll?.('details');
        if (inner) for (const d of inner) hookSlide(d);
      }
    }
  });
  obs.observe(panel, { childList: true, subtree: true });
}

function hookSlide(det) {
  if (det._slideHooked) return;
  det._slideHooked = true;
  // Wrap non-summary children in .collapse (skip if already wrapped).
  // That's the only thing the JS does — CSS owns the transition via
  // grid-template-rows: 0fr ↔ 1fr on `[open]`.
  if (det.querySelector(':scope > .collapse')) return;
  const children = Array.from(det.children).filter(c => c.tagName !== 'SUMMARY');
  if (children.length === 0) return;
  const collapse = document.createElement('div');
  collapse.className = 'collapse';
  for (const c of children) collapse.appendChild(c);
  det.appendChild(collapse);
}

// ── Catalog tree (Category → Subcategory → Scene) ───────────────────────
function buildCatalog(host, scene) {
  const catalog = scene._catalog;
  if (!catalog) {
    host.appendChild(emptyHint('Catalog not loaded.'));
    return;
  }
  const cats = catalog.categories || {};

  // group: cat -> sub -> [scenes]
  const byCat = new Map();
  for (const s of catalog.scenes) {
    const c = s.category || 'other';
    const sub = s.subcategory || '(misc)';
    if (!byCat.has(c)) byCat.set(c, new Map());
    const m = byCat.get(c);
    if (!m.has(sub)) m.set(sub, []);
    m.get(sub).push(s);
  }
  const catOrder = Array.from(byCat.keys()).sort((a, b) => {
    return (cats[a]?.order ?? 999) - (cats[b]?.order ?? 999);
  });

  for (const cat of catOrder) {
    const subs = byCat.get(cat);
    const total = Array.from(subs.values()).reduce((n, arr) => n + arr.length, 0);
    const det = document.createElement('details');
    det.open = true;
    det.className = 'tree-group tree-cat';
    const sum = document.createElement('summary');
    sum.textContent = `${cats[cat]?.title ?? cap(cat)} (${total})`;
    det.appendChild(sum);

    const subOrder = Array.from(subs.keys()).sort();
    for (const sub of subOrder) {
      const subItems = subs.get(sub).slice().sort((a, b) => (a.size_mb ?? 0) - (b.size_mb ?? 0));
      const subDet = document.createElement('details');
      subDet.open = true;
      subDet.className = 'tree-group tree-sub';
      const subSum = document.createElement('summary');
      subSum.textContent = `${sub} (${subItems.length})`;
      subDet.appendChild(subSum);
      for (const s of subItems) {
        const leaf = document.createElement('button');
        leaf.className = 'tree-leaf';
        leaf.dataset.sceneId = s.id;
        leaf.setAttribute('aria-pressed', String(s.id === scene._currentSceneId));
        leaf.innerHTML = `
          <strong>${esc(s.title)}</strong>
          <span class="meta">${esc(s.author)} · ${s.size_mb ?? '?'} MB</span>
          ${s.warning ? `<span class="warn">⚠ ${esc(s.warning)}</span>` : ''}
        `;
        leaf.addEventListener('click', () => scene.loadScene(s.id));
        subDet.appendChild(leaf);
      }
      det.appendChild(subDet);
    }
    host.appendChild(det);
  }

  scene.addEventListener('scene-loaded', (e) => {
    const id = e.detail.scene.id;
    for (const el of host.querySelectorAll('.tree-leaf')) {
      el.setAttribute('aria-pressed', String(el.dataset.sceneId === id));
    }
  });
}

// ── SplatWorld section ──────────────────────────────────────────────────
function buildSplatWorldSection(host, scene) {
  // World is a singleton attached to the scene so other modules can use it.
  const world = scene._splatworld ?? (scene._splatworld = new SplatWorld(scene));
  let phoneCane = scene._phoneCane ?? null;

  const intro = document.createElement('div');
  intro.className = 'credit';
  intro.innerHTML = `Approximate world model derived automatically
    every time you load a scene, then cached per scene so reloads
    are instant (<code>localStorage</code>). A handful of typed
    primitives (ground / wall / obstacle / pip) are synthesised from
    the splat's AABB. Real mesh extraction
    (<code>splat-transform&nbsp;-K</code>) is the upgrade path.`;
  host.appendChild(intro);

  const status = document.createElement('div');
  status.className = 'credit';
  host.appendChild(status);

  const detectBtn = document.createElement('button');
  detectBtn.style.width = '100%';
  detectBtn.style.padding = '8px';
  detectBtn.addEventListener('click', () => {
    try { world.detect(); }
    catch (e) { status.innerHTML = `<span class="warn">${esc(e.message)}</span>`; }
  });
  host.appendChild(detectBtn);

  const helpersRow = row();
  const helpersCb = document.createElement('input');
  helpersCb.type = 'checkbox';
  const helpersLbl = document.createElement('label');
  helpersLbl.className = 'experiment-toggle';
  helpersLbl.appendChild(helpersCb);
  const helpersSpan = document.createElement('span');
  helpersSpan.textContent = 'Show helper boxes (debug)';
  helpersLbl.appendChild(helpersSpan);
  helpersCb.addEventListener('change', () => world.setHelpersVisible(helpersCb.checked));
  host.appendChild(helpersLbl);

  // Sonar toggle — proximity ping audio, no UI beyond the toggle.
  const sonar = scene._sonar ?? (scene._sonar = new Sonar(scene, world));
  const sonarRow = document.createElement('label');
  sonarRow.className = 'experiment-toggle';
  const sonarCb = document.createElement('input');
  sonarCb.type = 'checkbox';
  sonarCb.addEventListener('change', async () => {
    try {
      if (sonarCb.checked) await sonar.start();
      else sonar.stop();
    } catch (e) {
      sonarCb.checked = false;
      status.innerHTML = `<span class="warn">${esc(e.message)}</span>`;
    }
  });
  sonarRow.appendChild(sonarCb);
  const sonarSpan = document.createElement('span');
  sonarSpan.innerHTML = 'Sonar pings <span class="meta">— forward-raycast → ping frequency + rate scale with proximity</span>';
  sonarRow.appendChild(sonarSpan);
  host.appendChild(sonarRow);
  host.appendChild(slider('Sonar range (m)',   1,  20, 0.5, sonar.maxRange, v => sonar.maxRange = v));
  host.appendChild(slider('Sonar volume',      0,  1,  0.05, sonar.volume,  v => sonar.volume = v));

  // Phone-sensor cane sub-details
  const caneDet = document.createElement('details');
  caneDet.className = 'tree-group tree-sub';
  const caneSum = document.createElement('summary');
  caneSum.textContent = 'Phone-sensor cane';
  caneDet.appendChild(caneSum);
  const caneBody = document.createElement('div');
  caneBody.className = 'menu-body';
  caneBody.innerHTML = `
    <div class="credit">
      <span class="meta">Tilt your phone to swing a virtual cane through the scene.
      Detected mesh primitives drive the audio palette
      (pip / thunk / screech) and a continuous low rumble swells as the
      tip approaches any surface.</span>
    </div>
    <div class="ai-model-status credit">cane: off</div>
    <div class="ai-model-action"></div>
  `;
  caneDet.appendChild(caneBody);
  host.appendChild(caneDet);

  const caneStatus = caneBody.querySelector('.ai-model-status');
  const caneAction = caneBody.querySelector('.ai-model-action');

  function paintCaneButton() {
    const running = phoneCane?.running;
    caneStatus.textContent = `cane: ${running ? 'on' : 'off'}`;
    caneAction.innerHTML = '';
    if (!world.enabled) {
      caneAction.innerHTML = `<span class="credit"><span class="meta">Enable mesh detection above first.</span></span>`;
      return;
    }
    const btn = document.createElement('button');
    btn.textContent = running ? 'Stop phone cane' : 'Start phone cane';
    btn.addEventListener('click', async () => {
      try {
        if (running) {
          phoneCane.stop();
        } else {
          if (!phoneCane) phoneCane = scene._phoneCane = new PhoneCane(scene, world);
          await phoneCane.start();
        }
      } catch (e) {
        caneStatus.innerHTML = `<span class="warn">${esc(e.message)}</span>`;
      } finally {
        paintCaneButton();
      }
    });
    caneAction.appendChild(btn);
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') {
      const note = document.createElement('div');
      note.className = 'credit';
      note.innerHTML = `<span class="meta">iOS: tapping Start prompts for motion permission.</span>`;
      caneAction.appendChild(note);
    }
  }

  function paint() {
    if (!world.enabled) {
      status.innerHTML = `<span class="meta">Mesh detection: <strong>off</strong></span>`;
      detectBtn.textContent = '✦ Enable mesh detection';
    } else {
      const counts = world.primitives.reduce((acc, p) => {
        acc[p.type] = (acc[p.type] || 0) + 1; return acc;
      }, {});
      const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(' · ');
      status.innerHTML = `<span class="meta">Mesh detection: <strong>on</strong> — ${summary || 'no primitives'}</span>`;
      detectBtn.textContent = '↻ Re-detect for current scene';
    }
    paintCaneButton();
  }

  world.addEventListener('changed', paint);
  paint();
}

// ── Tour section ────────────────────────────────────────────────────────
function buildTourSection(host, scene) {
  // Singleton attached to scene so the world & tour share state.
  const world = scene._splatworld ?? (scene._splatworld = new SplatWorld(scene));
  const tour = scene._tour ?? (scene._tour = new Tour(scene, world));

  const intro = document.createElement('div');
  intro.className = 'credit';
  intro.innerHTML = `Spiral fly-through of the current splat, looking at
    each major mesh primitive in turn. If SplatWorld mesh detection is
    on, the cane primitives become the look-at targets; otherwise the
    spiral just orbits the splat centre.
    <span class="meta">No image capture or AI hook yet — this is the
    camera-flight mock-up.</span>`;
  host.appendChild(intro);

  const status = document.createElement('div');
  status.className = 'credit';
  host.appendChild(status);

  const ctrl = document.createElement('div');
  ctrl.className = 'row';
  const startBtn = document.createElement('button');
  startBtn.style.flex = '2';
  const prevBtn = document.createElement('button');
  prevBtn.textContent = '◀ Prev';
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next ▶';
  ctrl.appendChild(startBtn);
  ctrl.appendChild(prevBtn);
  ctrl.appendChild(nextBtn);
  host.appendChild(ctrl);

  host.appendChild(slider('Step duration (s)', 0.5, 8, 0.1, tour.stepDuration, v => tour.stepDuration = v));
  host.appendChild(slider('Dwell (s)',        0,   3, 0.1, tour.dwellDuration, v => tour.dwellDuration = v));

  const paint = () => {
    startBtn.textContent = tour.running ? '■ Stop tour' : '▶ Start tour';
    const total = tour.waypoints.length;
    const cur = tour.idx + 1;
    if (tour.running) {
      const wp = tour.currentWaypoint;
      status.innerHTML = `<span class="meta">Waypoint <strong>${cur}</strong> / ${total}${wp ? ' — ' + esc(wp.label) : ''}</span>`;
      prevBtn.disabled = false;
      nextBtn.disabled = false;
    } else {
      status.innerHTML = `<span class="meta">Stopped. ${total} waypoints last computed.</span>`;
      prevBtn.disabled = true;
      nextBtn.disabled = true;
    }
  };

  startBtn.addEventListener('click', () => {
    try {
      if (tour.running) tour.stop();
      else tour.start();
    } catch (e) {
      status.innerHTML = `<span class="warn">${esc(e.message)}</span>`;
    }
  });
  prevBtn.addEventListener('click', () => tour.prev());
  nextBtn.addEventListener('click', () => tour.next());

  tour.addEventListener('changed', paint);
  tour.addEventListener('waypoint-reached', paint);
  paint();
}

// ── Drone section ──────────────────────────────────────────────────────
function buildDroneSection(host, scene) {
  // Reuse the shared SplatWorld instance so the drone can read mesh
  // primitives (if mesh detection is on, they become collision tests).
  const world = scene._splatworld ?? (scene._splatworld = new SplatWorld(scene));
  const drone = scene._drone ?? (scene._drone = new Drone(scene, world));

  const intro = document.createElement('div');
  intro.className = 'credit';
  intro.innerHTML = `Autonomous hoverdrone: continuous wander around the
    splat, with short-range raycast avoidance against SplatWorld
    primitives (turn on mesh detection above for better avoidance).
    Every few seconds it grabs a frame and runs the active
    object-detection model — discovered labels accumulate below.`;
  host.appendChild(intro);

  const status = document.createElement('div');
  status.className = 'credit';
  host.appendChild(status);

  const summary = document.createElement('div');
  summary.className = 'credit';
  host.appendChild(summary);

  const ctrlRow = document.createElement('div');
  ctrlRow.className = 'row';
  const startBtn = document.createElement('button');
  startBtn.style.flex = '2';
  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset survey';
  ctrlRow.appendChild(startBtn);
  ctrlRow.appendChild(resetBtn);
  host.appendChild(ctrlRow);

  host.appendChild(slider('Speed (m/s)',     0.1, 3,    0.1, drone.flightSpeed,    v => drone.flightSpeed = v));
  host.appendChild(slider('Turn rate (°/s)', 10,  180,  5,   drone.turnRate,       v => drone.turnRate = v));
  host.appendChild(slider('Look-ahead (m)',  0.2, 3,    0.1, drone.lookAhead,      v => drone.lookAhead = v));
  host.appendChild(slider('Survey every (s)',1,   10,   0.5, drone.surveyInterval, v => drone.surveyInterval = v));
  host.appendChild(slider('Wander curiosity',0,   2,    0.05,drone.wanderJitter,   v => drone.wanderJitter = v));

  function paint() {
    startBtn.textContent = drone.running ? '■ Stop drone' : '▶ Launch drone';
    resetBtn.disabled = drone.discovered.size === 0;
    status.innerHTML = drone.running
      ? `<span class="meta">Flying. Active model: ${esc(activeDetectorLabel(scene))}.</span>`
      : `<span class="meta">Idle.</span>`;
    const total = drone.discovered.size;
    summary.innerHTML = total === 0
      ? `<span class="meta">No discoveries yet.</span>`
      : `<strong>${total}</strong> objects seen — <span class="meta">${esc(drone.surveySummary())}</span>`;
  }

  startBtn.addEventListener('click', async () => {
    try {
      if (drone.running) {
        drone.stop();
      } else {
        // Pick the active model from the AI registry — must be an
        // object-detection model. If the current active isn't, fall
        // back to YOLOS Tiny.
        const ai = scene._aiState;
        let m = ai?.activeModel;
        if (!m || m.mode !== 'object-detection') {
          m = ai?.models?.find(x => x.id === 'yolos-tiny') ?? ai?.models?.find(x => x.mode === 'object-detection');
        }
        if (!m) throw new Error('No object-detection model available. Open the AI section.');
        scene.showResponse('Drone arming…', { badge: 'Drone', ttlMs: 4000 });
        await drone.start(m);
      }
    } catch (e) {
      status.innerHTML = `<span class="warn">${esc(e.message)}</span>`;
    } finally {
      paint();
    }
  });
  resetBtn.addEventListener('click', () => drone.resetSurvey());

  drone.addEventListener('changed', paint);
  drone.addEventListener('status', (e) => {
    status.innerHTML = `<span class="meta">${esc(e.detail.text)}</span>`;
  });
  drone.addEventListener('survey', (e) => {
    const { newLabels, total } = e.detail;
    if (newLabels.length) {
      scene.showResponse(
        `New: ${newLabels.join(', ')}  •  ${total} total`,
        { badge: 'Drone', ttlMs: 6000 },
      );
    }
    paint();
  });
  paint();
}

function activeDetectorLabel(scene) {
  const ai = scene._aiState;
  const m = ai?.activeModel;
  if (m?.mode === 'object-detection') return m.label;
  return 'YOLOS Tiny (fallback)';
}

// ── AI section ──────────────────────────────────────────────────────────
function buildAISection(host, scene) {
  const models = buildRegistry();
  const kind = browserKind();
  const webGPU = (typeof navigator !== 'undefined') && ('gpu' in navigator);
  // Platform-aware default: prefer Nano on Chrome (no download); on iOS
  // pick the smallest Gemma (Gemma 1 2b q4f16) to dodge the 1.5 GB
  // per-tab cap that crashes Gemma 2 2B mid-load.
  let activeId = (kind === 'chrome' || kind === 'edge') ? 'chrome-builtin' : pickDefaultModelId(models);
  // Expose so the Drone (and others) can find the active model.
  scene._aiState = { models, get activeModel() { return models.find(m => m.id === activeId); } };

  // ── Browser status banner ─────────────────────────────────────────────
  const banner = document.createElement('div');
  banner.className = 'credit ai-banner';
  host.appendChild(banner);
  renderBanner();

  function renderBanner() {
    const isWebKit = (kind === 'safari' || kind === 'safari-ios' || kind === 'chrome-ios' || kind === 'firefox-ios');
    if (isWebKit) {
      // iOS / desktop Safari: Prompt API not available. WebLLM is the path.
      banner.innerHTML = `
        <strong>Browser uses WebKit${kind.endsWith('ios') ? ' (iOS)' : ''}.</strong><br>
        Chrome's built-in Gemini Nano isn't available — Apple's WebKit
        doesn't ship the Prompt API, and on iOS every browser
        (including Chrome and Firefox) runs on WebKit by App Store rule.
        ${webGPU
          ? '<br>WebGPU is present, so the WebLLM models below should work after a one-time download.'
          : '<br><span class="warn">WebGPU is not available; WebLLM models will report unavailable.</span>'}
      `;
    } else if (kind === 'chrome' || kind === 'edge') {
      banner.innerHTML = `Chrome/Chromium detected. Built-in Nano needs the
        Prompt API flag enabled:
        <div class="ai-flag-row">
          <code>chrome://flags/#prompt-api-for-gemini-nano</code>
          <button class="ai-copy" aria-label="Copy flag URL">Copy</button>
        </div>
        <span class="meta">chrome:// URLs can't be opened from a web page —
        paste the URL in the address bar.</span>`;
      const copy = banner.querySelector('.ai-copy');
      copy.addEventListener('click', async () => {
        const txt = 'chrome://flags/#prompt-api-for-gemini-nano';
        try {
          await navigator.clipboard.writeText(txt);
          copy.textContent = 'Copied ✓';
          setTimeout(() => copy.textContent = 'Copy', 1500);
        } catch {
          copy.textContent = 'Press long → Copy';
        }
      });
    } else {
      banner.innerHTML = `Browser: <strong>${esc(kind)}</strong>. Built-in
        AI status depends on the browser; WebLLM is available via WebGPU
        ${webGPU ? '(detected)' : '(<span class="warn">not detected</span>)'}.`;
    }
  }

  // ── Active-model select + analyse button ──────────────────────────────
  const actions = document.createElement('div');
  actions.className = 'menu-body';
  host.appendChild(actions);

  const status = document.createElement('div');
  status.className = 'credit';
  status.textContent = 'Checking…';
  actions.appendChild(status);

  // Safe-mode toggle (force WASM, skip WebGPU entirely). iOS defaults
  // ON because WebGPU has been crashing tabs there. User can flip OFF
  // for speed on desktops.
  const safeRow = row();
  const safeLbl = document.createElement('label');
  safeLbl.className = 'experiment-toggle';
  const safeCb = document.createElement('input');
  safeCb.type = 'checkbox';
  const lsKey = 'aiSafeMode';
  let safeOn;
  try {
    const stored = localStorage.getItem(lsKey);
    safeOn = stored == null ? isIOS() : (stored === 'true');
    // Make sure localStorage reflects the effective state so
    // transformers-js.js reads it correctly.
    localStorage.setItem(lsKey, safeOn ? 'true' : 'false');
  } catch { safeOn = isIOS(); }
  safeCb.checked = safeOn;
  safeCb.addEventListener('change', () => {
    safeOn = safeCb.checked;
    try { localStorage.setItem(lsKey, safeOn ? 'true' : 'false'); } catch {}
    status.innerHTML = `<span class="meta">Safe mode ${safeOn ? 'ON (CPU/WASM)' : 'OFF (WebGPU)'}. Reload the page to apply.</span>`;
  });
  safeLbl.appendChild(safeCb);
  const safeSpan = document.createElement('span');
  safeSpan.innerHTML = 'Safe mode (CPU / WASM only) <span class="meta">— slower but skips WebGPU crashes</span>';
  safeLbl.appendChild(safeSpan);
  safeRow.appendChild(safeLbl);
  actions.appendChild(safeRow);

  const actRow = row();
  actRow.appendChild(span('Active'));
  const sel = document.createElement('select');
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    sel.appendChild(opt);
  }
  sel.value = activeId;
  sel.addEventListener('change', () => { activeId = sel.value; refresh(); });
  actRow.appendChild(sel);
  actions.appendChild(actRow);

  const analyse = document.createElement('button');
  analyse.textContent = '✦ Analyse current scene';
  analyse.style.width = '100%';
  analyse.style.padding = '10px';
  analyse.style.marginTop = '6px';
  analyse.addEventListener('click', () => runAnalysis({ persistent: false }).catch(err => {
    console.error('[ai] analysis failed', err);
    scene.showResponse(String(err.message || err), { badge: 'Error', ttlMs: 8000 });
  }));
  actions.appendChild(analyse);

  // ── Auto-relabel: re-run analysis every N seconds ─────────────────────
  // Default cadence: 3 s for object-detection (fast, useful as the camera
  // moves), 6 s for everything else (captioners are slower). The slider
  // lets users override.
  let autoTimer = null;
  let autoBusy = false;
  let autoIntervalSec = 3;
  function defaultIntervalForModel(m) {
    if (!m) return 3;
    return m.mode === 'object-detection' ? 3 : 6;
  }
  const autoRow = row();
  const autoLbl = document.createElement('label');
  autoLbl.className = 'experiment-toggle';
  const autoCb = document.createElement('input');
  autoCb.type = 'checkbox';
  autoLbl.appendChild(autoCb);
  const autoSpan = document.createElement('span');
  autoSpan.innerHTML = '↻ Auto-relabel <span class="meta">— rerun while toggle is on</span>';
  autoLbl.appendChild(autoSpan);
  autoRow.appendChild(autoLbl);
  actions.appendChild(autoRow);
  const intervalSlider = slider('Every N seconds', 1, 30, 1, autoIntervalSec, v => {
    autoIntervalSec = v;
    if (autoCb.checked) restartAuto();
  });
  actions.appendChild(intervalSlider);

  async function safeRun() {
    if (autoBusy) return;          // Skip if previous still running
    autoBusy = true;
    try { await runAnalysis({ persistent: true }); }
    catch (err) {
      console.warn('[auto-relabel] cycle failed:', err);
      scene.showResponse(String(err.message || err), { badge: 'Error', ttlMs: 5000 });
    } finally { autoBusy = false; }
  }
  function startAuto() {
    stopAuto();
    // Pick a sensible default for the freshly-active model on first
    // toggle-on (user can still override via the slider).
    const active = scene._aiState?.activeModel;
    autoIntervalSec = defaultIntervalForModel(active);
    intervalSlider.querySelector('input').value = String(autoIntervalSec);
    intervalSlider.querySelector('span').textContent = `Every N seconds: ${autoIntervalSec.toFixed(2)}`;
    safeRun();   // immediate first pass
    autoTimer = setInterval(safeRun, autoIntervalSec * 1000);
    analyse.disabled = true;
    analyse.textContent = '⟳ Auto-relabel running';
  }
  function stopAuto() {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    analyse.disabled = false;
    analyse.textContent = '✦ Analyse current scene';
    autoBusy = false;
  }
  function restartAuto() {
    if (!autoCb.checked) return;
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(safeRun, autoIntervalSec * 1000);
  }
  autoCb.addEventListener('change', () => {
    if (autoCb.checked) startAuto();
    else stopAuto();
  });

  // Text-to-speech toggle. Zero config: every modern browser ships
  // speechSynthesis. When ON, every settled response is spoken aloud.
  // Banner has its own 🔊 button for on-demand speak too.
  const ttsRow = row();
  const ttsLbl = document.createElement('label');
  ttsLbl.className = 'experiment-toggle';
  const ttsCb = document.createElement('input');
  ttsCb.type = 'checkbox';
  ttsCb.addEventListener('change', () => scene.setAutoSpeak?.(ttsCb.checked));
  ttsLbl.appendChild(ttsCb);
  const ttsSpan = document.createElement('span');
  ttsSpan.innerHTML = '🔊 Speak responses aloud <span class="meta">— Web Speech API, zero install</span>';
  ttsLbl.appendChild(ttsSpan);
  ttsRow.appendChild(ttsLbl);
  actions.appendChild(ttsRow);

  // Cancel auto on scene swap or active-model change — running
  // inference during a load is wasted work and the banner would lie.
  scene.addEventListener('scene-loading', () => { if (autoCb.checked) { autoCb.checked = false; stopAuto(); } });
  sel.addEventListener('change', () => { if (autoCb.checked) { autoCb.checked = false; stopAuto(); } });

  // ── Per-model sub-details ─────────────────────────────────────────────
  const modelsDet = document.createElement('details');
  modelsDet.className = 'tree-group';
  const modelsSum = document.createElement('summary');
  modelsSum.textContent = 'Models';
  modelsDet.appendChild(modelsSum);
  host.appendChild(modelsDet);

  for (const m of models) {
    const det = document.createElement('details');
    det.className = 'tree-group tree-sub';
    const sum = document.createElement('summary');
    sum.textContent = m.label;
    det.appendChild(sum);
    const iOS = isIOS();
    const warn = (m.mobileWarning && iOS) ? `<div class="warn-row">⚠ ${esc(m.mobileWarning)}</div>` : '';
    const body = document.createElement('div');
    body.className = 'menu-body ai-model-body';
    body.innerHTML = `
      <div class="credit">
        <span class="meta">${esc(m.provider)} · ${esc(m.runtime)}</span><br>
        <span class="meta">${esc(m.sizeHint)} · ${m.multimodal ? 'multimodal (text + image)' : 'text only'}</span>
        ${m.notes ? `<br><span class="meta">${esc(m.notes)}</span>` : ''}
      </div>
      ${warn}
      <div class="ai-model-status credit">checking…</div>
      <div class="ai-model-action"></div>
      ${m.clearCache ? `<button class="ai-clear-cache" title="Delete cached weights for this model">Clear cached weights</button>` : ''}
    `;
    det.appendChild(body);
    m._statusEl = body.querySelector('.ai-model-status');
    m._actionEl = body.querySelector('.ai-model-action');
    const clearBtn = body.querySelector('.ai-clear-cache');
    if (clearBtn) clearBtn.addEventListener('click', async () => {
      clearBtn.disabled = true;
      try {
        const did = await m.clearCache();
        clearBtn.textContent = did ? 'Cleared' : 'Nothing to clear';
        setTimeout(() => { clearBtn.textContent = 'Clear cached weights'; clearBtn.disabled = false; refresh(); }, 1500);
      } catch (e) {
        clearBtn.textContent = `Failed: ${e.message}`;
        clearBtn.disabled = false;
      }
    });
    modelsDet.appendChild(det);
  }

  setTimeout(refresh, 0);

  function setModelAction(m, html) { m._actionEl.innerHTML = html; }

  function wireSetup(m, a) {
    if (a === 'available') {
      setModelAction(m, `<button class="ai-activate">Use this model</button>`);
      m._actionEl.querySelector('.ai-activate').addEventListener('click', () => {
        sel.value = m.id; activeId = m.id; refresh();
      });
    } else if (a === 'downloadable') {
      // Two-step: explicit confirm before pulling weights down.
      setModelAction(m, `
        <button class="ai-setup">Set up (download ~${m.downloadGB.toFixed(1)} GB)</button>
      `);
      m._actionEl.querySelector('.ai-setup').addEventListener('click', () => {
        setModelAction(m, `
          <div class="ai-confirm">
            <span>Download <strong>${m.sizeHint}</strong> from the WebLLM CDN?
              Stored in browser Cache Storage; cleared by clearing site data.</span>
            <div class="row">
              <button class="ai-confirm-yes">Download</button>
              <button class="ai-confirm-no">Cancel</button>
            </div>
          </div>
        `);
        m._actionEl.querySelector('.ai-confirm-no').addEventListener('click', () => wireSetup(m, 'downloadable'));
        m._actionEl.querySelector('.ai-confirm-yes').addEventListener('click', () => doDownload(m));
      });
    } else if (a === 'downloading') {
      // Already in flight — let the progress box keep updating.
    } else {
      setModelAction(m, ''); // unavailable: nothing to do
    }
  }

  async function doDownload(m) {
    setModelAction(m, `
      <div class="ai-progress">
        <div class="ai-progress-bar"><div class="ai-progress-fill" style="width:0%"></div></div>
        <span class="ai-progress-text">Starting…</span>
      </div>
    `);
    const fill = m._actionEl.querySelector('.ai-progress-fill');
    const text = m._actionEl.querySelector('.ai-progress-text');
    try {
      await m.ensureReady((p) => {
        if (typeof p.progress === 'number') fill.style.width = `${Math.round(p.progress * 100)}%`;
        if (p.text) text.textContent = p.text;
      });
      m._statusEl.textContent = 'ready';
      sel.value = m.id; activeId = m.id;
      refresh();
    } catch (e) {
      m._statusEl.innerHTML = `<span class="warn">failed: ${esc(e.message)}</span>`;
      wireSetup(m, 'downloadable');
    }
  }

  async function refresh() {
    for (const m of models) {
      try {
        const a = await m.availability();
        m._statusEl.textContent = `status: ${a}`;
        wireSetup(m, a);
      } catch (e) {
        m._statusEl.innerHTML = `<span class="warn">error: ${esc(e.message)}</span>`;
      }
    }
    const active = models.find(m => m.id === activeId);
    if (!active) return;
    try {
      const a = await active.availability();
      status.innerHTML = `Active: <strong>${esc(active.label)}</strong><br>
        <span class="meta">${esc(active.provider)} · ${esc(active.runtime)} · ${esc(a)}</span>`;
      analyse.disabled = (a === 'unavailable');
    } catch (e) {
      status.textContent = `Active: ${active.label} — error: ${e.message}`;
      analyse.disabled = true;
    }
  }

  async function runAnalysis({ persistent = false } = {}) {
    const active = models.find(m => m.id === activeId);
    if (!active) return;
    const badge = active.label.split(' (')[0];
    // In persistent (auto-relabel) mode, the banner stays put until
    // the user dismisses it or toggle is off; ttl=30 s otherwise.
    const finalTtl = persistent ? 0 : 30000;
    scene.showResponse('Capturing scene…', { badge });
    const blob = await scene.captureSnapshot();
    const bitmap = await createImageBitmap(blob);
    const currentScene = scene.currentScene?.();
    const hint = currentScene ? `Scene: "${currentScene.title}" by ${currentScene.author} (${currentScene.category}${currentScene.subcategory ? '/' + currentScene.subcategory : ''}).` : '';
    if ((await active.availability()) !== 'available') {
      scene.showResponse('Preparing model…', { badge });
      await active.ensureReady((p) => {
        const pct = p.progress != null ? ` ${Math.round(p.progress * 100)}%` : '';
        scene.showResponse(`${p.text || 'Loading'}${pct}`, { badge });
      });
    } else {
      scene.showResponse('Thinking…', { badge });
    }
    const reply = await active.describe(bitmap, hint, currentScene);
    scene.showResponse(String(reply).trim(), { badge, ttlMs: finalTtl });
  }
}

// Browser-kind detection (rough, UA-based — fine for messaging).
function browserKind() {
  if (typeof navigator === 'undefined') return 'unknown';
  const ua = navigator.userAgent;
  if (/CriOS/.test(ua))  return 'chrome-ios';
  if (/FxiOS/.test(ua))  return 'firefox-ios';
  if (/EdgiOS/.test(ua)) return 'edge-ios';
  if (/iPad|iPhone|iPod/.test(ua)) return 'safari-ios';
  if (/Edg\//.test(ua))     return 'edge';
  if (/Chrome\//.test(ua))  return 'chrome';
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Safari\//.test(ua))  return 'safari';
  return 'unknown';
}

// ── Helpers ─────────────────────────────────────────────────────────────
function makeSection(title, openByDefault, band) {
  const det = document.createElement('details');
  det.className = 'menu-section';
  if (band) det.dataset.band = band;
  det.open = !!openByDefault;
  const sum = document.createElement('summary');
  sum.textContent = title;
  det.appendChild(sum);
  const body = document.createElement('div');
  body.className = 'menu-body';
  det.appendChild(body);
  return { el: det, body };
}

function slider(label, min, max, step, value, onChange) {
  const r = row();
  const lbl = span(`${label}: ${(+value).toFixed(2)}`);
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
  r.appendChild(lbl);
  r.appendChild(input);
  return r;
}

function row()        { const e = document.createElement('div'); e.className = 'row'; return e; }
function span(t)      { const e = document.createElement('span'); e.textContent = t; return e; }
function emptyHint(t) { const e = document.createElement('div'); e.className = 'credit'; e.textContent = t; return e; }
function cap(s)       { return s.charAt(0).toUpperCase() + s.slice(1); }
function humanise(n)  { return n.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }
function esc(s)       { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function escAttr(s)   { return esc(s); }

// Auto-attach
const tryAttach = () => {
  const scene = document.querySelector('splat-scene');
  if (!scene) return false;
  if (scene._app) { attach(scene); }
  else scene.addEventListener('ready', () => attach(scene), { once: true });
  return true;
};
if (!tryAttach()) {
  const obs = new MutationObserver(() => { if (tryAttach()) obs.disconnect(); });
  obs.observe(document.body, { childList: true, subtree: true });
}
