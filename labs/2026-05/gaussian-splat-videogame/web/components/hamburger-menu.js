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

import { buildRegistry } from '../ai/models.js';

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

  // ── Catalog ───────────────────────────────────────────────────────────
  const catalogSec = makeSection('Catalog', true);
  panel.appendChild(catalogSec.el);
  buildCatalog(catalogSec.body, scene);

  // ── Scene credit ──────────────────────────────────────────────────────
  const sceneSec = makeSection('Scene', true);
  panel.appendChild(sceneSec.el);
  const updateCredit = (s) => {
    sceneSec.body.innerHTML = s
      ? `<div class="credit">
           <strong>${esc(s.title)}</strong><br>
           by ${esc(s.author)} —
           <a href="${escAttr(s.license_url)}" target="_blank" rel="noopener">${esc(s.license)}</a>
           ${s.size_mb ? `<br><span class="meta">${s.size_mb} MB · ${esc(s.format)}</span>` : ''}
         </div>`
      : '<div class="credit">No scene loaded.</div>';
  };
  scene.addEventListener('scene-loading', e => updateCredit(e.detail.scene));
  scene.addEventListener('scene-loaded',  e => updateCredit(e.detail.scene));

  // ── Render ────────────────────────────────────────────────────────────
  const renderSec = makeSection('Render', true);
  panel.appendChild(renderSec.el);
  renderSec.body.appendChild(slider('Move speed', 0.2, 8, 0.1, scene._input.moveSpeed, v => scene._input.moveSpeed = v));
  renderSec.body.appendChild(slider('FOV', 40, 110, 1, scene.camera?.camera?.fov ?? 70, v => {
    if (scene.camera?.camera) scene.camera.camera.fov = v;
  }));
  // Mode toggle as a dropdown
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
  renderSec.body.appendChild(modeRow);

  // ── AI ────────────────────────────────────────────────────────────────
  const aiSec = makeSection('AI', false);
  panel.appendChild(aiSec.el);
  buildAISection(aiSec.body, scene);

  // ── Experiments ───────────────────────────────────────────────────────
  const xpSec = makeSection('Experiments', true);
  panel.appendChild(xpSec.el);
  xpSec.body.appendChild(emptyHint('(none registered yet)'));

  const rebuildExperiments = () => {
    xpSec.body.innerHTML = '';
    const names = scene.experiments();
    if (names.length === 0) {
      xpSec.body.appendChild(emptyHint('(none registered yet)'));
      return;
    }
    for (const name of names) {
      const wrap = document.createElement('div');
      wrap.className = 'experiment';
      const toggleRow = document.createElement('label');
      toggleRow.className = 'experiment-toggle';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = scene.isExperimentEnabled(name);
      cb.addEventListener('change', () => scene.toggleExperiment(name, cb.checked));
      toggleRow.appendChild(cb);
      const lbl = document.createElement('strong');
      lbl.textContent = humanise(name);
      toggleRow.appendChild(lbl);
      wrap.appendChild(toggleRow);

      // Per-experiment settings nested as a closed-by-default <details>
      const ex = scene._experiments.get(name);
      if (typeof ex?.renderSettings === 'function') {
        const sub = document.createElement('details');
        sub.className = 'experiment-settings';
        const sum = document.createElement('summary');
        sum.textContent = 'Settings';
        sub.appendChild(sum);
        const body = document.createElement('div');
        body.className = 'menu-body';
        sub.appendChild(body);
        ex.renderSettings(body);
        wrap.appendChild(sub);
      }
      xpSec.body.appendChild(wrap);
    }
  };
  scene.addEventListener('experiment-registered', rebuildExperiments);
  rebuildExperiments();

  // Trigger initial credit render if a scene is already loaded
  if (scene._currentSceneId) {
    const s = scene._catalog?.scenes.find(x => x.id === scene._currentSceneId);
    if (s) updateCredit(s);
  }
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

// ── AI section ──────────────────────────────────────────────────────────
function buildAISection(host, scene) {
  const models = buildRegistry();
  let activeId = models[0].id;

  // Status / action row
  const actions = document.createElement('div');
  actions.className = 'menu-body';
  host.appendChild(actions);

  const status = document.createElement('div');
  status.className = 'credit';
  status.textContent = 'Checking…';
  actions.appendChild(status);

  // Active-model select
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

  // Analyse button
  const analyse = document.createElement('button');
  analyse.textContent = '✦ Analyse current scene';
  analyse.style.width = '100%';
  analyse.style.padding = '10px';
  analyse.style.marginTop = '6px';
  analyse.addEventListener('click', () => runAnalysis().catch(err => {
    console.error('[ai] analysis failed', err);
    scene.showResponse(String(err.message || err), { badge: 'Error', ttlMs: 8000 });
  }));
  actions.appendChild(analyse);

  // Per-model sub-details
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
    const body = document.createElement('div');
    body.className = 'menu-body';
    body.innerHTML = `
      <div class="credit">
        <span class="meta">${esc(m.provider)} · ${esc(m.runtime)} · ${esc(m.sizeHint)}</span><br>
        ${m.multimodal ? 'multimodal (text + image)' : 'text only'}
        ${m.notes ? `<br><span class="meta">${esc(m.notes)}</span>` : ''}
      </div>
      <div class="row"><span class="model-status">checking…</span>
        <button class="model-setup">Set up</button></div>
    `;
    det.appendChild(body);
    const statusEl = body.querySelector('.model-status');
    const setupBtn = body.querySelector('.model-setup');
    setupBtn.addEventListener('click', async () => {
      try {
        setupBtn.disabled = true;
        statusEl.textContent = 'setting up…';
        await m.ensureReady((bytes) => { statusEl.textContent = `downloading ${formatBytes(bytes)}…`; });
        statusEl.textContent = 'ready';
        sel.value = m.id; activeId = m.id;
      } catch (e) {
        statusEl.textContent = `not ready: ${e.message}`;
      } finally {
        setupBtn.disabled = false;
        refresh();
      }
    });
    m._statusEl = statusEl;
    m._setupBtn = setupBtn;
    modelsDet.appendChild(det);
  }

  // Initial availability checks (deferred so the menu renders quickly)
  setTimeout(refresh, 0);

  async function refresh() {
    const active = models.find(m => m.id === activeId);
    if (!active) return;
    for (const m of models) {
      try {
        const a = await m.availability();
        if (m._statusEl) m._statusEl.textContent = `status: ${a}`;
        if (m._setupBtn) m._setupBtn.disabled = (a === 'available') || (a === 'unavailable' && m.id !== 'chrome-builtin');
      } catch (e) {
        if (m._statusEl) m._statusEl.textContent = `status: error`;
      }
    }
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

  async function runAnalysis() {
    const active = models.find(m => m.id === activeId);
    if (!active) return;
    scene.showResponse('Capturing scene…', { badge: active.label.split(' (')[0], ttlMs: 0 });
    const blob = await scene.captureSnapshot();
    const bitmap = await createImageBitmap(blob);
    scene.showResponse('Thinking…', { badge: active.label.split(' (')[0], ttlMs: 0 });
    const currentScene = scene.currentScene?.();
    const hint = currentScene ? `Scene title: "${currentScene.title}" by ${currentScene.author}.` : '';
    await active.ensureReady?.((bytes) => scene.showResponse(`Downloading model: ${formatBytes(bytes)}`, { badge: 'AI', ttlMs: 0 }));
    const reply = await active.describe(bitmap, hint);
    scene.showResponse(String(reply).trim(), { badge: active.label.split(' (')[0], ttlMs: 30000 });
  }
}

function formatBytes(n) {
  if (!n) return '?';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// ── Helpers ─────────────────────────────────────────────────────────────
function makeSection(title, openByDefault) {
  const det = document.createElement('details');
  det.className = 'menu-section';
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
