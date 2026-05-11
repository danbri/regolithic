// Hamburger menu with collapsible sections (using native <details>).
//
// Sections:
//   • Catalog    — scenes grouped by category, with author / license /
//                   size, click-to-load. Replaces the old top-of-screen
//                   chip picker entirely.
//   • Scene      — current scene credit (author, license, link).
//   • Render     — move speed, FOV, orbit/fly mode.
//   • Experiments — toggle each registered experiment; per-experiment
//                   settings panels are nested <details> (collapsed by
//                   default) so the menu stays scannable when many
//                   sliders exist.
//
// Side-effect module: auto-attaches to the first <splat-scene> on the
// page once it emits 'ready'.

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

// ── Catalog tree ─────────────────────────────────────────────────────────
function buildCatalog(host, scene) {
  const catalog = scene._catalog;
  if (!catalog) {
    host.appendChild(emptyHint('Catalog not loaded.'));
    return;
  }
  const cats = catalog.categories || {};
  const groups = new Map();
  for (const s of catalog.scenes) {
    const c = s.category || 'other';
    if (!groups.has(c)) groups.set(c, []);
    groups.get(c).push(s);
  }
  const ordered = Array.from(groups.keys()).sort((a, b) => {
    return (cats[a]?.order ?? 999) - (cats[b]?.order ?? 999);
  });

  for (const cat of ordered) {
    const det = document.createElement('details');
    det.open = true;
    det.className = 'tree-group';
    const sum = document.createElement('summary');
    sum.textContent = `${cats[cat]?.title ?? cap(cat)} (${groups.get(cat).length})`;
    det.appendChild(sum);

    // Sort within group by size
    const items = groups.get(cat).slice().sort((a, b) => (a.size_mb ?? 0) - (b.size_mb ?? 0));
    for (const s of items) {
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
      det.appendChild(leaf);
    }
    host.appendChild(det);
  }

  // Highlight active leaf as scenes change
  scene.addEventListener('scene-loaded', (e) => {
    const id = e.detail.scene.id;
    for (const el of host.querySelectorAll('.tree-leaf')) {
      el.setAttribute('aria-pressed', String(el.dataset.sceneId === id));
    }
  });
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
