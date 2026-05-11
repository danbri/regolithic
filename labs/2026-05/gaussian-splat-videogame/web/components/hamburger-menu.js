// Adds a hamburger button + slide-out settings/config panel to <splat-scene>.
// Side-effect module: auto-attaches to the first <splat-scene> on the page
// once it emits 'ready'.
//
// Panel contents:
//   • Scene credit (from the active catalog entry)
//   • Render settings (move speed, FOV)
//   • Experiment toggles (registered via splat-scene.registerExperiment)
//   • XR options (request-only — entry is via the canvas-corner button)

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

  // ── Sections ─────────────────────────────────────────────────────────
  const creditSec = section('Scene');
  panel.appendChild(creditSec);

  const renderSec = section('Render');
  panel.appendChild(renderSec);
  renderSec.appendChild(slider('Move speed', 0.2, 8, 0.1, scene._inputState.moveSpeed, v => {
    scene._inputState.moveSpeed = v;
  }));
  renderSec.appendChild(slider('FOV', 40, 110, 1, scene.camera?.camera?.fov ?? 70, v => {
    if (scene.camera?.camera) scene.camera.camera.fov = v;
  }));

  const xpSec = section('Experiments');
  panel.appendChild(xpSec);
  xpSec.appendChild(emptyHint('(none registered yet)'));

  const updateCredit = (sceneData) => {
    creditSec.querySelector('.body')?.remove();
    const body = document.createElement('div');
    body.className = 'body credit';
    body.innerHTML = sceneData
      ? `<strong>${escapeHtml(sceneData.title)}</strong><br>
         by ${escapeHtml(sceneData.author)} —
         <a href="${escapeAttr(sceneData.license_url)}" target="_blank" rel="noopener">
           ${escapeHtml(sceneData.license)}
         </a>`
      : 'No scene loaded.';
    creditSec.appendChild(body);
  };

  scene.addEventListener('scene-loaded', (e) => updateCredit(e.detail.scene));
  scene.addEventListener('scene-loading', (e) => updateCredit(e.detail.scene));

  scene.addEventListener('experiment-registered', () => rebuildExperimentSection());
  function rebuildExperimentSection() {
    xpSec.querySelectorAll('.body').forEach(n => n.remove());
    const names = scene.experiments();
    if (names.length === 0) {
      xpSec.appendChild(emptyHint('(none registered yet)'));
      return;
    }
    for (const name of names) {
      const wrap = document.createElement('div');
      wrap.className = 'body';
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = scene.isExperimentEnabled(name);
      cb.addEventListener('change', () => scene.toggleExperiment(name, cb.checked));
      lbl.appendChild(cb);
      const span = document.createElement('span');
      span.textContent = humanise(name);
      lbl.appendChild(span);
      wrap.appendChild(lbl);

      // Optional per-experiment settings panel
      const ex = scene._experiments.get(name);
      if (typeof ex?.renderSettings === 'function') {
        const sub = document.createElement('div');
        sub.style.paddingLeft = '24px';
        ex.renderSettings(sub);
        wrap.appendChild(sub);
      }
      xpSec.appendChild(wrap);
    }
  }
}

function section(title) {
  const s = document.createElement('div');
  s.className = 'menu-section';
  const h = document.createElement('h3');
  h.textContent = title;
  s.appendChild(h);
  return s;
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

function emptyHint(text) {
  const el = document.createElement('div');
  el.className = 'body credit';
  el.textContent = text;
  return el;
}

function humanise(name) {
  return name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

// Auto-attach to first <splat-scene>
const tryAttach = () => {
  const scene = document.querySelector('splat-scene');
  if (!scene) return false;
  if (scene._app) {
    attach(scene);
  } else {
    scene.addEventListener('ready', () => attach(scene), { once: true });
  }
  return true;
};

if (!tryAttach()) {
  // Wait for the splat-scene element to be added
  const obs = new MutationObserver(() => { if (tryAttach()) obs.disconnect(); });
  obs.observe(document.body, { childList: true, subtree: true });
}
