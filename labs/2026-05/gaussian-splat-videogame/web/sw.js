// Service worker for the Gaussian Splat Labs site.
//
// Strategies (all GET, all 200 responses; opaque responses cached for
// cross-origin model files):
//   • Same-origin app shell (HTML / CSS / JS / catalog.json) →
//     stale-while-revalidate. Page comes up from cache instantly; a
//     fresh copy lands in the background for next load.
//   • CDN files we know we want to keep (PlayCanvas splat CDN, HF
//     model weights, jsDelivr engine code) → cache-first. Once a file
//     is in cache it never round-trips again until a cache bust.
//   • Everything else: pass through (the browser's HTTP cache handles
//     it normally).
//
// Cache names are versioned so a bumped CACHE_VERSION cleanly evicts
// the old set on activate. To force a refresh from a deployed page:
//   navigator.serviceWorker.controller.postMessage('clearAll')
// (handled below).

const CACHE_VERSION = 'v4';
const SHELL_CACHE   = `splat-shell-${CACHE_VERSION}`;
const RUNTIME_CACHE = `splat-runtime-${CACHE_VERSION}`;

// Files that should always be cached on install so the app shell
// works fully offline after the very first load.
const SHELL_FILES = [
  './',
  './index.html',
  './styles.css',
  './catalog.json',
  './components/splat-scene.js',
  './components/hamburger-menu.js',
  './experiments/blind-cane.js',
  './experiments/wubwub.js',
  './splatworld/world.js',
  './splatworld/phone-cane.js',
  './splatworld/tour.js',
  './splatworld/drone.js',
  './splatworld/sonar.js',
  './ai/prompt-api.js',
  './ai/models.js',
  './ai/transformers-js.js',
  './ai/mediapipe.js',
];

// Cross-origin hosts whose responses we proactively cache.
const CACHEABLE_HOSTS = new Set([
  'd28zzqy0iyovbz.cloudfront.net',   // PlayCanvas splat CDN (scene weights)
  'huggingface.co',                  // Model weights (HF)
  'cdn-lfs.huggingface.co',          // HF LFS resolved URLs
  'cas-bridge.xethub.hf.co',         // HF Xet-backed storage
  'cdn.jsdelivr.net',                // Engine code (Transformers.js, WebLLM, MediaPipe)
  'esm.run',                         // Redirector to jsDelivr
  's3-eu-west-1.amazonaws.com',      // SuperSplat thumbnails
  'storage.googleapis.com',          // MediaPipe model weights
  'raw.githubusercontent.com',       // PlayCanvas engine (mjs) fallback
]);

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    // addAll fails atomically; tolerate individual misses so a deploy
    // hiccup doesn't stop the whole install.
    await Promise.all(SHELL_FILES.map(url =>
      cache.add(url).catch(err => console.warn('[sw] precache miss', url, err))
    ));
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter(k => k !== SHELL_CACHE && k !== RUNTIME_CACHE)
        .map(k => caches.delete(k))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => {
  if (event.data === 'clearAll') {
    event.waitUntil((async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
    })());
  } else if (event.data === 'status') {
    event.waitUntil(reportStatus(event.source));
  }
});

async function reportStatus(client) {
  const keys = await caches.keys();
  const counts = {};
  let total = 0;
  for (const name of keys) {
    const cache = await caches.open(name);
    const reqs = await cache.keys();
    counts[name] = reqs.length;
    total += reqs.length;
  }
  client?.postMessage({ kind: 'sw-status', counts, total });
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Same-origin app shell
  if (url.origin === self.location.origin) {
    // Skip the SW for chrome-extension:// etc., and for sw.js itself
    if (url.pathname.endsWith('/sw.js')) return;
    event.respondWith(staleWhileRevalidate(req, SHELL_CACHE));
    return;
  }

  // Cross-origin known CDNs
  if (CACHEABLE_HOSTS.has(url.hostname)) {
    event.respondWith(cacheFirst(req, RUNTIME_CACHE));
    return;
  }
  // Everything else: let the network handle it
});

async function staleWhileRevalidate(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  const networkPromise = fetch(req).then((resp) => {
    if (resp && resp.ok && resp.status < 400 && resp.type !== 'opaqueredirect') {
      // Don't fail the whole pipeline if a put() throws (quota etc).
      cache.put(req, resp.clone()).catch(() => {});
    }
    return resp;
  }).catch(() => cached);  // network down → fall back to cache
  return cached || networkPromise;
}

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  if (cached) return cached;
  try {
    // Cross-origin: 'no-cors' would give an opaque response. We want
    // the proper response when possible; HF / jsDelivr / CloudFront
    // all send Access-Control-Allow-Origin: *.
    const resp = await fetch(req);
    if (resp && (resp.ok || resp.type === 'opaque') && resp.status !== 206) {
      cache.put(req, resp.clone()).catch(() => {});
    }
    return resp;
  } catch (e) {
    return new Response(
      'Offline, and this resource is not cached yet.',
      { status: 503, statusText: 'Offline' },
    );
  }
}
