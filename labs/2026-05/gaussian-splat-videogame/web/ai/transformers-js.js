// Transformers.js (@huggingface/transformers) — WebGPU-backed multimodal.
//
// This is the iOS-viable path for *true* image input: a 4-bit quantized
// PaliGemma running through ONNX Runtime Web on top of WebGPU (which
// maps to Metal on Apple devices). Transformers.js handles model
// caching via the Cache API automatically.
//
// CDN entry (ESM): https://esm.run/@huggingface/transformers proxies
// the npm package's ESM build. First call pays for the engine code
// plus the chosen model's weights; subsequent loads come from cache.

const TJS_CDN = 'https://esm.run/@huggingface/transformers';
let _tjsModule = null;

async function loadTransformersJS(onProgress) {
  if (_tjsModule) return _tjsModule;
  onProgress?.({ stage: 'engine', text: 'Loading Transformers.js (ONNX + WebGPU)…' });
  _tjsModule = await import(/* @vite-ignore */ TJS_CDN);
  // Pin to remote-only models; never hunt for local paths.
  if (_tjsModule.env) _tjsModule.env.allowLocalModels = false;
  return _tjsModule;
}

export class TransformersJSModel {
  constructor({ id, label, provider, hfRepo, task, prompt, dtype, sizeHint, downloadGB, notes, iosSafe = true, postProcess }) {
    this.id = id;
    this.label = label;
    this.provider = provider;
    this.runtime = 'WebGPU via Transformers.js';
    this.hfRepo = hfRepo;
    this.task = task;             // e.g. 'image-text-to-text'
    this.prompt = prompt;         // model-specific instruction prefix
    this.dtype = dtype;           // 'q4' for iOS-safe; 'fp16' on desktop
    this.sizeHint = sizeHint;
    this.downloadGB = downloadGB;
    this.multimodal = true;
    this.notes = notes;
    this.iosSafe = iosSafe;
    this._postProcess = postProcess;
    this._pipe = null;
    this._initing = false;
  }

  async availability() {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return 'unavailable';
    if (this._pipe) return 'available';
    if (this._initing) return 'downloading';
    return 'downloadable';
  }

  async ensureReady(onProgress) {
    if (this._pipe || this._initing) return;
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available. iOS Safari 18+ ships it; older iOS may need a flag. Falling back to CPU/WASM would freeze on a 3B model.');
    }
    this._initing = true;
    try {
      const { pipeline } = await loadTransformersJS(onProgress);
      this._pipe = await pipeline(this.task, this.hfRepo, {
        device: 'webgpu',
        dtype: this.dtype,
        progress_callback: (data) => {
          // data: { status, name, file, progress, loaded, total }
          const pct = (data.progress != null) ? data.progress / 100 : undefined;
          const file = data.file || data.name || '';
          let text;
          if (data.status === 'progress' && data.total) {
            text = `Loading ${file}: ${fmt(data.loaded)} / ${fmt(data.total)}`;
          } else if (data.status === 'done') {
            text = `Fetched ${file}`;
          } else {
            text = `${data.status} ${file}`.trim();
          }
          onProgress?.({ stage: 'model', progress: pct, text });
        },
      });
    } finally {
      this._initing = false;
    }
  }

  async describe(image, hint /*, scene */) {
    if (!this._pipe) await this.ensureReady();

    // Normalise the input → an object URL the pipeline can fetch.
    // image may be HTMLCanvasElement, ImageBitmap, or Blob.
    const { url, revoke } = await imageToObjectUrl(image);
    try {
      const promptText = this.prompt || '';
      // Florence-2 wants only a task tag (e.g. <MORE_DETAILED_CAPTION>),
      // not a natural-language preamble. Other models may take both.
      const result = await this._pipe(url, promptText);
      let text;
      if (Array.isArray(result) && result[0]?.generated_text) text = result[0].generated_text;
      else if (result?.generated_text) text = result.generated_text;
      else if (typeof result === 'string') text = result;
      else text = JSON.stringify(result);
      return this._postProcess ? this._postProcess(text) : text;
    } finally {
      revoke?.();
    }
  }

  teardown() {
    // Transformers.js doesn't expose explicit pipeline disposal; null it
    // and let GC reclaim. The downloaded weights stay in Cache API.
    this._pipe = null;
  }

  async clearCache() {
    if (typeof caches === 'undefined') return false;
    const keys = await caches.keys();
    // Transformers.js puts entries in a cache named 'transformers-cache'
    // (configurable). Match on cache name or any entry whose URL
    // contains the repo path.
    const mine = keys.filter(k => /transformers|huggingface/i.test(k));
    await Promise.all(mine.map(k => caches.delete(k)));
    // Also wipe any entry whose request URL contains the repo
    for (const name of await caches.keys()) {
      const cache = await caches.open(name);
      const reqs = await cache.keys();
      for (const r of reqs) {
        if (r.url.includes(this.hfRepo)) await cache.delete(r);
      }
    }
    this.teardown();
    return true;
  }
}

async function imageToObjectUrl(image) {
  if (image instanceof Blob) {
    const url = URL.createObjectURL(image);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (image instanceof HTMLCanvasElement) {
    const blob = await new Promise((res) => image.toBlob(res, 'image/png'));
    if (!blob) throw new Error('canvas.toBlob returned null');
    const url = URL.createObjectURL(blob);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) {
    // Draw bitmap into a temp canvas, then read out as a blob.
    const canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(image.width, image.height)
      : document.createElement('canvas');
    if (!(canvas instanceof OffscreenCanvas)) {
      canvas.width = image.width;
      canvas.height = image.height;
    }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const blob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: 'image/png' })
      : await new Promise((res) => canvas.toBlob(res, 'image/png'));
    if (!blob) throw new Error('failed to encode bitmap → blob');
    const url = URL.createObjectURL(blob);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (typeof image === 'string') {
    // Already a URL
    return { url: image, revoke: null };
  }
  throw new Error('Unsupported image input type for TransformersJSModel.describe');
}

function fmt(n) {
  if (!n) return '?';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
