// MediaPipe object detection — different runtime from Transformers.js.
//
// We've hit a wall with ONNX Runtime Web on iOS Safari (tabs crash
// even on tiny encoder-only models). MediaPipe Tasks ships its own
// TFLite-backed runtime, purpose-built for mobile browsers, with a
// completely separate code path. Different bytes, different bugs.
//
// Engine: @mediapipe/tasks-vision (~140 KB ESM + ~11 MB WASM).
// Model:  EfficientDet Lite 0 (TFLite int8, ~5 MB) — ~80 COCO classes.
//
// The CDN-loaded engine + service-worker caching means subsequent
// loads come straight from cache.

const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/+esm';
const MP_WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

let _mpModule = null;
let _mpVision = null;

async function loadMediaPipe(onProgress) {
  if (_mpModule && _mpVision) return { mp: _mpModule, vision: _mpVision };
  onProgress?.({ stage: 'engine', text: 'Loading MediaPipe…' });
  _mpModule = await import(/* @vite-ignore */ MP_CDN);
  onProgress?.({ stage: 'engine', text: 'Initialising MediaPipe runtime…' });
  _mpVision = await _mpModule.FilesetResolver.forVisionTasks(MP_WASM_BASE);
  return { mp: _mpModule, vision: _mpVision };
}

export class MediaPipeDetector {
  constructor({
    id, label, modelUrl,
    sizeHint = '~5 MB', downloadGB = 0.005,
    delegate = 'CPU',     // 'CPU' | 'GPU' (GPU = WebGL)
    threshold = 0.3,
    topK = 10,
    notes,
    iosSafe = true,
    mobileWarning,
  }) {
    this.id = id;
    this.label = label;
    this.provider = 'Google · MediaPipe';
    this.runtime = `TFLite-${delegate} via MediaPipe Tasks`;
    this.modelUrl = modelUrl;
    this.sizeHint = sizeHint;
    this.downloadGB = downloadGB;
    this.mode = 'object-detection';
    this.multimodal = true;
    this.iosSafe = iosSafe;
    this.mobileWarning = mobileWarning;
    this.notes = notes;
    this.delegate = delegate;
    this.detectionThreshold = threshold;
    this.detectionTopK = topK;
    this._detector = null;
    this._initing = false;
  }

  async availability() {
    if (this._detector) return 'available';
    if (this._initing) return 'downloading';
    return 'downloadable';
  }

  async ensureReady(onProgress) {
    if (this._detector || this._initing) return;
    this._initing = true;
    try {
      const { mp, vision } = await loadMediaPipe(onProgress);
      onProgress?.({ stage: 'model', text: `Loading ${this.label}…` });
      this._detector = await mp.ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: this.modelUrl,
          delegate: this.delegate,
        },
        scoreThreshold: this.detectionThreshold,
        maxResults: this.detectionTopK,
        runningMode: 'IMAGE',
      });
      onProgress?.({ stage: 'model', text: 'Ready' });
    } finally {
      this._initing = false;
    }
  }

  // Structured output, normalised to Transformers.js's shape so the
  // Drone doesn't need a special case.
  async detect(image) {
    if (!this._detector) await this.ensureReady();
    const input = await toMPImage(image);
    const result = this._detector.detect(input);
    const detections = result?.detections ?? [];
    return detections.map(d => {
      const cat = (d.categories && d.categories[0]) || {};
      const bb = d.boundingBox;
      return {
        score: cat.score ?? 0,
        label: cat.categoryName ?? 'unknown',
        box: bb ? {
          xmin: bb.originX,
          ymin: bb.originY,
          xmax: bb.originX + bb.width,
          ymax: bb.originY + bb.height,
        } : null,
      };
    });
  }

  async describe(image /*, hint, scene */) {
    const detections = await this.detect(image);
    if (detections.length === 0) {
      return '(no objects detected — move the camera, or try a bigger model)';
    }
    // De-duplicate by label, keep best score
    const seen = new Map();
    for (const d of detections) {
      const cur = seen.get(d.label);
      if (!cur || d.score > cur.score) seen.set(d.label, d);
    }
    const list = [...seen.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, this.detectionTopK)
      .map(d => `${d.label} (${Math.round(d.score * 100)}%)`);
    return `Detected: ${list.join(' · ')}`;
  }

  teardown() {
    try { this._detector?.close?.(); } catch {}
    this._detector = null;
  }

  async clearCache() {
    // The MediaPipe runtime + model files live in the browser HTTP
    // cache + our service worker's runtime cache. The SW handles
    // cleanup globally; nothing model-specific to do here.
    return false;
  }
}

// MediaPipe accepts HTMLCanvasElement, HTMLImageElement, ImageBitmap,
// HTMLVideoElement directly. Coerce other inputs into a canvas.
async function toMPImage(image) {
  if (image instanceof HTMLCanvasElement) return image;
  if (image instanceof HTMLImageElement) return image;
  if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) return image;
  if (image instanceof Blob) {
    const bmp = await createImageBitmap(image);
    return bmp;
  }
  if (typeof image === 'string') {
    // URL → fetch → blob → bitmap
    const blob = await (await fetch(image)).blob();
    return await createImageBitmap(blob);
  }
  throw new Error('Unsupported image input for MediaPipeDetector');
}
