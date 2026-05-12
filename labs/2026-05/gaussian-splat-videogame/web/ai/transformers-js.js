// Transformers.js (@huggingface/transformers) — WebGPU-backed multimodal.
//
// Supports two API surfaces per model:
//   • mode 'pipeline'  — generic `pipeline('image-text-to-text', repo)`.
//     Works for Florence-2 and most VLMs.
//   • mode 'paligemma' — explicit AutoProcessor +
//     PaliGemmaForConditionalGeneration (per HF's own browser-PaliGemma
//     example). Required for PaliGemma 2 in current Transformers.js:
//     the pipeline route doesn't drive PaliGemma's special-token format
//     consistently across versions.
//
// CDN: latest jsDelivr ESM build (PaliGemma support landed in 3.2.0;
// using @latest gets a 4.x build).

const TJS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers/+esm';
let _tjsModule = null;

async function loadTransformersJS(onProgress) {
  if (_tjsModule) return _tjsModule;
  onProgress?.({ stage: 'engine', text: 'Loading Transformers.js (ONNX + WebGPU)…' });
  _tjsModule = await import(/* @vite-ignore */ TJS_CDN);
  if (_tjsModule.env) _tjsModule.env.allowLocalModels = false;
  return _tjsModule;
}

export class TransformersJSModel {
  constructor({
    id, label, provider, hfRepo,
    mode = 'pipeline',          // 'pipeline' | 'paligemma'
    task = 'image-text-to-text',
    inputFormat = 'url-prompt', // 'messages' | 'url-prompt' | 'url-only'
    prompt,
    dtype = 'q4f16',
    sizeHint, downloadGB,
    notes, iosSafe = true, mobileWarning,
    postProcess,
    maxNewTokens = 80,
  }) {
    this.id = id;
    this.label = label;
    this.provider = provider;
    this.runtime = 'WebGPU via Transformers.js';
    this.hfRepo = hfRepo;
    this.mode = mode;
    this.task = task;
    this.inputFormat = inputFormat;
    this.prompt = prompt;
    this.dtype = dtype;
    this.sizeHint = sizeHint;
    this.downloadGB = downloadGB;
    this.multimodal = true;
    this.notes = notes;
    this.iosSafe = iosSafe;
    this.mobileWarning = mobileWarning;
    this.maxNewTokens = maxNewTokens;
    this._postProcess = postProcess;
    this._engine = null;
    this._initing = false;
  }

  async availability() {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return 'unavailable';
    if (this._engine) return 'available';
    if (this._initing) return 'downloading';
    return 'downloadable';
  }

  async ensureReady(onProgress) {
    if (this._engine || this._initing) return;
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available. iOS Safari 18+ ships it; CPU fallback would freeze on a VLM.');
    }
    this._initing = true;
    try {
      const mod = await loadTransformersJS(onProgress);
      const progress_callback = (data) => {
        const pct = (data.progress != null) ? data.progress / 100 : undefined;
        const file = data.file || data.name || '';
        let text;
        if (data.status === 'progress' && data.total) {
          text = `Loading ${file}: ${fmtBytes(data.loaded)} / ${fmtBytes(data.total)}`;
        } else if (data.status === 'done') {
          text = `Fetched ${file}`;
        } else {
          text = `${data.status} ${file}`.trim();
        }
        onProgress?.({ stage: 'model', progress: pct, text });
      };

      if (this.mode === 'paligemma') {
        const processor = await mod.AutoProcessor.from_pretrained(this.hfRepo, { progress_callback });
        const model     = await mod.PaliGemmaForConditionalGeneration.from_pretrained(this.hfRepo, {
          device: 'webgpu',
          dtype: this.dtype,
          progress_callback,
        });
        this._engine = { kind: 'paligemma', processor, model, mod };
      } else {
        const pipe = await mod.pipeline(this.task, this.hfRepo, {
          device: 'webgpu',
          dtype: this.dtype,
          progress_callback,
        });
        this._engine = { kind: 'pipeline', pipe, mod };
      }
    } finally {
      this._initing = false;
    }
  }

  async describe(image, hint /*, scene */) {
    if (!this._engine) await this.ensureReady();
    const { kind, mod } = this._engine;

    if (kind === 'paligemma') {
      // Lower-level path: AutoProcessor → PaliGemmaForConditionalGeneration.
      // PaliGemma takes its prompt verbatim (e.g. "caption en"); the
      // article you cited spells it out the same way.
      const rawImage = await rawImageFrom(image, mod);
      const prompt = this.prompt || 'caption en';
      const inputs = await this._engine.processor(rawImage, prompt);
      const outputs = await this._engine.model.generate({
        ...inputs,
        max_new_tokens: this.maxNewTokens,
      });
      const decoded = this._engine.processor.batch_decode(outputs, { skip_special_tokens: true });
      let text = decoded?.[0] ?? '';
      // PaliGemma emits "<prompt> <caption>". Strip the prompt prefix.
      if (text.startsWith(prompt)) text = text.slice(prompt.length).trim();
      return this._postProcess ? this._postProcess(text) : text;
    }

    // pipeline path. Three input shapes the upstream pipeline accepts:
    //   • messages    — chat-style [{role,content:[{type:'image',image},{type:'text',text}]}]
    //                   (required by SmolVLM and other modern VLMs)
    //   • url-prompt  — pipe(url, promptString)  (Florence-2 et al.)
    //   • url-only    — pipe(url)                (pure captioners: ViT-GPT2)
    const { url, revoke } = await imageToObjectUrl(image);
    try {
      let result;
      if (this.inputFormat === 'messages') {
        const messages = [{
          role: 'user',
          content: [
            { type: 'image', image: url },
            { type: 'text',  text: this.prompt || 'Describe this image in 1–3 sentences.' },
          ],
        }];
        result = await this._engine.pipe(messages, { max_new_tokens: this.maxNewTokens });
      } else if (this.inputFormat === 'url-only') {
        result = await this._engine.pipe(url);
      } else {
        result = await this._engine.pipe(url, this.prompt || '');
      }
      const text = extractText(result);
      return this._postProcess ? this._postProcess(text) : text;
    } finally {
      revoke?.();
    }
  }

  teardown() {
    this._engine = null;
  }

  async clearCache() {
    if (typeof caches === 'undefined') return false;
    for (const name of await caches.keys()) {
      const cache = await caches.open(name);
      for (const req of await cache.keys()) {
        if (req.url.includes(this.hfRepo)) await cache.delete(req);
      }
    }
    this.teardown();
    return true;
  }
}

// Convert various image inputs to Transformers.js RawImage.
async function rawImageFrom(image, mod) {
  // RawImage.fromBlob handles Blob directly
  if (image instanceof Blob) return await mod.RawImage.fromBlob(image);
  if (image instanceof HTMLCanvasElement) {
    const blob = await new Promise(r => image.toBlob(r, 'image/png'));
    return await mod.RawImage.fromBlob(blob);
  }
  if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) {
    const canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(image.width, image.height)
      : Object.assign(document.createElement('canvas'), { width: image.width, height: image.height });
    canvas.getContext('2d').drawImage(image, 0, 0);
    const blob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: 'image/png' })
      : await new Promise(r => canvas.toBlob(r, 'image/png'));
    return await mod.RawImage.fromBlob(blob);
  }
  if (typeof image === 'string') return await mod.RawImage.read(image);
  throw new Error('Unsupported image input for TransformersJSModel');
}

// Convert various image inputs to an object URL (for pipeline route).
async function imageToObjectUrl(image) {
  if (image instanceof Blob) {
    const url = URL.createObjectURL(image);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (image instanceof HTMLCanvasElement) {
    const blob = await new Promise(r => image.toBlob(r, 'image/png'));
    if (!blob) throw new Error('canvas.toBlob returned null');
    const url = URL.createObjectURL(blob);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) {
    const canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(image.width, image.height)
      : Object.assign(document.createElement('canvas'), { width: image.width, height: image.height });
    canvas.getContext('2d').drawImage(image, 0, 0);
    const blob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: 'image/png' })
      : await new Promise(r => canvas.toBlob(r, 'image/png'));
    if (!blob) throw new Error('failed to encode bitmap → blob');
    const url = URL.createObjectURL(blob);
    return { url, revoke: () => URL.revokeObjectURL(url) };
  }
  if (typeof image === 'string') return { url: image, revoke: null };
  throw new Error('Unsupported image input type for TransformersJSModel.describe');
}

function fmtBytes(n) {
  if (!n) return '?';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// Extract a text reply from the various shapes Transformers.js
// returns. For chat-messages calls the result is typically
// [{ generated_text: [...messages, { role:'assistant', content:'...' }] }];
// for simple url-prompt calls it's [{ generated_text: '...' }].
function extractText(result) {
  if (typeof result === 'string') return result;
  const first = Array.isArray(result) ? result[0] : result;
  if (!first) return JSON.stringify(result);
  const gt = first.generated_text;
  if (typeof gt === 'string') return gt;
  if (Array.isArray(gt)) {
    const asst = [...gt].reverse().find(m => m?.role === 'assistant');
    if (asst) {
      if (typeof asst.content === 'string') return asst.content;
      if (Array.isArray(asst.content)) {
        return asst.content.map(c => c?.text ?? c?.value ?? '').filter(Boolean).join(' ');
      }
    }
  }
  return JSON.stringify(result);
}
