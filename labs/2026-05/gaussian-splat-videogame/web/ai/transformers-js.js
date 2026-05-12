// Transformers.js (@huggingface/transformers) — WebGPU-backed multimodal.
//
// Supports four API surfaces per model:
//   • mode 'pipeline'  — generic `pipeline(task, repo)`.
//     For pure captioners (ViT-GPT2, DistilViT) and Florence-2-style
//     models that take a URL + simple prompt.
//   • mode 'paligemma' — AutoProcessor + PaliGemmaForConditionalGeneration.
//     Required by PaliGemma's special-token prompt format.
//   • mode 'smolvlm'   — AutoProcessor + AutoModelForVision2Seq +
//     apply_chat_template. Required by SmolVLM / Idefics3-family VLMs.
//   • mode 'object-detection' — pipeline('object-detection', repo).
//     Output is [{ score, label, box }] — formatted into a comma list
//     for display. Most stable in-browser path: encoder-only, no
//     autoregressive decoder, simple ops, tiny weights.
//
// Caching: Transformers.js puts downloaded weights in Cache API
// (browser Cache Storage); ONNX Runtime Web caches compiled WebGPU
// shaders in IndexedDB. HF CDN sends Cache-Control so first load is
// the only slow one. Plus the site-level service worker on top.
//
// CDN: latest jsDelivr +esm build of @huggingface/transformers.

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
    mode = 'pipeline',
    task = 'image-to-text',
    inputFormat = 'url-prompt',
    prompt,
    dtype = 'q4f16',
    sizeHint, downloadGB,
    notes, iosSafe = true, mobileWarning,
    postProcess,
    maxNewTokens = 80,
    detectionThreshold = 0.3,
    detectionTopK = 10,
    candidateLabels,
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
    this.detectionThreshold = detectionThreshold;
    this.detectionTopK = detectionTopK;
    this.candidateLabels = candidateLabels;
    this._postProcess = postProcess;
    this._engine = null;
    this._initing = false;
  }

  async availability() {
    // Always available — the WASM backend works on any browser with
    // WebAssembly support. WebGPU is preferred where present, but we
    // no longer hard-require it.
    if (this._engine) return 'available';
    if (this._initing) return 'downloading';
    return 'downloadable';
  }

  async ensureReady(onProgress) {
    if (this._engine || this._initing) return;
    // Device choice — WebGPU is fast but iOS Safari's WebGPU has been
    // crashing the tab on non-trivial models. We default to WASM on
    // iOS (slower but compatible) and let other platforms use WebGPU.
    // Models can override via `device` option; users can flip global
    // safe-mode in localStorage ('aiSafeMode' = 'true').
    const safeMode = (typeof localStorage !== 'undefined') &&
                     (localStorage.getItem('aiSafeMode') === 'true');
    const isIOSDevice = typeof navigator !== 'undefined' &&
                       /iPad|iPhone|iPod|CriOS|FxiOS|EdgiOS/.test(navigator.userAgent);
    let device = this.device ?? 'webgpu';
    if (safeMode || isIOSDevice) device = 'wasm';
    // Final fallback: if no WebGPU and we ended up requesting it, switch to wasm.
    if (device === 'webgpu' && !('gpu' in (typeof navigator !== 'undefined' ? navigator : {}))) {
      device = 'wasm';
    }
    this._effectiveDevice = device;

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
        onProgress?.({ stage: 'model', progress: pct, text, device });
      };

      if (this.mode === 'paligemma') {
        const processor = await mod.AutoProcessor.from_pretrained(this.hfRepo, { progress_callback });
        const model     = await mod.PaliGemmaForConditionalGeneration.from_pretrained(this.hfRepo, {
          device,
          dtype: this.dtype,
          progress_callback,
        });
        this._engine = { kind: 'paligemma', processor, model, mod };
      } else if (this.mode === 'smolvlm') {
        const processor = await mod.AutoProcessor.from_pretrained(this.hfRepo, { progress_callback });
        const model     = await mod.AutoModelForVision2Seq.from_pretrained(this.hfRepo, {
          device,
          dtype: this.dtype,
          progress_callback,
        });
        this._engine = { kind: 'smolvlm', processor, model, mod };
      } else if (this.mode === 'object-detection') {
        const detector = await mod.pipeline('object-detection', this.hfRepo, {
          device,
          dtype: this.dtype,
          progress_callback,
        });
        this._engine = { kind: 'object-detection', detector, mod };
      } else {
        const pipe = await mod.pipeline(this.task, this.hfRepo, {
          device,
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

    if (kind === 'smolvlm') {
      // SmolVLM / Idefics3 path. The chat-template handler injects the
      // image placeholder + special tokens correctly; passing raw text
      // to `processor()` without chat formatting will silently confuse
      // the model.
      const rawImage = await rawImageFrom(image, mod);
      const { processor, model } = this._engine;
      const messages = [{
        role: 'user',
        content: [
          { type: 'image' },
          { type: 'text', text: this.prompt || 'Describe this image in 1–3 sentences.' },
        ],
      }];
      const promptText = processor.apply_chat_template(messages, { add_generation_prompt: true });
      const inputs = await processor(promptText, [rawImage]);
      const generated = await model.generate({
        ...inputs,
        max_new_tokens: this.maxNewTokens,
      });
      // Trim the prompt tokens from the front of the generated tensor
      // so batch_decode returns only the assistant's reply. Some
      // Transformers.js versions have a tensor.slice() with this exact
      // shape; fall back to string-trim if the slice errors.
      let decoded;
      try {
        const inputLen = inputs.input_ids.dims.at(-1);
        const trimmed = generated.slice(null, [inputLen, null]);
        decoded = processor.batch_decode(trimmed, { skip_special_tokens: true });
      } catch {
        decoded = processor.batch_decode(generated, { skip_special_tokens: true });
        // Best-effort string-trim of the prompt prefix
        const promptStr = promptText.replace(/<[^>]+>/g, '').trim();
        if (decoded[0]?.includes(promptStr)) {
          decoded[0] = decoded[0].split(promptStr).pop().trim();
        }
      }
      let text = (decoded?.[0] ?? '').trim();
      // SmolVLM sometimes emits "Assistant:" prefix
      text = text.replace(/^Assistant:\s*/i, '').trim();
      return this._postProcess ? this._postProcess(text) : text;
    }

    if (kind === 'object-detection') {
      const { url, revoke } = await imageToObjectUrl(image);
      try {
        const opts = {
          threshold: this.detectionThreshold,
          percentage: true,
        };
        // Zero-shot detectors (OWL-ViT, OWLv2) take candidate labels.
        const isZeroShot = /owl/i.test(this.hfRepo);
        let detections;
        if (isZeroShot && this.candidateLabels) {
          detections = await this._engine.detector(url, this.candidateLabels, opts);
        } else {
          detections = await this._engine.detector(url, opts);
        }
        // detections: [{ score, label, box: { xmin, ymin, xmax, ymax } }]
        if (!Array.isArray(detections) || detections.length === 0) {
          return '(no objects detected — try moving the camera, or pick a different model)';
        }
        const sorted = [...detections].sort((a, b) => b.score - a.score).slice(0, this.detectionTopK);
        // De-duplicate labels, keep the best score per label.
        const seen = new Map();
        for (const d of sorted) {
          const cur = seen.get(d.label);
          if (!cur || d.score > cur.score) seen.set(d.label, d);
        }
        const summary = [...seen.values()]
          .map(d => `${d.label} (${Math.round(d.score * 100)}%)`)
          .join(' · ');
        const text = `Detected: ${summary}`;
        return this._postProcess ? this._postProcess(text) : text;
      } finally {
        revoke?.();
      }
    }

    // pipeline path. Two input shapes the image-to-text pipeline accepts:
    //   • url-prompt  — pipe(url, promptString)  (Florence-2 et al.)
    //   • url-only    — pipe(url)                (pure captioners: ViT-GPT2)
    // The chat-messages form is handled by the 'smolvlm' mode above —
    // the pipeline factory's image-to-text task throws "Unsupported
    // input type: object" on a messages array.
    const { url, revoke } = await imageToObjectUrl(image);
    try {
      let result;
      if (this.inputFormat === 'url-only') {
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

  // Structured object-detection. Returns the raw pipeline output:
  //   [{ score, label, box: { xmin, ymin, xmax, ymax } }, ...]
  // Throws for non-detection models. Used by the autonomous Drone so
  // it can reason about which labels are new and not have to parse
  // describe()'s formatted string.
  async detect(image) {
    if (this.mode !== 'object-detection') {
      throw new Error(`detect() is only supported on object-detection models (this one is ${this.mode})`);
    }
    if (!this._engine) await this.ensureReady();
    const { url, revoke } = await imageToObjectUrl(image);
    try {
      const opts = { threshold: this.detectionThreshold, percentage: true };
      const isZeroShot = /owl/i.test(this.hfRepo);
      let result;
      if (isZeroShot && this.candidateLabels) {
        result = await this._engine.detector(url, this.candidateLabels, opts);
      } else {
        result = await this._engine.detector(url, opts);
      }
      return Array.isArray(result) ? result : [];
    } finally {
      revoke?.();
    }
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
