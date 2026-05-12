// AI model registry.
//
// Each model exposes a uniform shape:
//   { id, label, provider, runtime, sizeHint, multimodal, downloadGB,
//     availability(): Promise<'available'|'downloadable'|'downloading'|'unavailable'>,
//     ensureReady(progressCb): Promise<void>     // sets up session/engine
//     describe(canvas, hint, scene): Promise<string>
//     teardown(): void }
//
// Two backends wired:
//   • Chrome Prompt API (Gemini Nano) — true multimodal, on-device.
//   • WebLLM (https://webllm.mlc.ai/) — three open-weights models loaded
//     lazily from the MLC CDN on first use. WebLLM caches the weights
//     in Cache Storage, so subsequent loads skip the multi-GB download.
//     Vision-capable WebLLM models are not yet GA; for now the WebLLM
//     path is *text-only* (it sees scene metadata, not pixels). The
//     menu surfaces this honestly.

import { getLanguageModelAPI, checkAvailability, createSession, promptMultimodal } from './prompt-api.js';
import { TransformersJSModel } from './transformers-js.js';

const DESCRIBE_SYSTEM =
  'You are a concise art and 3D-scene critic. Given a Gaussian-splat scene, write a vivid 1–3 sentence description of subject, mood, and palette. Do not preface with phrases like "the image shows" or "based on the metadata".';

// WebLLM CDN entry. esm.run resolves to a versioned ESM build; first
// access pays for the engine code (~500 KB) plus the chosen model's
// weights (~1.5–3 GB). Subsequent loads are cached.
const WEBLLM_CDN = 'https://esm.run/@mlc-ai/web-llm';
let _webllmModule = null;
async function loadWebLLM(onProgress) {
  if (_webllmModule) return _webllmModule;
  onProgress?.({ stage: 'engine', text: 'Loading WebLLM engine…' });
  _webllmModule = await import(/* @vite-ignore */ WEBLLM_CDN);
  return _webllmModule;
}

// ── Chrome Built-in (Gemini Nano) ──────────────────────────────────────
class ChromeBuiltinModel {
  constructor() {
    this.id = 'chrome-builtin';
    this.label = 'Gemini Nano (Chrome built-in)';
    this.provider = 'Google / Chrome';
    this.runtime = 'on-device, Prompt API';
    this.sizeHint = 'already on device';
    this.downloadGB = 0;
    this.multimodal = true;
    this._session = null;
  }
  async availability() {
    if (!getLanguageModelAPI()) return 'unavailable';
    return checkAvailability({ expectInputs: ['text', 'image'] });
  }
  async ensureReady(onProgress) {
    if (this._session) return;
    this._session = await createSession({
      systemPrompt: DESCRIBE_SYSTEM,
      expectInputs: ['text', 'image'],
      temperature: 0.4,
      onDownloadProgress: (loaded) => onProgress?.({ stage: 'model', loaded, text: 'Downloading Gemini Nano…' }),
    });
  }
  async describe(canvas, hint /*, scene */) {
    if (!this._session) await this.ensureReady();
    let image = canvas;
    if (typeof createImageBitmap === 'function') {
      try { image = await createImageBitmap(canvas); } catch {}
    }
    const text = (hint ? `${hint}\n\n` : '') +
      'Describe this rendered Gaussian-splat scene in 1–3 sentences.';
    return await promptMultimodal(this._session, { text, image });
  }
  teardown() {
    try { this._session?.destroy?.(); } catch {}
    this._session = null;
  }
}

// ── WebLLM-backed text-only models ─────────────────────────────────────
class WebLLMModel {
  constructor({ id, label, provider, modelId, sizeHint, downloadGB, notes, iosSafe, mobileWarning }) {
    this.id = id;
    this.label = label;
    this.provider = provider;
    this.runtime = 'WebGPU via WebLLM';
    this.modelId = modelId;
    this.sizeHint = sizeHint;
    this.downloadGB = downloadGB;
    this.multimodal = false; // WebLLM vision not yet GA — text-only path
    this.notes = notes;
    this.iosSafe = !!iosSafe;
    this.mobileWarning = mobileWarning;
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
      throw new Error('WebGPU not available in this browser. iOS Safari 18+, Chrome, Edge, and Arc support it; Firefox is behind a flag.');
    }
    this._initing = true;
    try {
      const { CreateMLCEngine } = await loadWebLLM(onProgress);
      this._engine = await CreateMLCEngine(this.modelId, {
        initProgressCallback: (report) => {
          onProgress?.({
            stage: 'model',
            progress: report.progress,
            text: report.text,
            timeElapsed: report.timeElapsed,
          });
        },
      });
    } finally {
      this._initing = false;
    }
  }
  async describe(_canvas, hint, scene) {
    if (!this._engine) await this.ensureReady();
    const meta = scene
      ? `Scene title: "${scene.title}". Author: ${scene.author}. Category: ${scene.category}${scene.subcategory ? ' / ' + scene.subcategory : ''}. License: ${scene.license}.`
      : (hint || '');
    const userText =
      `${meta}\n\nNote: you are working from metadata only — you cannot see the image. ` +
      `Describe this Gaussian-splat scene in 1–3 sentences (subject, mood, palette) using what the metadata implies.`;
    const completion = await this._engine.chat.completions.create({
      messages: [
        { role: 'system', content: DESCRIBE_SYSTEM },
        { role: 'user',   content: userText },
      ],
      temperature: 0.5,
      max_tokens: 180,
    });
    return completion.choices?.[0]?.message?.content ?? '(no response)';
  }
  teardown() {
    try { this._engine?.unload?.(); } catch {}
    this._engine = null;
  }
  // Best-effort cache clear via Cache Storage (WebLLM stores weights in
  // named caches keyed off the model id).
  async clearCache() {
    if (typeof caches === 'undefined') return false;
    const keys = await caches.keys();
    const mine = keys.filter(k => k.includes(this.modelId) || k.includes('webllm'));
    await Promise.all(mine.map(k => caches.delete(k)));
    this.teardown();
    return mine.length > 0;
  }
}

// Rough iOS detection — used to default to the smallest Gemma on
// iPhone / iPad where Safari's per-tab cap (~1.5 GB) makes the larger
// Gemma 2 2B variant crash mid-load on many devices.
export function isIOS() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /iPad|iPhone|iPod/.test(ua) || /CriOS|FxiOS|EdgiOS/.test(ua);
}

// ── Registry ───────────────────────────────────────────────────────────
export function buildRegistry() {
  return [
    new ChromeBuiltinModel(),
    // Multimodal via Transformers.js — true image input.
    //
    // SmolVLM (HuggingFaceTB) is the spiritual answer to "smaller
    // PaliGemma": HF's purpose-built small VLM line for browser
    // inference. q4f16 totals are tiny (189–341 MB), the architecture
    // is exported as Idefics3/SmolVLM in Transformers.js latest, and
    // the repos are ungated.
    new TransformersJSModel({
      id: 'smolvlm-256m',
      label: 'SmolVLM 256M (multimodal, tiny)',
      provider: 'HuggingFaceTB (Apache-2.0)',
      hfRepo: 'HuggingFaceTB/SmolVLM-256M-Instruct',
      mode: 'pipeline',
      task: 'image-text-to-text',
      prompt: 'Describe this image in 1–2 sentences.',
      dtype: 'q4f16',
      sizeHint: '~190 MB (q4f16)',
      downloadGB: 0.19,
      iosSafe: true,
      maxNewTokens: 120,
      notes: 'HF\'s purpose-built tiny VLM. Sees the rendered image. Comfortably fits iOS Safari\'s tab cap with headroom to spare. Captions are short but on-topic.',
    }),
    new TransformersJSModel({
      id: 'smolvlm-500m',
      label: 'SmolVLM 500M (multimodal, iOS default)',
      provider: 'HuggingFaceTB (Apache-2.0)',
      hfRepo: 'HuggingFaceTB/SmolVLM-500M-Instruct',
      mode: 'pipeline',
      task: 'image-text-to-text',
      prompt: 'Describe this image in 1–3 sentences.',
      dtype: 'q4f16',
      sizeHint: '~340 MB (q4f16)',
      downloadGB: 0.34,
      iosSafe: true,
      maxNewTokens: 160,
      notes: 'Larger SmolVLM — richer captions, still iOS-safe. Default on iOS.',
    }),
    // PaliGemma 2 — the model you asked for. Crashes iOS at ~2.7 GiB;
    // kept for desktop users. Lower-level API path.
    new TransformersJSModel({
      id: 'paligemma2-3b',
      label: 'PaliGemma 2 3B (multimodal, desktop)',
      provider: 'Google · onnx-community',
      hfRepo: 'onnx-community/paligemma2-3b-pt-224',
      mode: 'paligemma',
      task: 'image-text-to-text',
      prompt: 'caption en',
      dtype: 'q4f16',
      sizeHint: '~2.7 GiB (q4f16)',
      downloadGB: 2.7,
      iosSafe: false,
      mobileWarning: 'Confirmed to crash iOS Safari mid-download (weight set is past the ~1.5 GB per-tab cap). Desktop only.',
      maxNewTokens: 80,
      notes: 'True multimodal — sees the image. Pre-trained PaliGemma 2 (the only ungated PaliGemma ONNX on HF). Use this on desktop; on iOS pick SmolVLM 500M.',
    }),
    // Florence-2 — Microsoft, MIT, alternative iOS-safe multimodal.
    new TransformersJSModel({
      id: 'florence2-base',
      label: 'Florence-2 Base (multimodal, alt)',
      provider: 'Microsoft (MIT) · onnx-community',
      hfRepo: 'onnx-community/Florence-2-base-ft',
      mode: 'pipeline',
      task: 'image-text-to-text',
      prompt: '<MORE_DETAILED_CAPTION>',
      dtype: 'q4',
      sizeHint: '~270 MB (q4)',
      downloadGB: 0.27,
      iosSafe: true,
      notes: 'MIT-licensed alternative VLM, ungated, iOS-safe. More literal than SmolVLM but tends to be more accurate on object names.',
      postProcess: (t) => t.replace(/<[A-Z_]+>/g, '').trim(),
    }),
    new TransformersJSModel({
      id: 'florence2-large',
      label: 'Florence-2 Large (multimodal)',
      provider: 'Microsoft (MIT) · onnx-community',
      hfRepo: 'onnx-community/Florence-2-large-ft',
      mode: 'pipeline',
      task: 'image-text-to-text',
      prompt: '<MORE_DETAILED_CAPTION>',
      dtype: 'q4',
      sizeHint: '~770 MB (q4)',
      downloadGB: 0.77,
      iosSafe: true,
      notes: 'Larger Florence-2 — richer descriptions. Still iOS-viable.',
      postProcess: (t) => t.replace(/<[A-Z_]+>/g, '').trim(),
    }),
    // Smallest first — iOS-safe by default. The big variants are at the
    // bottom and flagged with mobileWarning.
    new WebLLMModel({
      id: 'smollm2-135m',
      label: 'SmolLM2 135M Instruct',
      provider: 'HuggingFace (open weights)',
      modelId: 'SmolLM2-135M-Instruct-q0f16-MLC',
      sizeHint: '~270 MB',
      downloadGB: 0.27,
      iosSafe: true,
      notes: 'Tiny safety-net model. Fast first download, fits anywhere, prose is basic. Text-only.',
    }),
    new WebLLMModel({
      id: 'llama-3.2-1b',
      label: 'Llama 3.2 1B Instruct',
      provider: 'Meta',
      modelId: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
      sizeHint: '~750 MB',
      downloadGB: 0.75,
      iosSafe: true,
      notes: 'Modern small model; safe text-only fallback on iOS / older devices.',
    }),
    new WebLLMModel({
      id: 'gemma-2b',
      label: 'Gemma 2b Instruct (text)',
      provider: 'Google DeepMind (open weights)',
      modelId: 'gemma-2b-it-q4f16_1-MLC',
      sizeHint: '~1.3 GB',
      downloadGB: 1.3,
      iosSafe: true,
      notes: 'Smallest *Gemma* in WebLLM (Gemma 1 2B). Text-only. Gemma 3/4 aren\'t yet packaged for WebLLM.',
    }),
    new WebLLMModel({
      id: 'gemma-2-2b',
      label: 'Gemma 2 2B Instruct',
      provider: 'Google DeepMind (open weights)',
      modelId: 'gemma-2-2b-it-q4f16_1-MLC',
      sizeHint: '~1.5 GB',
      downloadGB: 1.5,
      iosSafe: false,
      mobileWarning: 'Crashes iOS Safari mid-load on most iPhones — desktop only.',
      notes: 'Latest Gemma in WebLLM. Text-only. Better prose than Gemma 1, at the cost of memory headroom.',
    }),
    new WebLLMModel({
      id: 'llama-3.2-3b',
      label: 'Llama 3.2 3B Instruct',
      provider: 'Meta',
      modelId: 'Llama-3.2-3B-Instruct-q4f32_1-MLC',
      sizeHint: '~1.9 GB',
      downloadGB: 1.9,
      iosSafe: false,
      mobileWarning: 'Too large for iOS Safari per-tab cap. Desktop only.',
      notes: 'Strong general-purpose 3B; text-only via WebLLM.',
    }),
    new WebLLMModel({
      id: 'qwen-2.5-3b',
      label: 'Qwen 2.5 3B Instruct',
      provider: 'Alibaba',
      modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
      sizeHint: '~1.8 GB',
      downloadGB: 1.8,
      iosSafe: false,
      mobileWarning: 'Too large for iOS Safari per-tab cap. Desktop only.',
      notes: 'Excellent at descriptive prose; text-only.',
    }),
  ];
}

// Pick the best default model for the current platform.
//   • iOS / WebKit: SmolVLM 500M — small, ungated, real multimodal,
//     fits comfortably (~340 MB) where PaliGemma can't (~2.7 GB).
//   • Chrome/Edge with Nano available: Nano (no download).
//   • Other desktop: PaliGemma 2.
export function pickDefaultModelId(models) {
  if (isIOS()) {
    return models.find(m => m.id === 'smolvlm-500m')?.id
        ?? models.find(m => m.id === 'florence2-base')?.id
        ?? models[1].id;
  }
  return models.find(m => m.id === 'paligemma2-3b')?.id
      ?? models.find(m => m.id === 'smolvlm-500m')?.id
      ?? models[0].id;
}
