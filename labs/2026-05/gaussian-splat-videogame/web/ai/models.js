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
  constructor({ id, label, provider, modelId, sizeHint, downloadGB, notes }) {
    this.id = id;
    this.label = label;
    this.provider = provider;
    this.runtime = 'WebGPU via WebLLM';
    this.modelId = modelId;
    this.sizeHint = sizeHint;
    this.downloadGB = downloadGB;
    this.multimodal = false; // WebLLM vision not yet GA — text-only path
    this.notes = notes;
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
          // report = { progress: 0..1, text, timeElapsed }
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
}

// ── Registry ───────────────────────────────────────────────────────────
export function buildRegistry() {
  return [
    new ChromeBuiltinModel(),
    new WebLLMModel({
      id: 'gemma-2-2b',
      label: 'Gemma 2 2B Instruct',
      provider: 'Google DeepMind (open weights)',
      modelId: 'gemma-2-2b-it-q4f16_1-MLC',
      sizeHint: '~1.5 GB',
      downloadGB: 1.5,
      notes: 'Smallest Gemma in WebLLM; quick first download, good for low-RAM devices. Text-only; will upgrade to the multimodal Gemma 3 family when WebLLM publishes them.',
    }),
    new WebLLMModel({
      id: 'llama-3.2-3b',
      label: 'Llama 3.2 3B Instruct',
      provider: 'Meta',
      modelId: 'Llama-3.2-3B-Instruct-q4f32_1-MLC',
      sizeHint: '~1.9 GB',
      downloadGB: 1.9,
      notes: 'Strong general-purpose 3B; text-only via WebLLM. The 11B-Vision variant exists upstream but isn\'t in WebLLM yet.',
    }),
    new WebLLMModel({
      id: 'qwen-2.5-3b',
      label: 'Qwen 2.5 3B Instruct',
      provider: 'Alibaba',
      modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
      sizeHint: '~1.8 GB',
      downloadGB: 1.8,
      notes: 'Excellent at descriptive prose; text-only. The VL multimodal Qwen variants aren\'t yet WebLLM-targeted.',
    }),
  ];
}
