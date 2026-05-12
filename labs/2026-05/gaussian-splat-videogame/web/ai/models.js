// AI model registry.
//
// Each model exposes a uniform shape:
//   { id, label, provider, runtime, sizeHint, multimodal,
//     availability(): Promise<'available'|'downloadable'|'downloading'|'unavailable'>,
//     ensureReady(progressCb): Promise<void>     // sets up session/engine
//     describe(canvas, scenePrompt): Promise<string>
//     teardown(): void }
//
// Only the Chrome Prompt API (Nano) is fully wired. The WebLLM-backed
// models are stubbed: they correctly report 'unavailable' / 'downloadable'
// and surface a clear "Setup" affordance, but the actual download +
// inference loop is deferred behind a flag — multi-GB downloads
// shouldn't fire unless the user explicitly asks for them, and WebLLM's
// vision-model surface is still moving fast.

import { getLanguageModelAPI, checkAvailability, createSession, promptMultimodal } from './prompt-api.js';

const DESCRIBE_SYSTEM =
  'You are a concise art and 3D-scene critic. When shown a frame, give a vivid 1–3 sentence description of what is rendered (subject, mood, palette). Do not preface with phrases like "the image shows".';

// ── Chrome Built-in (Gemini Nano) ──────────────────────────────────────
class ChromeBuiltinModel {
  constructor() {
    this.id = 'chrome-builtin';
    this.label = 'Gemini Nano (Chrome built-in)';
    this.provider = 'Google / Chrome';
    this.runtime = 'on-device, Prompt API';
    this.sizeHint = 'already on device';
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
      onDownloadProgress: onProgress,
    });
  }
  async describe(canvas, scenePrompt) {
    if (!this._session) await this.ensureReady();
    // Use an ImageBitmap snapshot for portability; canvas works in
    // current Chrome too, but ImageBitmap decouples from the live GL
    // surface and is safer if the buffer is cleared between frames.
    let image = canvas;
    if (typeof createImageBitmap === 'function') {
      try { image = await createImageBitmap(canvas); } catch {}
    }
    const text = scenePrompt
      ? `${scenePrompt}\n\nDescribe this rendered Gaussian-splat scene in 1–3 sentences.`
      : 'Describe this rendered Gaussian-splat scene in 1–3 sentences.';
    return await promptMultimodal(this._session, { text, image });
  }
  teardown() {
    try { this._session?.destroy?.(); } catch {}
    this._session = null;
  }
}

// ── WebLLM-backed stubs ────────────────────────────────────────────────
// Real WebLLM integration (https://webllm.mlc.ai/) is ~1.5 MB of engine
// glue plus a multi-GB model download per pick. We register the models so
// the menu can offer them, but `ensureReady` is gated behind an explicit
// user opt-in and currently throws to surface a clear "not wired yet"
// message rather than silently failing or kicking off a huge download.

class WebLLMStub {
  constructor({ id, label, provider, repo, sizeHint, multimodal, notes }) {
    this.id = id;
    this.label = label;
    this.provider = provider;
    this.runtime = 'WebGPU via WebLLM (planned)';
    this.repo = repo;
    this.sizeHint = sizeHint;
    this.multimodal = !!multimodal;
    this.notes = notes;
  }
  async availability() {
    // WebGPU presence is necessary; WebLLM itself isn't loaded until
    // ensureReady() is called.
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return 'unavailable';
    return 'downloadable';
  }
  async ensureReady() {
    // Intentional: don't auto-download multi-GB. The menu's "Setup"
    // button surfaces this so the user knows what's expected.
    throw new Error(
      `${this.label} integration is not yet wired up. ` +
      `When implemented it will download ${this.sizeHint} from the WebLLM CDN ` +
      `and run on-device via WebGPU. See ai/models.js for the planned path.`
    );
  }
  async describe() { await this.ensureReady(); }
  teardown() {}
}

// ── Registry ───────────────────────────────────────────────────────────
export function buildRegistry() {
  return [
    new ChromeBuiltinModel(),
    new WebLLMStub({
      id: 'gemma-3-4b-it',
      label: 'Gemma 3 4B Instruct',
      provider: 'Google DeepMind',
      repo: 'mlc-ai/gemma-3-4b-it-q4f32_1-MLC',
      sizeHint: '~2.5 GB',
      multimodal: true,
      notes: 'Open-weights Gemma 3 multimodal variant. Image input via SigLIP encoder.',
    }),
    new WebLLMStub({
      id: 'llama-3.2-vision-11b',
      label: 'Llama 3.2 11B Vision Instruct',
      provider: 'Meta',
      repo: 'mlc-ai/Llama-3.2-11B-Vision-Instruct-q4f16_1-MLC',
      sizeHint: '~6.5 GB',
      multimodal: true,
      notes: 'Strong general-purpose multimodal; large download.',
    }),
    new WebLLMStub({
      id: 'qwen2.5-vl-7b',
      label: 'Qwen 2.5 VL 7B',
      provider: 'Alibaba',
      repo: 'mlc-ai/Qwen2.5-VL-7B-Instruct-q4f16_1-MLC',
      sizeHint: '~4.5 GB',
      multimodal: true,
      notes: 'Excellent at fine-grained scene description.',
    }),
  ];
}
