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
import { MediaPipeDetector } from './mediapipe.js';

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
    // ─── MediaPipe object detection ────────────────────────────────────
    // Different runtime from Transformers.js — TFLite via MediaPipe
    // Tasks. iOS Safari has been crashing every ONNX-Runtime-Web model
    // we've tried, including tiny encoder-only ones; MediaPipe ships a
    // separate WASM purpose-built for mobile browsers. iOS DEFAULT.
    new MediaPipeDetector({
      id: 'mp-efficientdet-lite0',
      label: 'EfficientDet Lite 0 (MediaPipe, iOS default)',
      modelUrl: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
      sizeHint: '~5 MB (TFLite int8)',
      downloadGB: 0.005,
      delegate: 'CPU',
      threshold: 0.3,
      topK: 10,
      notes: 'Google MediaPipe object detector — TFLite runtime, different code path from ONNX. Designed for mobile browsers, much better iOS-tested than Transformers.js. ~5 MB model + ~11 MB MediaPipe WASM (cached after first load).',
    }),
    new MediaPipeDetector({
      id: 'mp-efficientdet-lite2',
      label: 'EfficientDet Lite 2 (MediaPipe, better)',
      modelUrl: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite',
      sizeHint: '~8 MB (TFLite int8)',
      downloadGB: 0.008,
      delegate: 'CPU',
      threshold: 0.3,
      topK: 10,
      notes: 'Larger EfficientDet — more accurate detection at slightly higher compute.',
    }),
    new MediaPipeDetector({
      id: 'mp-ssd-mobilenet',
      label: 'SSD MobileNet V2 (MediaPipe)',
      modelUrl: 'https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/latest/ssd_mobilenet_v2.tflite',
      sizeHint: '~6 MB (TFLite fp16)',
      downloadGB: 0.006,
      delegate: 'CPU',
      threshold: 0.3,
      topK: 10,
      notes: 'Classic MobileNet V2 + SSD detection head. Slightly different label distribution than EfficientDet.',
    }),
    // ─── ONNX-based object detection tier ───────────────────────────────
    // Encoder-only models, no autoregressive decoder, tiny weights. The
    // most reliable browser-inference path on iOS WebGPU. Output is a
    // structured list of (label, score, bbox) rather than prose, but
    // for "what's in the scene?" that's often more useful than a fuzzy
    // sentence.
    new TransformersJSModel({
      id: 'yolos-tiny',
      label: 'YOLOS Tiny — object detect (iOS default)',
      provider: 'Xenova (Apache-2.0)',
      hfRepo: 'Xenova/yolos-tiny',
      mode: 'object-detection',
      task: 'object-detection',
      dtype: 'q8',
      sizeHint: '~9 MB (int8)',
      downloadGB: 0.009,
      iosSafe: true,
      detectionThreshold: 0.3,
      detectionTopK: 12,
      notes: 'Tiny YOLOS, ~9 MB. Detects ~80 COCO classes. Output is "Detected: chair (87%), table (76%), ...". The simplest, smallest, most-stable model in the menu — switch to this if captioners crash.',
    }),
    new TransformersJSModel({
      id: 'detr-resnet50',
      label: 'DETR ResNet-50 — object detect (accurate)',
      provider: 'Xenova (Apache-2.0)',
      hfRepo: 'Xenova/detr-resnet-50',
      mode: 'object-detection',
      task: 'object-detection',
      dtype: 'q8',
      sizeHint: '~41 MB (int8)',
      downloadGB: 0.041,
      iosSafe: true,
      detectionThreshold: 0.4,
      detectionTopK: 12,
      notes: 'Meta\'s DETR with ResNet-50 backbone. ~80 COCO classes. More accurate than YOLOS-tiny, still tiny by today\'s standards.',
    }),
    new TransformersJSModel({
      id: 'owlvit-base',
      label: 'OWL-ViT — zero-shot object detect',
      provider: 'Xenova / Google (Apache-2.0)',
      hfRepo: 'Xenova/owlvit-base-patch32',
      mode: 'object-detection',
      task: 'zero-shot-object-detection',
      dtype: 'q8',
      sizeHint: '~148 MB (int8)',
      downloadGB: 0.148,
      iosSafe: true,
      detectionThreshold: 0.1,
      detectionTopK: 12,
      candidateLabels: ['chair','table','person','book','lamp','cup','plate','food','sculpture','painting','plant','window','door','vehicle','wall','floor','ceiling','tomato','apple','flower','toy','tool'],
      notes: 'Open-vocabulary detector — looks for the candidate labels you supply (defaults to a general indoor set). Slower than YOLOS but finds things outside the COCO classes.',
    }),
    // Multimodal via Transformers.js — true image input.
    //
    // SmolVLM (HuggingFaceTB) is the spiritual answer to "smaller
    // PaliGemma": HF's purpose-built small VLM line for browser
    // inference. q4f16 totals are tiny (189–341 MB), the architecture
    // is exported as Idefics3/SmolVLM in Transformers.js latest, and
    // the repos are ungated.
    new TransformersJSModel({
      id: 'smolvlm-256m',
      label: 'SmolVLM 256M (multimodal, fragile on iOS)',
      provider: 'HuggingFaceTB (Apache-2.0)',
      hfRepo: 'HuggingFaceTB/SmolVLM-256M-Instruct',
      mode: 'smolvlm',
      task: 'image-to-text',
      prompt: 'Describe this image in 1–2 sentences.',
      dtype: 'q4f16',
      sizeHint: '~190 MB (q4f16)',
      downloadGB: 0.19,
      iosSafe: false,
      mobileWarning: 'Observed crashing iOS Safari mid-load — likely WebGPU shader-compile on Idefics3 attention ops. On iOS use DistilViT or ViT-GPT2.',
      maxNewTokens: 120,
      notes: 'HF\'s purpose-built tiny VLM. Architecture is more complex than the ViT encoder-decoder captioners; on iOS Safari\'s WebGPU it has been seen to crash before inference. Works fine on Chrome/Edge desktop.',
    }),
    new TransformersJSModel({
      id: 'smolvlm-500m',
      label: 'SmolVLM 500M (multimodal, fragile on iOS)',
      provider: 'HuggingFaceTB (Apache-2.0)',
      hfRepo: 'HuggingFaceTB/SmolVLM-500M-Instruct',
      mode: 'smolvlm',
      task: 'image-to-text',
      prompt: 'Describe this image in 1–3 sentences.',
      dtype: 'q4f16',
      sizeHint: '~340 MB (q4f16)',
      downloadGB: 0.34,
      iosSafe: false,
      mobileWarning: 'Same iOS WebGPU fragility as the 256M variant. Desktop only.',
      maxNewTokens: 160,
      notes: 'Larger SmolVLM — richer captions on desktop, doesn\'t survive iOS Safari\'s WebGPU.',
    }),
    // Even-lighter tier: pure image captioners (encoder-decoder, no chat).
    // No prompting — just "describe what's in this picture". These are
    // tiny, fast, and the most reliably stable on iOS WebGPU (their
    // ops are basic ViT + GPT2 vs the more exotic attention patterns
    // in SmolVLM / PaliGemma).
    new TransformersJSModel({
      id: 'vit-gpt2',
      label: 'ViT-GPT2 (classic captioner, iOS-stable)',
      provider: 'Xenova (NLP Connect base)',
      hfRepo: 'Xenova/vit-gpt2-image-captioning',
      mode: 'pipeline',
      task: 'image-to-text',
      inputFormat: 'url-only',
      dtype: 'q8',
      sizeHint: '~246 MB (int8)',
      downloadGB: 0.25,
      iosSafe: true,
      maxNewTokens: 40,
      notes: 'Classic ViT encoder + GPT2 decoder, MIT licensed. Pure image captioning (no conversation). In Transformers.js since v1; the most thoroughly browser-tested option here. Single-sentence captions.',
    }),
    new TransformersJSModel({
      id: 'distilvit',
      label: 'DistilViT (iOS default — most stable)',
      provider: 'Mozilla',
      hfRepo: 'Mozilla/distilvit',
      mode: 'pipeline',
      task: 'image-to-text',
      inputFormat: 'url-only',
      dtype: 'q8',
      sizeHint: '~190 MB (int8)',
      downloadGB: 0.19,
      iosSafe: true,
      maxNewTokens: 40,
      notes: 'Mozilla\'s distilled image captioner, built explicitly for browser inference (used in Firefox alt-text). Smallest viable model in the menu and the architecture is purpose-trimmed for stability. Single-sentence captions; this is the iOS default after SmolVLM was found to crash iOS WebGPU.',
    }),
    // PaliGemma 2 — the model you asked for. Crashes iOS at ~2.7 GiB;
    // kept for desktop users. Lower-level API path.
    new TransformersJSModel({
      id: 'paligemma2-3b',
      label: 'PaliGemma 2 3B (multimodal, desktop)',
      provider: 'Google · onnx-community',
      hfRepo: 'onnx-community/paligemma2-3b-pt-224',
      mode: 'paligemma',
      task: 'image-to-text',
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
      task: 'image-to-text',
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
      task: 'image-to-text',
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
//   • iOS / WebKit: MediaPipe EfficientDet Lite 0 — different runtime
//     from Transformers.js / ONNX Runtime Web (which has been crashing
//     iOS Safari on every model we've tried, even at 9 MB).
//   • Chrome/Edge with Nano available: Nano (no download).
//   • Other desktop: PaliGemma 2.
export function pickDefaultModelId(models) {
  if (isIOS()) {
    return models.find(m => m.id === 'mp-efficientdet-lite0')?.id
        ?? models.find(m => m.id === 'mp-ssd-mobilenet')?.id
        ?? models.find(m => m.id === 'yolos-tiny')?.id
        ?? models[1].id;
  }
  return models.find(m => m.id === 'paligemma2-3b')?.id
      ?? models.find(m => m.id === 'mp-efficientdet-lite0')?.id
      ?? models[0].id;
}
