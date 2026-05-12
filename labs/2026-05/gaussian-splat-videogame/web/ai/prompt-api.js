// Chrome built-in AI (Gemini Nano) via the Prompt API.
//
// Detection is duck-typed across the API renames that have happened over
// the past year:
//   • `window.LanguageModel` — current (2026) spec entry-point.
//   • `window.ai.languageModel` — earlier prefix (Origin Trial era).
//   • Neither present → 'unavailable' (no Chrome built-in AI here).
//
// availability() returns one of: 'available', 'downloadable',
// 'downloading', 'unavailable' (modelled on the W3C Web AI proposal).

export function getLanguageModelAPI() {
  if (typeof window === 'undefined') return null;
  if ('LanguageModel' in window) return window.LanguageModel;
  if (window.ai?.languageModel) return window.ai.languageModel;
  return null;
}

export async function checkAvailability({ expectInputs = ['text'] } = {}) {
  const lm = getLanguageModelAPI();
  if (!lm) return 'unavailable';
  try {
    const expectedInputs = expectInputs.map(t => ({ type: t }));
    if (typeof lm.availability === 'function') {
      return await lm.availability({ expectedInputs });
    }
    // Older shape: ai.canCreateGenericSession()
    if (typeof lm.capabilities === 'function') {
      const c = await lm.capabilities();
      return c.available ?? 'unavailable';
    }
  } catch (e) {
    console.warn('[prompt-api] availability check failed', e);
  }
  return 'unavailable';
}

export async function createSession({
  systemPrompt,
  expectInputs = ['text', 'image'],
  temperature,
  topK,
  onDownloadProgress,
} = {}) {
  const lm = getLanguageModelAPI();
  if (!lm) throw new Error('Prompt API not available in this browser');

  const opts = {
    expectedInputs: expectInputs.map(t => ({ type: t })),
  };
  if (systemPrompt)     opts.systemPrompt = systemPrompt;
  if (temperature != null) opts.temperature = temperature;
  if (topK != null)        opts.topK = topK;
  if (onDownloadProgress) {
    opts.monitor = (m) => {
      m.addEventListener?.('downloadprogress', (e) => onDownloadProgress(e.loaded));
    };
  }

  // Newer signature: LanguageModel.create(opts). Older: ai.languageModel.create(opts).
  if (typeof lm.create === 'function') return await lm.create(opts);
  throw new Error('Prompt API present but missing create()');
}

// Send a multimodal prompt to a session. `image` may be an
// HTMLCanvasElement, HTMLImageElement, ImageBitmap, or Blob.
export async function promptMultimodal(session, { text, image }) {
  if (!session) throw new Error('No session');
  // The 2026 API takes an array of message parts.
  const content = [];
  if (text) content.push({ type: 'text', value: text });
  if (image) content.push({ type: 'image', value: image });
  // Some intermediate Chrome versions accepted just a string for text-only
  // calls; promptStreaming/prompt both exist.
  if (typeof session.prompt === 'function') {
    try {
      return await session.prompt(content.length > 1 || image ? content : (text ?? ''));
    } catch (e) {
      // Fallback to string form on older signatures
      if (!image) return await session.prompt(text ?? '');
      throw e;
    }
  }
  throw new Error('Session has no prompt() method');
}
