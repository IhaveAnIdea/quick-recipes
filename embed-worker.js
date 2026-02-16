/**
 * Embedding worker for mobile — runs Xenova/paraphrase-MiniLM-L3-v2 in isolation
 * to avoid main-thread memory pressure and iOS Safari jetsam.
 * Uses q4 quantization (~5 MB) for minimal footprint.
 */
const MODEL_ID = "Xenova/paraphrase-MiniLM-L3-v2";
const DIM = 384;

let extractor = null;
let loading = false;
let loadPromise = null;

async function ensureExtractor() {
  if (extractor) return;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    loading = true;
    const T = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js");
    const env = T.env || T.default?.env;
    if (env) {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      if (env.backends?.onnx?.wasm) env.backends.onnx.wasm.numThreads = 1;
    }
    const pipe = T.pipeline || T.default?.pipeline;
    try {
      extractor = await pipe("feature-extraction", MODEL_ID, {
        pooling: "mean",
        normalize: true,
        device: "wasm",
        dtype: "q4",
      });
    } catch (e) {
      try {
        extractor = await pipe("feature-extraction", MODEL_ID, {
          pooling: "mean",
          normalize: true,
          device: "wasm",
          dtype: "q8",
        });
      } catch (e2) {
        throw e;
      }
    }
    loading = false;
  })();
  return loadPromise;
}

function extractVec(out, dim) {
  let vec;
  if (out?.data) vec = new Float32Array(out.data);
  else if (out?.tolist) vec = new Float32Array(out.tolist().flat(Infinity));
  else if (Array.isArray(out)) vec = new Float32Array(out.flat(Infinity));
  else throw new Error("Unexpected embedding output.");
  if (vec.length > dim * 1.5) {
    const tokens = Math.round(vec.length / dim);
    const pooled = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      for (let t = 0; t < tokens; t++) sum += vec[t * dim + i];
      pooled[i] = sum / tokens;
    }
    let n = 0;
    for (let i = 0; i < dim; i++) n += pooled[i] * pooled[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < dim; i++) pooled[i] /= n;
    vec = pooled;
  }
  return vec;
}

self.onmessage = async (e) => {
  const { type, id, text } = e.data || {};
  if (type !== "embed" || !text) return;
  try {
    await ensureExtractor();
    const out = await extractor(text, { pooling: "mean", normalize: true });
    const vec = extractVec(out, DIM);
    self.postMessage({ type: "embed", id, vec }, [vec.buffer]);
  } catch (err) {
    self.postMessage({ type: "embed", id, error: String(err?.message || err) });
  }
};
