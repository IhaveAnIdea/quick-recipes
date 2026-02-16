/**
 * Embedding worker for mobile — runs Xenova/paraphrase-MiniLM-L3-v2.
 *
 * Uses @xenova/transformers v2.15.1 instead of v3 because v3's JSEP/ASYNCIFY
 * WASM build causes a memory balloon (10+ GB) that crashes iOS Safari.
 * See: https://github.com/huggingface/transformers.js/issues/1242
 *
 * v2 uses the non-JSEP WASM build and is confirmed working on all iOS versions.
 */
const MODEL_ID = "Xenova/paraphrase-MiniLM-L3-v2";
const DIM = 384;

let extractor = null;
let loadPromise = null;

async function ensureExtractor() {
  if (extractor) return;
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    const T = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1");
    const env = T.env || T.default?.env;
    if (env) {
      env.allowLocalModels = false;
      env.useBrowserCache = true;
    }
    const pipe = T.pipeline || T.default?.pipeline;
    extractor = await pipe("feature-extraction", MODEL_ID, {
      pooling: "mean",
      normalize: true,
    });
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
