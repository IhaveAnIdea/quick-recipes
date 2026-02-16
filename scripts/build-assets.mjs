// scripts/build-assets.mjs
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

import { pipeline, env } from "@huggingface/transformers";

// -------------------- CONFIG --------------------
const OUT_DIR = "assets";
const BUILD_ID = new Date().toISOString().slice(0, 10);

// Embedding model
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;

// Datasets (redistributable with attribution)
const OPEN_RECIPE_URL =
  "https://raw.githubusercontent.com/jakevdp/open-recipe-data/main/recipeitems.json.gz";

// IMPORTANT: correct file for gossminn/wikibooks-cookbook
// Dataset README lists: recipes_parsed.mini.json  :contentReference[oaicite:2]{index=2}
const WIKIBOOKS_JSON_URL =
  "https://huggingface.co/datasets/gossminn/wikibooks-cookbook/resolve/main/recipes_parsed.mini.json";

// Size controls (tune)
const MAX_OPENRECIPES = 15000;
const MAX_WIKIBOOKS = 4000;

// Batch embedding (keep modest to avoid RAM spikes)
const BATCH = 48;

// -------------------- HELPERS --------------------
async function fetchBytes(url) {
  const r = await fetch(url, { redirect: "follow" });
  if (!r.ok) throw new Error(`Fetch failed: ${url} (${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

function gunzip(u8) {
  return zlib.gunzipSync(Buffer.from(u8));
}

function normalizeText(s) {
  return (s || "")
    .replace(/\u00a0/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function toLines(s) {
  return (s || "")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function normalizeIngredientLine(line) {
  return (line || "")
    .replace(/^\s*[-*•]+\s*/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildIngredientsLines(ingredientsText) {
  const lines = toLines(ingredientsText)
    .map(normalizeIngredientLine)
    .filter(Boolean);
  return lines.filter((l) => l.length > 2).slice(0, 300);
}

function dedupeKey(r) {
  const t = (r.title || "").toLowerCase();
  const firstIng = (buildIngredientsLines(r.ingredients || "")[0] || "").toLowerCase();
  return `${t}|${firstIng}`.slice(0, 280);
}

function safeJsonParseMaybeNDJSON(text, label) {
  // Try plain JSON first
  try {
    return JSON.parse(text);
  } catch {
    // Fallback: some corpora are NDJSON / concatenated objects
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const arr = [];
    for (const ln of lines) {
      try {
        arr.push(JSON.parse(ln));
      } catch {
        // ignore non-json lines
      }
    }
    if (arr.length) return arr;
    throw new Error(`Could not parse ${label} as JSON or NDJSON.`);
  }
}

// -------------------- TAGGING --------------------
const DIET_TAGS = [
  { tag: "vegan", re: /\bvegan\b/i },
  { tag: "vegetarian", re: /\bvegetarian\b/i },
  { tag: "gluten-free", re: /\bgluten[- ]free\b|\bglutenfree\b/i },
  { tag: "dairy-free", re: /\bdairy[- ]free\b/i },
  { tag: "nut-free", re: /\bnut[- ]free\b/i },
  { tag: "keto", re: /\bketo\b/i },
  { tag: "paleo", re: /\bpaleo\b/i },
  { tag: "low-sodium", re: /\blow[- ]sodium\b/i },
  { tag: "low-carb", re: /\blow[- ]carb\b/i },
];

const CUISINE_TAGS = [
  { tag: "mexican", re: /\b(taco|tortilla|enchil|quesad|pozole|tamale|mole|salsa|chilaqu)\b/i },
  { tag: "italian", re: /\b(pasta|risotto|pesto|parmig|gnocchi|lasagna|marinara|carbonara)\b/i },
  { tag: "indian", re: /\b(curry|masala|tandoori|naan|dal\b|paneer|biryani|garam)\b/i },
  { tag: "japanese", re: /\b(ramen|miso|teriyaki|udon|soba|yakitori|onigiri|tempura|sushi)\b/i },
  { tag: "korean", re: /\b(kimchi|gochujang|bibimbap|bulgogi|tteok)\b/i },
  { tag: "thai", re: /\b(pad thai|tom yum|coconut milk|green curry|red curry|fish sauce|lemongrass)\b/i },
  { tag: "vietnamese", re: /\b(pho\b|banh mi|nuoc mam|rice paper|vermicelli)\b/i },
  { tag: "chinese", re: /\b(mapo|szech|sichuan|kung pao|dumpling|wonton|lo mein|chow mein)\b/i },
  { tag: "middle-eastern", re: /\b(hummus|tahini|shawarma|falafel|za'atar|tabbouleh)\b/i },
  { tag: "mediterranean", re: /\b(olives|feta|tzatziki|oregano|chickpea|couscous)\b/i },
];

const CATEGORY_TAGS = [
  { tag: "snack", re: /\b(snack|granola bar|trail mix)\b/i },
  { tag: "breakfast", re: /\b(breakfast|pancake|waffle|omelet|oatmeal|granola)\b/i },
  { tag: "dessert", re: /\b(dessert|cake|cookie|brownie|pie|ice cream|pudding)\b/i },
  { tag: "soup", re: /\b(soup|stew|broth|chowder)\b/i },
  { tag: "salad", re: /\b(salad)\b/i },
];

function inferTags(r) {
  const blob = [r.title, r.ingredients, r.instructions].filter(Boolean).join("\n");
  const tags = new Set();
  for (const t of DIET_TAGS) if (t.re.test(blob)) tags.add(t.tag);
  for (const t of CUISINE_TAGS) if (t.re.test(blob)) tags.add(t.tag);
  for (const t of CATEGORY_TAGS) if (t.re.test(blob)) tags.add(t.tag);

  const ing = (r.ingredients || "").toLowerCase();
  if (/\b(tofu|tempeh|nutritional yeast|lentil|chickpea)\b/.test(ing)) tags.add("plant-forward");
  if (/\b(15 min|20 min|30 min|quick|easy)\b/i.test(blob)) tags.add("quick");
  if (/\b(chicken|turkey|beef|fish|salmon|tuna|tofu|lentil|beans|egg|yogurt)\b/i.test(blob))
    tags.add("high-protein");

  return [...tags].slice(0, 14);
}

function buildEmbedDoc(r) {
  const parts = [
    r.title || "",
    (r.tags || []).join(", "),
    r.ingredients || "",
    (r.instructions || "").slice(0, 1200),
  ];
  return normalizeText(parts.join("\n"));
}

// -------------------- PARSING DATASETS --------------------
async function loadOpenRecipes() {
  console.log("Downloading Open Recipes…");
  const gz = await fetchBytes(OPEN_RECIPE_URL);
  const text = gunzip(gz).toString("utf-8");
  const data = safeJsonParseMaybeNDJSON(text, "Open Recipes");

  const out = [];
  for (const it of data) {
    const title = normalizeText(it.name);
    const ingredients = normalizeText(Array.isArray(it.ingredients) ? it.ingredients.join("\n") : it.ingredients);
    const instructions = normalizeText(it.instructions);

    if (!title && !instructions) continue;

    const rec = {
      id: `openrecipes_${out.length}`,
      source: "openrecipes",
      title,
      url: it.url || "",
      ingredients,
      instructions,
      tags: [],
    };
    rec.tags = inferTags(rec);
    rec.ingredients_lines = buildIngredientsLines(rec.ingredients);
    out.push(rec);

    if (out.length >= MAX_OPENRECIPES) break;
  }
  return out;
}

async function loadWikibooks() {
  console.log("Downloading Wikibooks Cookbook…");
  const u8 = await fetchBytes(WIKIBOOKS_JSON_URL);
  const rows = JSON.parse(Buffer.from(u8).toString("utf-8"));

  const out = [];
  for (const row of rows) {
    const rd = row.recipe_data || row; // dataset stores objects under recipe_data
    const title = normalizeText(rd.title);
    const url = rd.url || "";

    const lines = Array.isArray(rd.text_lines) ? rd.text_lines : [];
    const ing = lines
      .filter((x) => String(x.section || "").toLowerCase().includes("ingredient"))
      .map((x) => x.text)
      .join("\n");

    const proc = lines
      .filter((x) => /procedure|directions|method/i.test(x.section || ""))
      .map((x) => x.text)
      .join("\n");

    const fallback = lines.map((x) => x.text).join("\n");

    const ingredients = normalizeText(ing);
    const instructions = normalizeText(proc || fallback);

    if (!title && !instructions) continue;

    const rec = {
      id: `wikibooks_${out.length}`,
      source: "wikibooks",
      title,
      url,
      ingredients,
      instructions,
      tags: [],
    };
    rec.tags = inferTags(rec);
    rec.ingredients_lines = buildIngredientsLines(rec.ingredients);

    out.push(rec);
    if (out.length >= MAX_WIKIBOOKS) break;
  }
  return out;
}

// -------------------- EMBEDDING OUTPUT NORMALIZER --------------------
function getEmbeddingRow(out, j, dim) {
  // v3 Tensor: out.dims shows shape, out.data is flat TypedArray
  // After { pooling: "mean", normalize: true }, shape is [batch, dim].
  if (out?.data && out.data.length >= (j + 1) * dim) {
    return out.data.subarray(j * dim, (j + 1) * dim);
  }

  // Array of tensors (one per input)
  if (Array.isArray(out)) {
    const item = out[j];
    if (item?.data && item.data.length >= dim) return item.data.subarray(0, dim);
    if (item && ArrayBuffer.isView(item) && item.length >= dim) return item.subarray(0, dim);
  }

  throw new Error(`Unexpected embedding output shape for row ${j}. out.data.length=${out?.data?.length}, dims=${out?.dims}`);
}

function isFiniteVec(v) {
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (!Number.isFinite(x)) return false;
  }
  return true;
}

// -------------------- MAIN BUILD --------------------
async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const [openRecipes, wikibooks] = await Promise.all([loadOpenRecipes(), loadWikibooks()]);

  // Merge + dedupe
  const seen = new Set();
  const merged = [];
  for (const r of [...openRecipes, ...wikibooks]) {
    const key = dedupeKey(r);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(r);
  }

  console.log(`Merged recipes (before filter): ${merged.length}`);

  // Filter to recipes with instructions (removes ~80% of junk)
  const withInstructions = merged.filter(
    (r) => r.instructions && r.instructions.trim().length > 10
  );
  console.log(`With instructions: ${withInstructions.length} (removed ${merged.length - withInstructions.length})`);

  // Use filtered set from here on
  const finalRecipes = withInstructions;

  // Load embedder — MUST match the runtime library (@huggingface/transformers v3)
  // so that build-time recipe vectors live in the same space as runtime query vectors.
  console.log("Loading embedding model…");
  env.allowLocalModels = false;
  const embedder = await pipeline("feature-extraction", MODEL_ID, {
    dtype: "fp32",
  });

  // Embeddings
  const vectors = new Float32Array(finalRecipes.length * DIM);

  for (let i = 0; i < finalRecipes.length; i += BATCH) {
    const batch = finalRecipes.slice(i, i + BATCH);
    const docs = batch.map(buildEmbedDoc);

    console.log(`Embedding ${i}..${Math.min(i + BATCH, finalRecipes.length)} / ${finalRecipes.length}`);
    const out = await embedder(docs, { pooling: "mean", normalize: true });

    for (let j = 0; j < batch.length; j++) {
      const row = getEmbeddingRow(out, j, DIM);
      if (!isFiniteVec(row)) {
        vectors.fill(0, (i + j) * DIM, (i + j + 1) * DIM);
        continue;
      }
      vectors.set(row, (i + j) * DIM);
    }
  }

  // Save raw embeddings as a flat Float32Array binary (browser uses brute-force cosine search)
  console.log("Writing embeddings binary…");
  fs.writeFileSync(path.join(OUT_DIR, "embeddings.bin"), Buffer.from(vectors.buffer));

  // Write recipes.json (label order must match embedding order)
  const recipesOut = finalRecipes.map((r, label) => ({
    label,
    id: r.id,
    source: r.source,
    title: r.title,
    url: r.url,
    tags: r.tags,
    ingredients: r.ingredients,
    ingredients_lines: r.ingredients_lines,
    instructions: r.instructions,
  }));
  fs.writeFileSync(path.join(OUT_DIR, "recipes.json"), JSON.stringify(recipesOut));

  // Metadata
  const meta = {
    buildId: BUILD_ID,
    model: MODEL_ID,
    dim: DIM,
    count: finalRecipes.length,
    search: "brute-force-cosine",
    sources: [
      { name: "Wikibooks Cookbook", license: "CC BY-SA 4.0", ref: "https://en.wikibooks.org/wiki/Cookbook:Recipes" },
      { name: "Open Recipes / open-recipe-data", license: "CC BY 3.0", ref: "https://github.com/jakevdp/open-recipe-data" },
    ],
  };
  fs.writeFileSync(path.join(OUT_DIR, "dataset_meta.json"), JSON.stringify(meta, null, 2));

  console.log("Done.");
  console.log(`Wrote:
  - ${OUT_DIR}/recipes.json
  - ${OUT_DIR}/dataset_meta.json
  - ${OUT_DIR}/embeddings.bin
  `);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
