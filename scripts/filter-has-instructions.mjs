// Filters the dataset to only recipes that have instructions.
// Also rebuilds embeddings.bin to match the new label order.
import fs from "node:fs";
import path from "node:path";

const ASSETS = "assets";
const DIM = 384;

const recipes = JSON.parse(fs.readFileSync(path.join(ASSETS, "recipes.json"), "utf-8"));
const embBuf = fs.readFileSync(path.join(ASSETS, "embeddings.bin"));
const allVectors = new Float32Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 4);

console.log(`Total recipes: ${recipes.length}`);

const filtered = [];
const filteredVecs = [];

for (let i = 0; i < recipes.length; i++) {
  const r = recipes[i];
  if (r.instructions && r.instructions.trim().length > 10) {
    const newLabel = filtered.length;
    filtered.push({ ...r, label: newLabel });
    filteredVecs.push(allVectors.subarray(i * DIM, (i + 1) * DIM));
  }
}

console.log(`With instructions: ${filtered.length}`);
console.log(`Removed: ${recipes.length - filtered.length}`);

// Write filtered recipes
fs.writeFileSync(path.join(ASSETS, "recipes.json"), JSON.stringify(filtered));
console.log(`Wrote recipes.json (${filtered.length} recipes)`);

// Write filtered embeddings
const newVectors = new Float32Array(filtered.length * DIM);
for (let i = 0; i < filteredVecs.length; i++) {
  newVectors.set(filteredVecs[i], i * DIM);
}
fs.writeFileSync(path.join(ASSETS, "embeddings.bin"), Buffer.from(newVectors.buffer));
console.log(`Wrote embeddings.bin (${newVectors.byteLength} bytes)`);

// Update meta
const meta = JSON.parse(fs.readFileSync(path.join(ASSETS, "dataset_meta.json"), "utf-8"));
meta.count = filtered.length;
meta.buildId = new Date().toISOString().slice(0, 10);
fs.writeFileSync(path.join(ASSETS, "dataset_meta.json"), JSON.stringify(meta, null, 2));
console.log(`Updated dataset_meta.json (count: ${meta.count})`);
