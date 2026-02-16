// Quick script: extract raw embedding vectors from existing hnsw.index
// so the browser can use brute-force cosine search (no hnswlib-wasm needed)
import fs from "node:fs";
import path from "node:path";
import hnswPkg from "hnswlib-node";
const { HierarchicalNSW } = hnswPkg;

const META = JSON.parse(fs.readFileSync("assets/dataset_meta.json", "utf-8"));
const DIM = META.dim;       // 384
const COUNT = META.count;   // 18840

console.log(`Loading hnsw.index (dim=${DIM}, count=${COUNT})…`);
const index = new HierarchicalNSW("cosine", DIM);
index.readIndexSync(path.join("assets", "hnsw.index"));

const vectors = new Float32Array(COUNT * DIM);
for (let label = 0; label < COUNT; label++) {
  const vec = index.getPoint(label);
  vectors.set(vec, label * DIM);
  if (label % 5000 === 0) console.log(`  extracted ${label}/${COUNT}`);
}

const outPath = path.join("assets", "embeddings.bin");
fs.writeFileSync(outPath, Buffer.from(vectors.buffer));
console.log(`Wrote ${outPath} (${vectors.byteLength} bytes)`);
