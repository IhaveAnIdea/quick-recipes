# Attribution & Licensing

## Application Code

Quick Recipes is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Craig Winter

## Recipe Data

This project includes recipe content from the following sources. The recipe
data (titles, ingredients, instructions, tags) is redistributed under the
terms of their respective licenses.

### Wikibooks Cookbook

- **Source:** <https://en.wikibooks.org/wiki/Cookbook:Recipes>
- **License:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- **Notes:** Recipe titles, ingredients, and instructions are sourced from
  community-contributed Wikibooks Cookbook pages. Original page URLs are
  preserved where available.

### Open Recipes / open-recipe-data

- **Source:** <https://github.com/jakevdp/open-recipe-data>
- **License:** [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/)
- **Notes:** Recipe data compiled from various openly licensed sources.
  Original recipe URLs and source identifiers are preserved where available.

## AI / Embedding Model

- **Model:** [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
- **Library:** [@huggingface/transformers](https://github.com/huggingface/transformers.js) (Apache 2.0)
- **Runtime:** [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT)
- **Notes:** Prebuilt sentence embeddings are used for semantic search.
  The embedding model runs client-side in the browser via WebAssembly.

## Third-Party Libraries (loaded on demand)

| Library | License | Use |
|---------|---------|-----|
| [jsPDF](https://github.com/parallax/jsPDF) | MIT | PDF export |
| [docx](https://github.com/dolanmiri/docx) | MIT | DOCX export |

## CC BY-SA 4.0 — ShareAlike Notice

Portions of the recipe data are licensed under CC BY-SA 4.0. If you
redistribute the recipe dataset (or a derivative), you must license your
version under the same or a compatible license and provide attribution to
the Wikibooks Cookbook contributors.

The application source code itself is MIT-licensed and is **not** subject
to the ShareAlike requirement — only the recipe content is.
