# Studio Word Map

Type an English or Spanish word to see its 10 closest words in general English, view the neighborhood
in 2D or 3D, read it in English and Spanish, and collect coins under your username.

- `public/` — the site (index.html, vocab.json, vectors.bin). No build step.
- `netlify/functions/coins.mjs` — `/api/coins`: coins per username and the leaderboard, stored in Netlify Blobs.

**Data**
- Word vectors: GloVe 6B, 100 dimensions (Wikipedia 2014 + Gigaword 5), the 30,000 most frequent words,
  quantized to int8. Stanford NLP, Public Domain Dedication and License.
- English–Spanish: Wiktionary via github.com/doozan/spanish_data (CC-BY-SA), frequency list from
  github.com/hermitdave/FrequencyWords (CC-BY-SA 3.0).
