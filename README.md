# Studio Word Map

Type an English or Spanish word to see its 10 closest words in a **Word2Vec** model, view the neighborhood
in 2D or 3D (principal component analysis), read it in English and Spanish, and collect coins under your username.

Live: https://2300nlphw2.netlify.app

## How it works

1. **Model:** Google's pretrained Word2Vec (skip-gram, trained on ~100 billion words of Google News).
   `model/build_word2vec.py` keeps the 25,000 most frequent lowercase English words at 300 dimensions
   and writes `public/vectors.bin` (int8) and `public/vocab.json`.
2. **Closest words:** the browser computes cosine similarity between the searched word's vector and all
   25,000 others and shows the top 10. `+` / `-` do vector arithmetic: `king - man + woman` → queen.
3. **Maps:** for each search, PCA on the word and its 50/100/200 nearest neighbors; the top 2 components
   give the 2D view and the top 3 give the 3D view (`pca()` and `renderPlot()` in `public/index.html`).
4. **Spanish:** a Spanish word is translated to English first (miedo → fear), then searched. Results can be
   shown in English or Spanish. Translations come from Wiktionary.
5. **Coins:** each new word earns 10 coins (15 if rare). `netlify/functions/coins.mjs` (`/api/coins`) saves
   each username's discovered words in Netlify Blobs and serves a top-10 leaderboard.

## Files

| Path | What |
| --- | --- |
| `public/` | The site: `index.html`, `vocab.json`, `vectors.bin`. No build step. |
| `netlify/functions/coins.mjs` | Coins per username + leaderboard |
| `model/build_word2vec.py` | Builds `public/` model files from Google News Word2Vec |
| `model/train_word2vec_numpy.py` | Word2Vec (skip-gram, negative sampling) implemented from scratch in NumPy |
| `model/translations.txt` | Hand-made English→Spanish table used to fill dictionary gaps |

## Rebuild the model

```bash
cd model
pip install -r requirements.txt
curl -L -o word2vec-google-news-300.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz
git clone --depth 1 https://github.com/doozan/spanish_data.git
python build_word2vec.py --w2v word2vec-google-news-300.gz --es-data spanish_data
```

## Data and licenses

- Word2Vec Google News vectors: Mikolov et al., 2013, "Distributed Representations of Words and Phrases
  and their Compositionality" (Apache 2.0, code.google.com/archive/p/word2vec).
- English–Spanish: Wiktionary via github.com/doozan/spanish_data (CC-BY-SA); Spanish word frequencies from
  github.com/hermitdave/FrequencyWords (CC-BY-SA 3.0).
