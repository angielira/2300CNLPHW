"""Build the Word Map's model files from Google's pretrained Word2Vec vectors.

Input
  - GoogleNews-vectors-negative300 (Word2Vec, skip-gram trained by Google on ~100 billion words of
    Google News; 3,000,000 words and phrases x 300 dimensions). Download either way:
        python -c "import gensim.downloader as api; print(api.load('word2vec-google-news-300', return_path=True))"
        curl -L -o word2vec-google-news-300.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz
  - Spanish data from github.com/doozan/spanish_data: es-en.data (Wiktionary, CC-BY-SA) and frequency.csv
  - translations.txt: hand-made English->Spanish table from the project (fills gaps)

Output (written to ../public)
  - vectors.bin : int8 vectors, one row of 300 values per word (row-normalized, scaled to [-127, 127])
  - vocab.json  : words, Spanish translations, and a Spanish -> English search table

Usage
  python build_word2vec.py --w2v word2vec-google-news-300.gz --es-data spanish_data/ [--words 25000]
"""
import argparse, csv, gzip, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--w2v", required=True, help="GoogleNews-vectors-negative300 .bin or .bin.gz / gensim-data .gz")
p.add_argument("--es-data", required=True, help="folder with es-en.data and frequency.csv")
p.add_argument("--translations", default=Path(__file__).with_name("translations.txt"))
p.add_argument("--words", type=int, default=25000, help="how many of the most frequent words to keep")
p.add_argument("--out", default=Path(__file__).resolve().parent.parent / "public")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ---------- 1. Read Word2Vec binary format (no gensim needed) ----------
# Format: header "<count> <dim>\n", then per entry: word bytes, a space, dim float32 values.
# Entries are sorted by frequency, so the first matches are the most common words.
WORD_RE = re.compile(r"^[a-z]+(?:[-'][a-z]+)?$")   # lowercase single words only (skips names, phrases_with_underscores)
opener = gzip.open if str(a.w2v).endswith(".gz") else open
words, rows = [], []
with opener(a.w2v, "rb") as f:
    total, dim = map(int, f.readline().split())
    rowbytes = 4 * dim
    for _ in range(total):
        wb = bytearray()
        while True:
            ch = f.read(1)
            if ch == b" ": break
            if ch != b"\n": wb.extend(ch)
        vec = f.read(rowbytes)
        w = wb.decode("utf-8", errors="ignore")
        if WORD_RE.match(w) and (len(w) > 1 or w in ("a", "i")):
            words.append(w); rows.append(np.frombuffer(vec, dtype="<f4"))
            if len(words) == a.words: break
E = np.stack(rows).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
idx = {w: i for i, w in enumerate(words)}
print(f"Word2Vec: kept {len(words)} of {total:,} entries, {dim} dimensions", file=sys.stderr)

# ---------- 2. English <-> Spanish ----------
sfreq = {}
with open(Path(a.es_data) / "frequency.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        w = row["spanish"].strip().lower(); sfreq[w] = max(sfreq.get(w, 0), int(row["count"]))

SKIP = re.compile(r"\b(form|plural|spelling|singular|feminine|masculine|abbreviation|initialism|alternative|misspelling|obsolete|archaic|diminutive|augmentative|superlative|conjugation|participle|gerund|clipping|synonym|eye dialect|letter|name of)\b", re.I)
STOP = set("a an the to of in on at by for with from as or and be is are was its his her their your my our one some any this that".split())
def candidates(gloss):
    g = re.sub(r"\([^)]*\)", " ", gloss).replace('"', "")
    found = []
    for part in re.split(r"[;,/]| or ", g):
        w = re.sub(r"^(to|a|an|the)\s+", "", part.strip().lower())
        if w in idx and w not in STOP: found.append(w)
    return found

es2en = defaultdict(list); head = pos = None
with open(Path(a.es_data) / "es-en.data", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line == "_____": head = None; continue
        if head is None: head = line.strip().lower(); continue
        s = line.strip()
        if s.startswith("pos:"): pos = s[4:].strip(); continue
        if line.startswith("  gloss:") and pos not in ("prop", "suffix", "prefix", "letter", "abbrev"):
            g = s[6:].strip()
            if SKIP.search(g) or not re.match(r"^[a-záéíóúñü]+(?:[- ][a-záéíóúñü]+)?$", head): continue
            for c in candidates(g):
                if c not in es2en[head]: es2en[head].append(c)

# English -> Spanish: the most frequent Spanish word whose FIRST meaning is the English word.
cand = defaultdict(list)
for es, ens in es2en.items():
    for rank, en in enumerate(ens[:4]):
        cand[en].append((sfreq.get(es, 0) * (1.0 if rank == 0 else 0.05 / rank), es))
en2es = {en: max(c)[1] for en, c in cand.items() if max(c)[0] > 0}
curated = {}
for line in open(a.translations, encoding="utf-8"):
    q = line.rstrip("\n").split("|")
    if len(q) == 4 and not q[0].startswith("#") and q[0] in idx: curated[q[0]] = q[0] if q[3] == "=" else q[3]
en2es = {**curated, **en2es}
en2es.update({k: v for k, v in {"time": "tiempo", "store": "tienda", "media": "medios", "post": "publicación",
              "spring": "primavera", "season": "temporada", "play": "jugar", "kind": "tipo", "case": "caso", "fine": "bien"}.items() if k in idx})
es_list = [en2es.get(w, "") for w in words]

es_search = {}
for es, ens in es2en.items():
    for en in ens:
        if en in idx: es_search[es] = idx[en]; break
for i, es in enumerate(es_list):
    if es and es not in es_search: es_search[es] = i
print(f"Spanish: {sum(map(bool, es_list))} English words translated, {len(es_search)} Spanish words searchable", file=sys.stderr)

# ---------- 3. Write files ----------
Q = np.clip(np.round(E * 127 / np.abs(E).max(axis=1, keepdims=True)), -127, 127).astype(np.int8)
Q.tofile(out / "vectors.bin")
json.dump({"dim": dim, "words": words, "es": es_list, "esSearch": es_search,
           "model": "Word2Vec", "source": f"Google News Word2Vec (skip-gram, ~100B words), the {len(words):,} most frequent lowercase words, 300 dimensions",
           "dictionary": "Wiktionary via doozan/spanish_data (CC-BY-SA)"},
          open(out / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# ---------- 4. Sanity check: nearest neighbors ----------
def nearest(w, k=10):
    s = E @ E[idx[w]]; return [words[i] for i in np.argsort(-s)[1:k + 1]]
for w in ["fear", "bakery", "brand", "logo", "coffee", "happy", "spanish"]:
    if w in idx: print(f"  {w:8} -> {', '.join(nearest(w))}", file=sys.stderr)
q = E[idx["king"]] - E[idx["man"]] + E[idx["woman"]]; q /= np.linalg.norm(q)
s = E @ q
print("  king - man + woman ->", [words[i] for i in np.argsort(-s) if words[i] not in ("king", "man", "woman")][:3], file=sys.stderr)
print(f"wrote {out/'vectors.bin'} ({(out/'vectors.bin').stat().st_size//1024} KB) and vocab.json", file=sys.stderr)
