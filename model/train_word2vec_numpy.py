"""Word2Vec from scratch: skip-gram with negative sampling, written in NumPy (no gensim).

This is the model the Word Map first used, trained on this project's own text (prompts, avatar research,
diary, brand system, site copy). The live site now uses Google's pretrained Word2Vec instead
(see build_word2vec.py), because a ~7,500-word corpus is too small for general-English neighbors.
Put .txt/.md files in model/corpus/ and run:  python train_word2vec_numpy.py

Outputs:
  out/vectors.txt   word2vec text format (loadable with gensim KeyedVectors.load_word2vec_format)
  out/model.json    vocab, counts, normalized vectors, 2D + 3D t-SNE and PCA coordinates
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

DIM, WINDOW, NEG, EPOCHS, MIN_COUNT, LR0, BATCH = 64, 5, 5, 60, 3, 0.05, 512
rng = np.random.default_rng(7)

STOP = set("""
a about above after again against all also am an and any are aren't as at be because been before being below between both but by
can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't
have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves just will one also get like use used using via e.g etc per s t
el la los las de del y o en un una unos unas que se por para con su sus al lo le les mi tu es son te me nos ya mas más pero como
""".split())


def text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ts":
        # Site copy file: keep only the human-readable string literals, not the code.
        return "\n".join(m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.){3,})"', raw))
    # Drop fenced code blocks (component source code, JSON, CSS) so the vocabulary is about meaning, not syntax.
    raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
    return raw


def tokenize(text: str):
    text = unicodedata.normalize("NFC", text.lower()).replace("’", "'")
    # Sentence-ish segments so windows don't jump across headings/bullets.
    for seg in re.split(r"[.!?\n:;|•]+", text):
        toks = re.findall(r"[a-záéíóúñü]+(?:'[a-z]+)?(?:-[a-záéíóúñü]+)*", seg)
        toks = [t for t in toks if len(t) > 1 and t not in STOP]
        if len(toks) >= 2:
            yield toks


files = sorted(CORPUS.glob("*"))
sentences = [s for f in files for s in tokenize(text_of(f))]
counts = Counter(w for s in sentences for w in s)
vocab = [w for w, c in counts.most_common() if c >= MIN_COUNT]
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
sentences = [[idx[w] for w in s if w in idx] for s in sentences]
sentences = [s for s in sentences if len(s) >= 2]
n_tokens = sum(len(s) for s in sentences)
print(f"files={len(files)} raw_tokens={sum(counts.values())} vocab={V} train_tokens={n_tokens}", file=sys.stderr)

freq = np.array([counts[w] for w in vocab], dtype=np.float64)
# Frequent-word subsampling (Mikolov et al.) and the unigram^0.75 noise distribution.
t = 1e-3
p_keep = np.minimum(1.0, (np.sqrt(freq / (t * freq.sum())) + 1) * (t * freq.sum()) / freq)
noise = freq ** 0.75
noise /= noise.sum()

W_in = (rng.random((V, DIM)) - 0.5) / DIM
W_out = np.zeros((V, DIM))


def make_pairs():
    centers, contexts = [], []
    for s in sentences:
        s = [w for w in s if rng.random() < p_keep[w]]
        for i, c in enumerate(s):
            b = rng.integers(1, WINDOW + 1)  # dynamic window
            for j in range(max(0, i - b), min(len(s), i + b + 1)):
                if j != i:
                    centers.append(c)
                    contexts.append(s[j])
    return np.array(centers), np.array(contexts)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


for epoch in range(EPOCHS):
    lr = LR0 * (1 - epoch / EPOCHS) + 1e-4
    c_ids, o_ids = make_pairs()
    order = rng.permutation(len(c_ids))
    c_ids, o_ids = c_ids[order], o_ids[order]
    loss_sum = 0.0
    for start in range(0, len(c_ids), BATCH):
        c = c_ids[start:start + BATCH]
        o = o_ids[start:start + BATCH]
        n = rng.choice(V, size=(len(c), NEG), p=noise)
        h = W_in[c]                                   # (B, D)
        targets = np.concatenate([o[:, None], n], 1)  # (B, 1+NEG)
        U = W_out[targets]                            # (B, 1+NEG, D)
        scores = np.einsum("bd,bkd->bk", h, U)
        labels = np.zeros_like(scores); labels[:, 0] = 1
        p = sigmoid(scores)
        g = (p - labels)                              # gradient of logistic loss
        loss_sum += -(np.log(p[:, 0] + 1e-9).sum() + np.log(1 - p[:, 1:] + 1e-9).sum())
        grad_h = np.einsum("bk,bkd->bd", g, U)
        grad_U = g[:, :, None] * h[:, None, :]
        np.add.at(W_out, targets, -lr * grad_U)
        np.add.at(W_in, c, -lr * grad_h)
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch + 1}/{EPOCHS} pairs={len(c_ids)} loss/pair={loss_sum / len(c_ids):.3f}", file=sys.stderr)

E = W_in + W_out  # summing input+output vectors is a common, slightly stronger choice for small corpora
E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)

# word2vec text format
with open(OUT / "vectors.txt", "w") as f:
    f.write(f"{V} {DIM}\n")
    for w, v in zip(vocab, E):
        f.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")


def nn(word, k=10):
    sims = E_norm @ E_norm[idx[word]]
    best = np.argsort(-sims)[1:k + 1]
    return [(vocab[i], round(float(sims[i]), 3)) for i in best]


for probe in ["brand", "instagram", "logo", "bakery", "fear", "price", "spanish", "daniela", "content", "customers"]:
    if probe in idx:
        print(probe, "->", ", ".join(w for w, _ in nn(probe)), file=sys.stderr)

# Projections: t-SNE (neighborhoods) and PCA (global directions), 2D and 3D.
perp = min(30, max(5, V // 40))
tsne2 = TSNE(2, metric="cosine", perplexity=perp, init="pca", random_state=7, max_iter=1500).fit_transform(E_norm)
tsne3 = TSNE(3, metric="cosine", perplexity=perp, init="pca", random_state=7, max_iter=1500).fit_transform(E_norm)
pca2 = PCA(2, random_state=7).fit_transform(E_norm)
pca3 = PCA(3, random_state=7).fit_transform(E_norm)


def scale(a):
    a = a - a.mean(0)
    return np.round(a / np.abs(a).max(), 4).tolist()


model = {
    "meta": {
        "dim": DIM, "window": WINDOW, "negative": NEG, "epochs": EPOCHS, "min_count": MIN_COUNT,
        "vocab_size": V, "train_tokens": n_tokens, "files": [f.name for f in files], "perplexity": perp,
    },
    "words": vocab,
    "counts": [int(counts[w]) for w in vocab],
    # Normalized vectors, quantized to int8 to keep the page small; cosine ranking is preserved closely.
    "vectors_q": np.round(E_norm * 127).astype(int).tolist(),
    "tsne2": scale(tsne2), "tsne3": scale(tsne3), "pca2": scale(pca2), "pca3": scale(pca3),
}
(OUT / "model.json").write_text(json.dumps(model, separators=(",", ":")))
print("wrote", OUT / "model.json", f"{(OUT / 'model.json').stat().st_size / 1024:.0f} KB", file=sys.stderr)
