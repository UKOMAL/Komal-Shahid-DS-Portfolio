"""Dense (semantic) retrieval.

Two backends:

* BGE via sentence-transformers, if it is installed. This is the model the
  production system uses. Set USE_MODEL = True.
* Latent Semantic Analysis over the corpus, as the default. TF-IDF weighted
  term-document matrix, truncated SVD, cosine similarity in the reduced
  space. Only numpy required.

LSA is a genuine semantic method, not a stand-in: it matches on co-occurrence
structure, so it can connect "graphics card" to "GPU" when those words appear
in similar contexts. It is much weaker than a transformer, but it exhibits the
same characteristic behaviour — good on paraphrase, lossy on rare exact tokens,
because truncating to k dimensions discards precisely the low-variance
directions that a unique identifier lives in.

That trade-off is the whole reason hybrid retrieval exists.
"""

import re
import math
import numpy as np

USE_MODEL = False
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COMPONENTS = 5   # truncation rank; see evaluate.py --sweep for why this matters


def _tokens(text):
    return re.findall(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", text.lower())


class DenseRetriever:
    def __init__(self, documents):
        self.ids = [d["id"] for d in documents]
        self.texts = [d["text"] for d in documents]
        self.model = None

        if USE_MODEL:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(MODEL_NAME)
                self.vectors = self.model.encode(self.texts, normalize_embeddings=True)
                return
            except Exception:
                self.model = None

        self._fit_lsa()

    # ---------- LSA ----------
    def _fit_lsa(self):
        docs = [_tokens(t) for t in self.texts]
        vocab = sorted({w for d in docs for w in d})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n_docs = len(docs)

        tf = np.zeros((n_docs, len(vocab)))
        for i, d in enumerate(docs):
            for w in d:
                tf[i, self.vocab[w]] += 1.0
            if len(d):
                tf[i] /= len(d)

        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        X = tf * self.idf

        # Truncated SVD. Vt rows are the semantic axes.
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(COMPONENTS, len(S))
        self.axes = Vt[:k]                    # k x |vocab|
        self.vectors = self._normalize(X @ self.axes.T)

    @staticmethod
    def _normalize(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms

    def _embed(self, text):
        if self.model is not None:
            return self.model.encode([text], normalize_embeddings=True)[0]
        v = np.zeros(len(self.vocab))
        toks = _tokens(text)
        for w in toks:
            if w in self.vocab:
                v[self.vocab[w]] += 1.0
        if len(toks):
            v /= len(toks)
        v = v * self.idf
        q = self.axes @ v
        n = np.linalg.norm(q)
        return q / n if n else q

    def search(self, query, k=5):
        q = self._embed(query)
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        return [self.ids[i] for i in order]

    @property
    def backend(self):
        return MODEL_NAME if self.model is not None else f"LSA-{COMPONENTS}d"
