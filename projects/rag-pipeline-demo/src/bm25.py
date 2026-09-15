"""BM25 Okapi, implemented from scratch. No dependencies.

Lexical retrieval. Scores a document by how often the query's terms appear
in it, damped by term frequency saturation and normalised by document
length, and weighted by how rare each term is across the corpus.

The property that matters here: a rare token like CUDA_ERR_4417 gets a very
high IDF, so a document containing it scores far above one that does not.
That is exactly what dense retrieval cannot guarantee.
"""

import math
import re
from collections import Counter


def tokenize(text):
    # Keep digits and underscores attached so CUDA_ERR_4417 and A10G-24
    # survive as single tokens rather than being split into fragments.
    return re.findall(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.ids = [d["id"] for d in documents]
        self.docs = [tokenize(d["text"]) for d in documents]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.docs)
        self.freqs = [Counter(d) for d in self.docs]

        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        # Probabilistic IDF with the +0.5 smoothing from Robertson/Sparck Jones.
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score(self, query, doc_index):
        total = 0.0
        freqs = self.freqs[doc_index]
        length = self.lengths[doc_index]
        for term in tokenize(query):
            if term not in freqs:
                continue
            f = freqs[term]
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf.get(term, 0.0) * numerator / denominator
        return total

    def search(self, query, k=5):
        scored = [(self.ids[i], self.score(query, i)) for i in range(len(self.docs))]
        scored = [(doc_id, s) for doc_id, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return [doc_id for doc_id, _ in scored[:k]]
