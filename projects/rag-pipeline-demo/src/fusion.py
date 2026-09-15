"""Reciprocal Rank Fusion.

Merges several ranked lists into one. The key property is that it works on
RANK POSITION, not on score, so a BM25 score (unbounded, corpus-dependent)
and a cosine similarity (bounded 0..1) can be combined without any attempt
to normalise them onto a shared scale.

    RRF(d) = sum over rankings of  1 / (k + rank(d))

k dampens the influence of the very top positions; 60 is the value from the
original Cormack et al. paper and is what the production system uses.
"""

from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + position + 1)
    return sorted(scores, key=lambda d: -scores[d])
