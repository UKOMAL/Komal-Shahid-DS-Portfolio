# Reference implementation

A runnable hybrid retrieval pipeline. No API keys, no model downloads, no
setup beyond numpy.

```bash
cd src
python evaluate.py           # recall@1 / recall@3 for each retriever
python evaluate.py --sweep   # the compression trade-off — read this one
python test_pipeline.py      # 13 tests
python demo.py "CUDA_ERR_4417"
```

## Files

| File | What it is |
|---|---|
| `bm25.py` | BM25 Okapi from scratch. The tokenizer deliberately keeps `CUDA_ERR_4417` and `A10G-24` intact — splitting identifiers is how lexical retrieval gets quietly broken. |
| `dense.py` | Semantic retrieval. LSA over the corpus by default; BGE via sentence-transformers if you set `USE_MODEL = True`. |
| `fusion.py` | Reciprocal Rank Fusion. Operates on rank position, so BM25 scores and cosine similarities never need to be normalised onto a shared scale. |
| `pipeline.py` | Wires the three together. `mode=` selects `dense`, `bm25` or `hybrid`. |
| `corpus.py` | 16 support documents, all topically similar on purpose, plus labelled exact-term and paraphrase query sets. |
| `evaluate.py` | Measurement. |
| `test_pipeline.py` | Tests, including assertions on the retrieval claims made here. |

## What this demonstrates, and what it doesn't

Running the sweep shows the mechanism clearly:

```
   rank  dense exact@1  dense para@1  bm25 exact@1  hybrid exact@1
      2            25%           25%          100%            100%
      3            75%           25%          100%            100%
      4           100%           50%          100%            100%
      5           100%           75%          100%            100%
```

As the truncation rank drops, the dense retriever loses exact-term recall
while BM25 holds at 100%. The reason is structural: a rare identifier lives
in a low-variance direction of the term space, and truncation discards
low-variance directions first. Compression is lossy exactly where precision
matters most. Hybrid recovers it.

**Being straight about the limits.** On a 16-document corpus, a rank-5 LSA
model already answers every exact-term query, so at the default setting
hybrid *ties* dense at rank 1 rather than beating it:

```
All queries (n=8)     recall@1   recall@3
  dense                    88%        88%
  bm25                     75%       100%
  hybrid                   88%       100%
```

Hybrid wins on recall@3 and never loses to either single retriever — but a
toy corpus cannot prove hybrid superiority, and this one doesn't claim to.
The failure this models is one I hit on a production corpus of a very
different size, where the dense index had far more competing documents per
query and the compression loss was not recoverable by widening `k`.

What you can take from the code: the mechanism is real, reproducible, and
visible in four lines of a sweep. What you cannot take from it: a benchmark
result.

## Swapping in the real embedding model

```python
# dense.py
USE_MODEL = True
MODEL_NAME = "BAAI/bge-small-en-v1.5"
```

Requires `sentence-transformers` and `torch`. The retriever interface does
not change, so `pipeline.py` and `evaluate.py` work unmodified — which is the
point of keeping retrieval behind a `search(query, k)` method.
