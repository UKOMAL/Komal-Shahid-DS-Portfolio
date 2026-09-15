"""Measure the retrievers instead of asserting things about them.

Two outputs:

  python evaluate.py           recall@1 / recall@3 per mode
  python evaluate.py --sweep   the compression / exact-term trade-off

Read the sweep before you read anything else. It is the actual finding.
"""

import sys
import importlib

from corpus import DOCUMENTS, EXACT_TERM_QUERIES, PARAPHRASE_QUERIES

MODES = ["dense", "bm25", "hybrid"]
ALL_QUERIES = EXACT_TERM_QUERIES + PARAPHRASE_QUERIES


def recall(rag, queries, mode, n):
    return sum(1 for q, e in queries if e in rag.retrieve(q, n, mode=mode)) / len(queries)


def table(rag, title, queries):
    print(f"\n{title}  (n={len(queries)})")
    print(f"  {'mode':<8} {'recall@1':>9} {'recall@3':>9}")
    for mode in MODES:
        print(f"  {mode:<8} {recall(rag, queries, mode, 1):>9.0%} "
              f"{recall(rag, queries, mode, 3):>9.0%}")


def sweep():
    """Vary the LSA truncation rank and watch exact-term recall move."""
    import dense, pipeline
    print("\nCompression vs exact-term recall")
    print("Lower rank = more aggressive compression of the term space.\n")
    print(f"  {'rank':>5} {'dense exact@1':>14} {'dense para@1':>13} "
          f"{'bm25 exact@1':>13} {'hybrid exact@1':>15}")
    print("  " + "-" * 64)
    for k in (2, 3, 4, 5, 8, 12):
        dense.COMPONENTS = k
        importlib.reload(pipeline)
        rag = pipeline.HybridRAG(DOCUMENTS)
        print(f"  {k:>5} {recall(rag, EXACT_TERM_QUERIES, 'dense', 1):>14.0%} "
              f"{recall(rag, PARAPHRASE_QUERIES, 'dense', 1):>13.0%} "
              f"{recall(rag, EXACT_TERM_QUERIES, 'bm25', 1):>13.0%} "
              f"{recall(rag, EXACT_TERM_QUERIES, 'hybrid', 1):>15.0%}")
    print("""
  Reading this: as the rank drops, the dense retriever loses exact-term
  recall while BM25 holds at 100%, because a rare identifier lives in a
  low-variance direction that truncation throws away first. Hybrid tracks
  BM25 on those queries and keeps the dense retriever's paraphrase ability.

  Note honestly: on a corpus this small, a rank-5+ LSA model already
  resolves every exact-term query, so hybrid TIES dense rather than
  beating it. The compression failure is real and reproducible, but a
  16-document toy corpus is not where it bites. See the README.""")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
        raise SystemExit

    from pipeline import HybridRAG
    rag = HybridRAG(DOCUMENTS)
    print(f"documents: {len(DOCUMENTS)}   dense backend: {rag.dense.backend}")
    table(rag, "Exact-term queries", EXACT_TERM_QUERIES)
    table(rag, "Paraphrase queries", PARAPHRASE_QUERIES)
    table(rag, "All queries", ALL_QUERIES)
    print("\nRun with --sweep to see the compression trade-off.")
