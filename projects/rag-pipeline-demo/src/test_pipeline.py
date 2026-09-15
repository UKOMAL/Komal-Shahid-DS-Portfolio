"""Tests. Run: python test_pipeline.py"""

from corpus import DOCUMENTS, EXACT_TERM_QUERIES, PARAPHRASE_QUERIES
from bm25 import BM25, tokenize
from fusion import reciprocal_rank_fusion
from pipeline import HybridRAG

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


print("tokenizer")
check("identifier survives intact", "cuda_err_4417" in tokenize("Error CUDA_ERR_4417 raised"))
check("hyphenated part number survives", "a10g-24" in tokenize("Part number A10G-24 is"))

print("\nbm25")
bm = BM25(DOCUMENTS)
check("finds the exact identifier", bm.search("CUDA_ERR_4417", 1) == ["KB-1001"])
check("scores zero for absent terms", bm.score("zzzz_not_present", 0) == 0.0)

print("\nrrf")
# A doc both retrievers rank first should win outright.
fused = reciprocal_rank_fusion([["b", "a", "c"], ["b", "c", "a"]])
check("doc ranked first by both wins", fused[0] == "b", f"got {fused}")
check("all docs retained", set(fused) == {"a", "b", "c"})
single = reciprocal_rank_fusion([["x", "y"]])
check("single ranking passes through", single == ["x", "y"])

# k controls how much rank position matters. At large k the curve is nearly
# flat, so being first in one list barely outweighs being second in both.
# At small k the top slot dominates. This is the knob, and it is worth
# knowing which way it turns.
tight = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=1)
check("small k favours the top slot", tight[0] in ("a", "c"), f"got {tight}")
loose = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=1000)
check("large k flattens toward consistency", abs(len(loose) - 3) == 0)

print("\npipeline")
rag = HybridRAG(DOCUMENTS)
check("returns requested depth", len(rag.retrieve("gpu memory", 3)) == 3)
check("modes are distinguishable",
      rag.retrieve("authenticate to the cloud", 3, mode="bm25")
      != rag.retrieve("authenticate to the cloud", 3, mode="dense"))
check("context carries citations",
      all("id" in c and "text" in c for c in rag.answer_context("CUDA_ERR_4417", 2)))

print("\nretrieval quality (the claims the README makes)")
ex1 = sum(1 for q, e in EXACT_TERM_QUERIES if e in rag.retrieve(q, 1, mode="bm25"))
check("bm25 gets every exact-term query at rank 1", ex1 == len(EXACT_TERM_QUERIES),
      f"{ex1}/{len(EXACT_TERM_QUERIES)}")
h3 = sum(1 for q, e in EXACT_TERM_QUERIES + PARAPHRASE_QUERIES
         if e in rag.retrieve(q, 3, mode="hybrid"))
check("hybrid recalls every query by rank 3", h3 == 8, f"{h3}/8")
d3 = sum(1 for q, e in EXACT_TERM_QUERIES + PARAPHRASE_QUERIES
         if e in rag.retrieve(q, 3, mode="dense"))
check("hybrid recall@3 beats dense-only", h3 > d3, f"hybrid {h3} vs dense {d3}")

print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
