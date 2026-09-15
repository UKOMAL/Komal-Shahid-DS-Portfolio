"""Show all three retrievers side by side for one query.

    python demo.py "CUDA_ERR_4417"
"""

import sys
from corpus import DOCUMENTS
from pipeline import HybridRAG

query = " ".join(sys.argv[1:]) or "my graphics card isn't showing up inside docker"
rag = HybridRAG(DOCUMENTS)

print(f'query: "{query}"')
print(f"dense backend: {rag.dense.backend}\n")

for mode in ("dense", "bm25", "hybrid"):
    print(f"{mode}:")
    for i, doc_id in enumerate(rag.retrieve(query, 3, mode=mode), 1):
        text = rag.documents[doc_id]
        print(f"  {i}. {doc_id}  {text[:72]}...")
    print()

print("context handed to the LLM (hybrid, top 2):")
for c in rag.answer_context(query, 2):
    print(f"  [{c['id']}] {c['text'][:88]}...")
