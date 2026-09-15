"""The hybrid pipeline: BM25 + dense, fused with RRF."""

from bm25 import BM25
from dense import DenseRetriever
from fusion import reciprocal_rank_fusion


class HybridRAG:
    def __init__(self, documents, k=60):
        self.documents = {d["id"]: d["text"] for d in documents}
        self.bm25 = BM25(documents)
        self.dense = DenseRetriever(documents)
        self.k = k

    def retrieve(self, query, top_k=3, mode="hybrid"):
        if mode == "bm25":
            return self.bm25.search(query, top_k)
        if mode == "dense":
            return self.dense.search(query, top_k)

        # Retrieve wider than we return, then fuse and cut.
        dense_hits = self.dense.search(query, top_k * 3)
        sparse_hits = self.bm25.search(query, top_k * 3)
        return reciprocal_rank_fusion([dense_hits, sparse_hits], k=self.k)[:top_k]

    def answer_context(self, query, top_k=3):
        """What would be handed to the LLM, with citations."""
        return [
            {"id": doc_id, "text": self.documents[doc_id]}
            for doc_id in self.retrieve(query, top_k)
        ]
