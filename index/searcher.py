"""Searcher: thin orchestration layer over BM25 and vector recall.

Both recall paths take the same query string and return the same shape,
list[(doc_id: str, score: float)] sorted by score descending.
Merging / fusion of the two is Week 4, not here.
"""

from __future__ import annotations

try:
    from .bm25 import bm25_search as bm25_rank, InvertedIndex
    from .embeddings import VectorStore
except ImportError:
    from bm25 import bm25_search as bm25_rank, InvertedIndex
    from embeddings import VectorStore


class Searcher:
    def __init__(self, bm25_index: InvertedIndex, vector_store: VectorStore) -> None:
        self.bm25_index = bm25_index
        self.vector_store = vector_store

    def bm25_search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        return bm25_rank(self.bm25_index, query, k)

    def vector_search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        return self.vector_store.search(query, k)


if __name__ == "__main__":
    try:
        from .bm25 import IndexBuilder
    except ImportError:
        from bm25 import IndexBuilder

    builder = IndexBuilder()
    bm25_index = builder.build([("doc1", "北京是中国的首都"), ("doc2", "上海是沿海城市")])
    vector_store = VectorStore.build([("doc1", "北京是中国的首都"), ("doc2", "上海是沿海城市")])
    searcher = Searcher(bm25_index, vector_store)

    assert searcher.bm25_search("") == []
    assert searcher.vector_search("") == []

    bm25_hits = searcher.bm25_search("北京")
    vector_hits = searcher.vector_search("北京")
    print("bm25  :", bm25_hits)
    print("vector:", vector_hits)
    assert bm25_hits and bm25_hits[0][0] == "doc1"
    assert vector_hits and vector_hits[0][0] == "doc1"
    assert [s for _, s in bm25_hits] == sorted([s for _, s in bm25_hits], reverse=True)
    assert [s for _, s in vector_hits] == sorted([s for _, s in vector_hits], reverse=True)
    print("Searcher self-tests passed")
