"""Hybrid fusion: merge BM25 and vector hits into one ordered list.

Both recall paths return list[(doc_id: str, score: float)] with
incomparable score scales (BM25 ~10, cosine ~0.5), so fusing them by
weighted sum of raw scores is meaningless. Two algorithms, same interface:

    rrf:      Reciprocal Rank Fusion. Ranks only, scores ignored,
              immune to scale differences.
              score(doc) = sum over lists containing doc of 1 / (RRF_K + rank)

    weighted: min-max normalize each list's scores to [0, 1], then
              alpha * bm25_norm + (1 - alpha) * vector_norm.
              alpha = 1 is pure BM25 ordering, alpha = 0 is pure vector.

Contract:
    fuse(bm25_hits, vector_hits, k, method="rrf", alpha=0.5)
        -> list[(doc_id, score)] sorted by score descending,
        doc_ids deduplicated, at most k entries; fewer if candidates
        are short. Never raises on empty inputs or k = 0 (returns []).
"""

from __future__ import annotations

RRF_K = 60  # standard RRF constant; distinct from the return-count k
DEFAULT_ALPHA = 0.5


def _rrf_score(hits: list[tuple[str, float]]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for rank, (doc_id, _score) in enumerate(hits, start=1):
        acc[doc_id] = acc.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return acc


def _minmax_normalize(hits: list[tuple[str, float]]) -> dict[str, float]:
    """Map scores to [0, 1]. A doc absent from this list contributes 0."""
    if not hits:
        return {}
    scores = [score for _, score in hits]
    low, high = min(scores), max(scores)
    if high == low:
        # no relative signal: every returned doc is equally "best"
        return {doc_id: 1.0 for doc_id, _ in hits}
    return {doc_id: (score - low) / (high - low) for doc_id, score in hits}


def fuse(
    bm25_hits: list[tuple[str, float]],
    vector_hits: list[tuple[str, float]],
    k: int,
    method: str = "rrf",
    alpha: float = DEFAULT_ALPHA,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists into one deduplicated top-k."""
    if k <= 0:
        return []
    if method == "rrf":
        acc = _rrf_score(bm25_hits)
        for doc_id, score in _rrf_score(vector_hits).items():
            acc[doc_id] = acc.get(doc_id, 0.0) + score
    elif method == "weighted":
        alpha = min(max(alpha, 0.0), 1.0)
        bm25_norm = _minmax_normalize(bm25_hits)
        vec_norm = _minmax_normalize(vector_hits)
        acc = {
            doc_id: alpha * bm25_norm.get(doc_id, 0.0)
            + (1.0 - alpha) * vec_norm.get(doc_id, 0.0)
            # sorted() keeps tie-breaking deterministic across runs
            # (set iteration order varies with the hash salt)
            for doc_id in sorted(set(bm25_norm) | set(vec_norm))
        }
    else:
        raise ValueError(f"unknown fusion method: {method}")
    ranked = sorted(acc.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    bm25 = [("a", 11.8723), ("b", 5.2), ("c", 1.1)]
    vec = [("b", 0.73), ("c", 0.45), ("d", 0.40)]

    rrf = fuse(bm25, vec, k=3, method="rrf")
    print("rrf      :", rrf)
    assert [doc for doc, _ in rrf] == ["b", "c", "a"]

    scaled = fuse([(d, s * 100) for d, s in bm25], vec, k=3, method="rrf")
    assert rrf == scaled, "RRF must ignore score magnitude"

    weighted = fuse(bm25, vec, k=4, method="weighted", alpha=0.5)
    print("weighted :", weighted)
    for _, score in weighted:
        assert 0.0 <= score <= 1.0

    assert fuse([], [], k=5) == []
    assert fuse(bm25, [], k=2) == [("a", 1.0 / (RRF_K + 1)), ("b", 1.0 / (RRF_K + 2))]
    assert fuse(bm25, vec, k=0) == []
    print("fusion self-tests passed")
