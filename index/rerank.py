"""Rerank: re-order the fused top-N using the scores fusion discarded.

RRF fusion picks candidates by rank only and throws scores away. Rerank
takes the fused top-N (the ONLY allowed candidate set) and re-scores each
doc with the min-max normalized raw scores from both recall paths:

    score(doc) = alpha * bm25_norm.get(doc, 0.0)
                 + (1 - alpha) * vec_norm.get(doc, 0.0)

A doc missing from one path gets 0 for that path. The doc set is
immutable: rerank reorders, never adds or removes.

Contract:
    rerank(fused_hits, bm25_hits, vector_hits, alpha=0.5)
        -> list[(doc_id, score)] sorted descending, scores in [0, 1],
        same doc set as fused_hits; [] if fused_hits is empty.
"""

from __future__ import annotations

try:
    from .fusion import _minmax_normalize
except ImportError:
    from fusion import _minmax_normalize


def rerank(
    fused_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    vector_hits: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    if not fused_hits:
        return []
    alpha = min(max(alpha, 0.0), 1.0)
    bm25_norm = _minmax_normalize(bm25_hits)
    vec_norm = _minmax_normalize(vector_hits)
    scored = [
        (
            doc_id,
            alpha * bm25_norm.get(doc_id, 0.0)
            + (1.0 - alpha) * vec_norm.get(doc_id, 0.0),
        )
        for doc_id, _ in fused_hits
    ]
    # stable sort: fused (RRF) order survives as the tiebreaker
    return sorted(scored, key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    bm25 = [("A", 11.8), ("B", 0.001)]
    vec = [("B", 0.9)]
    fused = [("B", 0.032522), ("A", 0.016393)]  # RRF order: B first

    result = rerank(fused, bm25, vec, alpha=0.6)
    print("rerank   :", result)
    assert [doc for doc, _ in result] == ["A", "B"]
    assert set(d for d, _ in result) == set(d for d, _ in fused)
    assert rerank([], bm25, vec) == []
    zeroed = rerank(fused, [], [])  # no raw scores anywhere: all zeros, order kept
    assert [doc for doc, _ in zeroed] == ["B", "A"]
    assert all(score == 0.0 for _, score in zeroed)
    for _, score in result:
        assert 0.0 <= score <= 1.0
    print("rerank self-tests passed")
