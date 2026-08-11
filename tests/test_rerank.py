"""Tests for index/rerank.py: re-score fused top-N with raw-path scores.

Run with:  python -m unittest discover tests -t .
"""

import unittest

from index.fusion import fuse
from index.rerank import rerank

# Fused candidate set: RRF orders B > C > A (two-sided hits win).
BM25_HITS = [("A", 11.8), ("B", 5.0), ("C", 0.001)]
VEC_HITS = [("C", 0.9), ("B", 0.8)]


def ids(pairs):
    return [doc_id for doc_id, _ in pairs]


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.fused = fuse(BM25_HITS, VEC_HITS, k=3, method="rrf")

    def test_doc_set_is_unchanged(self):
        result = rerank(self.fused, BM25_HITS, VEC_HITS)
        self.assertEqual(set(ids(result)), set(ids(self.fused)))
        self.assertEqual(len(result), len(self.fused))

    def test_output_sorted_descending(self):
        result = rerank(self.fused, BM25_HITS, VEC_HITS)
        scores = [score for _, score in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_in_unit_interval(self):
        result = rerank(self.fused, BM25_HITS, VEC_HITS)
        for _, score in result:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_alpha_one_follows_pure_bm25_normalized_order(self):
        result = rerank(self.fused, BM25_HITS, VEC_HITS, alpha=1.0)
        self.assertEqual(ids(result), ["A", "B", "C"])
        by_id = dict(result)
        self.assertAlmostEqual(by_id["A"], 1.0, places=6)
        self.assertAlmostEqual(by_id["B"], 0.4237, places=4)
        self.assertEqual(by_id["C"], 0.0)

    def test_alpha_zero_follows_pure_vector_normalized_order(self):
        result = rerank(self.fused, BM25_HITS, VEC_HITS, alpha=0.0)
        self.assertEqual(ids(result), ["C", "B", "A"])

    def test_alpha_out_of_range_is_clamped(self):
        self.assertEqual(
            rerank(self.fused, BM25_HITS, VEC_HITS, alpha=2.0),
            rerank(self.fused, BM25_HITS, VEC_HITS, alpha=1.0),
        )


class RerankCorrectsRrfTests(unittest.TestCase):
    """RRF flattens the BM25 score cliff; rerank picks it back up."""

    def setUp(self):
        # A crushes B in BM25 (11.8 vs 0.001) but is absent from vector.
        self.bm25 = [("A", 11.8), ("B", 0.001)]
        self.vec = [("B", 0.9)]

    def test_rrf_mis_orders_due_to_score_loss(self):
        fused = fuse(self.bm25, self.vec, k=2, method="rrf")
        self.assertEqual(ids(fused), ["B", "A"])

    def test_rerank_restores_bm25_favorite(self):
        fused = fuse(self.bm25, self.vec, k=2, method="rrf")
        result = rerank(fused, self.bm25, self.vec, alpha=0.6)
        self.assertEqual(ids(result), ["A", "B"])
        self.assertEqual(set(ids(result)), set(ids(fused)))
        self.assertAlmostEqual(result[0][1], 0.6, places=6)
        self.assertAlmostEqual(result[1][1], 0.4, places=6)


class RerankEdgeCaseTests(unittest.TestCase):
    def test_empty_fused_returns_empty(self):
        self.assertEqual(rerank([], BM25_HITS, VEC_HITS), [])

    def test_doc_in_fused_but_in_neither_path_stays_with_zero(self):
        result = rerank([("X", 1.0)], [("A", 1.0)], [])
        self.assertEqual(ids(result), ["X"])
        self.assertEqual(result[0][1], 0.0)

    def test_doc_only_in_one_path_gets_zero_from_the_other(self):
        bm25 = [("A", 11.8), ("B", 9.0), ("C", 5.0)]
        vec = [("A", 0.9)]
        fused = fuse(bm25, vec, k=3, method="rrf")
        self.assertEqual(ids(fused), ["A", "B", "C"])
        result = rerank(fused, bm25, vec, alpha=0.5)
        by_id = dict(result)
        # bm25 norm: A=1.0, B=(9-5)/6.8=0.5882, C=0.0 ; vec norm: A=1.0
        # A = 0.5*1.0 + 0.5*1.0 = 1.0 ; B = 0.5*0.5882 + 0.5*0 = 0.2941
        self.assertEqual(ids(result), ["A", "B", "C"])
        self.assertAlmostEqual(by_id["A"], 1.0, places=6)
        self.assertAlmostEqual(by_id["B"], 0.2941, places=4)
        self.assertEqual(by_id["C"], 0.0)

    def test_no_duplicates_introduced(self):
        fused = fuse(BM25_HITS, VEC_HITS, k=3, method="rrf")
        result = rerank(fused, BM25_HITS, VEC_HITS)
        self.assertEqual(len(ids(result)), len(set(ids(result))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
