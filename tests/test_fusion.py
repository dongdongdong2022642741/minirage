"""Tests for index/fusion.py: RRF and min-max weighted sum.

Run with:  python -m unittest discover tests -t .
"""

import unittest

from index.fusion import RRF_K, fuse

# BM25 scores (~10) and cosine scores (~0.5) are deliberately on
# different scales, like the real recall paths.
BM25_HITS = [("a", 11.8723), ("b", 5.2), ("c", 1.1)]
VEC_HITS = [("b", 0.73), ("c", 0.45), ("d", 0.40)]


def ids(pairs):
    return [doc_id for doc_id, _ in pairs]


class RrfTests(unittest.TestCase):
    def test_returns_at_most_k_deduplicated_descending(self):
        result = fuse(BM25_HITS, VEC_HITS, k=3, method="rrf")
        self.assertEqual(ids(result), ["b", "c", "a"])
        scores = [score for _, score in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(result), 3)

    def test_doc_in_both_lists_appears_once(self):
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="rrf")
        self.assertEqual(len(result), len(set(ids(result))))
        self.assertEqual(set(ids(result)), {"a", "b", "c", "d"})

    def test_ranking_is_invariant_to_score_magnitude(self):
        scaled = fuse([(d, s * 100) for d, s in BM25_HITS], VEC_HITS, k=5, method="rrf")
        baseline = fuse(BM25_HITS, VEC_HITS, k=5, method="rrf")
        self.assertEqual(scaled, baseline)

    def test_doc_in_both_lists_outranks_doc_in_one_list(self):
        # b ranks 1st in vector and 2nd in bm25; a ranks 1st in bm25 only.
        # 1/61 + 1/62 > 1/61, so the two-sided hit wins.
        result = fuse(BM25_HITS, VEC_HITS[:1], k=2, method="rrf")
        self.assertEqual(ids(result), ["b", "a"])
        self.assertGreater(result[0][1], result[1][1])


class WeightedSumTests(unittest.TestCase):
    def test_scores_stay_in_unit_interval(self):
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="weighted", alpha=0.5)
        for _, score in result:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_alpha_one_equals_pure_bm25_ordering(self):
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="weighted", alpha=1.0)
        bm25_ids = [doc for doc, _ in BM25_HITS]
        self.assertEqual([d for d, _ in result if d in bm25_ids], bm25_ids)

    def test_alpha_zero_equals_pure_vector_ordering(self):
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="weighted", alpha=0.0)
        vec_ids = [doc for doc, _ in VEC_HITS]
        self.assertEqual([d for d, _ in result if d in vec_ids], vec_ids)

    def test_alpha_out_of_range_is_clamped(self):
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="weighted", alpha=2.0)
        for _, score in result:
            self.assertLessEqual(score, 1.0)

    def test_known_minmax_weights(self):
        # bm25 norm: a->1.0, b->(5.2-1.1)/10.7723=0.3806, c->0.0
        # vec  norm: b->1.0, c->(0.45-0.40)/0.33=0.1515, d->0.0
        result = fuse(BM25_HITS, VEC_HITS, k=5, method="weighted", alpha=0.5)
        by_id = dict(result)
        self.assertAlmostEqual(by_id["a"], 0.5, places=4)
        self.assertAlmostEqual(by_id["b"], 0.6903, places=4)
        self.assertAlmostEqual(by_id["c"], 0.0758, places=4)
        self.assertEqual(by_id["d"], 0.0)


class EdgeCaseTests(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual(fuse([], [], k=5), [])

    def test_one_side_empty_keeps_other_ordering(self):
        result = fuse(BM25_HITS, [], k=2, method="rrf")
        self.assertEqual(ids(result), ["a", "b"])
        expected = [1.0 / (RRF_K + 1), 1.0 / (RRF_K + 2)]
        self.assertEqual([s for _, s in result], expected)

    def test_vector_only(self):
        result = fuse([], VEC_HITS, k=2, method="rrf")
        self.assertEqual(ids(result), ["b", "c"])

    def test_k_zero_returns_empty_for_both_methods(self):
        self.assertEqual(fuse(BM25_HITS, VEC_HITS, k=0, method="rrf"), [])
        self.assertEqual(fuse(BM25_HITS, VEC_HITS, k=0, method="weighted"), [])

    def test_k_larger_than_candidates_returns_fewer_without_error(self):
        result = fuse(BM25_HITS, VEC_HITS, k=100, method="rrf")
        self.assertEqual(len(result), 4)

    def test_single_doc_path_does_not_divide_by_zero(self):
        one = [("a", 5.0)]
        result = fuse(one, one, k=2, method="weighted")
        self.assertEqual(ids(result), ["a"])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            fuse(BM25_HITS, VEC_HITS, k=5, method="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
