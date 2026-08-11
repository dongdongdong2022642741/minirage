"""Tests for the index module: BM25, VectorStore, Searcher.

Run with:  python -m unittest discover tests -t .

Red line: this file must pass WITHOUT SILICONFLOW_API_KEY. VectorStore is
always constructed with a fake embed_fn or a hand-built matrix, never via
VectorStore.build (which would call the real embedding API).
"""

import unittest

import numpy as np

from index import IndexBuilder, VectorStore, bm25_search
from index.searcher import Searcher


def build_two_docs():
    return IndexBuilder().build([("doc1", "北京是首都"), ("doc2", "上海是城市")])


class Bm25IndexTests(unittest.TestCase):
    def test_postings_maps_word_to_doc_tf(self):
        idx = IndexBuilder().build([("doc1", "北京是首都")])
        self.assertEqual(idx.postings["北京"], {"doc1": 1})

    def test_doc_len_counts_tokens_not_chars(self):
        idx = IndexBuilder().build([("doc1", "北京是首都")])
        self.assertEqual(idx.doc_len, {"doc1": 4})

    def test_N_and_avgdl(self):
        idx = build_two_docs()
        self.assertEqual(idx.N, 2)
        self.assertEqual(idx.avgdl, 4.0)

    def test_idf_precomputed_at_build_time(self):
        idx = build_two_docs()
        self.assertIn("北京", idx.idf)
        self.assertAlmostEqual(idx.idf["北京"], 0.693147, places=5)


class Bm25SearchTests(unittest.TestCase):
    def test_empty_index_returns_empty(self):
        idx = IndexBuilder().build([])
        self.assertEqual(bm25_search(idx, "任何词"), [])

    def test_absent_term_returns_empty(self):
        self.assertEqual(bm25_search(build_two_docs(), "不存在词xyz"), [])

    def test_empty_query_returns_empty(self):
        idx = build_two_docs()
        self.assertEqual(bm25_search(idx, ""), [])
        self.assertEqual(bm25_search(idx, "   "), [])

    def test_exact_keyword_ranks_relevant_doc_first(self):
        result = bm25_search(build_two_docs(), "北京首都")
        self.assertEqual(result[0][0], "doc1")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][1], 1.386294, places=5)

    def test_reverse_keyword_ranks_other_doc_first(self):
        result = bm25_search(build_two_docs(), "上海城市")
        self.assertEqual(result[0][0], "doc2")
        self.assertEqual(len(result), 1)

    def test_scores_sorted_descending(self):
        idx = IndexBuilder().build(
            [("d1", "北京是首都"), ("d2", "北京是北京"), ("d3", "上海是城市")]
        )
        result = bm25_search(idx, "北京首都")
        scores = [score for _, score in result]
        self.assertEqual(scores, sorted(scores, reverse=True))


def fake_embed(texts):
    lookup = {
        "q": [1, 0, 0],
        "doc1": [1, 0, 0],
        "doc2": [0, 1, 0],
        "北京": [1, 0, 0],
    }
    return [lookup[t] for t in texts]


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        # 3 维假矩阵：行数与 doc_ids 一一对应，列数与 fake_embed 输出对齐。
        self.matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.store = VectorStore(["doc1", "doc2"], self.matrix, embed_fn=fake_embed)

    def test_same_direction_vector_ranks_first(self):
        result = self.store.search("q")
        self.assertEqual(result[0][0], "doc1")
        self.assertAlmostEqual(result[0][1], 1.0, places=6)

    def test_score_is_cosine_not_dot_product(self):
        # doc2 同方向但模长更大 [2,0,0]；余弦只看方向，应与 doc1 并列 1.0。
        matrix = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
        store = VectorStore(["doc1", "doc2"], matrix, embed_fn=fake_embed)
        result = store.search("q")
        self.assertEqual(len(result), 2)
        for doc_id, score in result:
            self.assertAlmostEqual(score, 1.0, places=6)

    def test_orthogonal_vector_scores_zero(self):
        result = self.store.search("q")
        by_id = dict(result)
        self.assertAlmostEqual(by_id["doc2"], 0.0, places=6)

    def test_empty_store_returns_empty(self):
        empty = VectorStore([], np.zeros((0, 3), dtype=np.float32), embed_fn=fake_embed)
        self.assertEqual(empty.search("q"), [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("   "), [])


class SearcherTests(unittest.TestCase):
    def setUp(self):
        self.bm25_index = build_two_docs()
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.vector_store = VectorStore(["doc1", "doc2"], matrix, embed_fn=fake_embed)
        self.searcher = Searcher(self.bm25_index, self.vector_store)

    def test_both_recall_paths_rank_doc1_first(self):
        bm25_hits = self.searcher.bm25_search("北京")
        vector_hits = self.searcher.vector_search("北京")
        self.assertEqual(bm25_hits[0][0], "doc1")
        self.assertEqual(vector_hits[0][0], "doc1")

    def test_both_recall_paths_return_descending_scores(self):
        for hits in (self.searcher.bm25_search("北京"), self.searcher.vector_search("北京")):
            scores = [score for _, score in hits]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_both_recall_paths_return_str_float_pairs(self):
        for hits in (self.searcher.bm25_search("北京"), self.searcher.vector_search("北京")):
            for doc_id, score in hits:
                self.assertIsInstance(doc_id, str)
                self.assertIsInstance(score, float)

    def test_empty_query_returns_empty_for_both(self):
        self.assertEqual(self.searcher.bm25_search(""), [])
        self.assertEqual(self.searcher.vector_search(""), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
