"""Tests for the index module: BM25, VectorStore, Searcher.

Run with:  python -m unittest discover tests -t .

Red line: this file must pass WITHOUT SILICONFLOW_API_KEY. VectorStore is
always constructed with a fake embed_fn or a hand-built matrix, never via
VectorStore.build (which would call the real embedding API).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from index import IndexBuilder, VectorStore, bm25_search
from index.embeddings import (
    EMBEDDING_DIM,
    build_with_cache,
    embedding_cache_key,
    load_cached_vector,
    save_cached_vector,
)
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


class EmbeddingCacheKeyTests(unittest.TestCase):
    def test_identical_model_and_text_share_key(self):
        self.assertEqual(
            embedding_cache_key("BAAI/bge-m3", "员工请假需要审批"),
            embedding_cache_key("BAAI/bge-m3", "员工请假需要审批"),
        )

    def test_text_change_changes_key(self):
        self.assertNotEqual(
            embedding_cache_key("m", "试用期90天"), embedding_cache_key("m", "试用期60天")
        )

    def test_model_change_changes_key(self):
        self.assertNotEqual(
            embedding_cache_key("bge-m3", "同一文本"),
            embedding_cache_key("other-model", "同一文本"),
        )

    def test_separator_prevents_concatenation_ambiguity(self):
        self.assertNotEqual(embedding_cache_key("ab", "c"), embedding_cache_key("a", "bc"))


class SingleVectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_round_trip(self):
        key = "0123456789abcdef" * 4
        vector = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        save_cached_vector(self.root, key, vector)

        loaded = load_cached_vector(self.root, key)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (EMBEDDING_DIM,))
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_allclose(loaded, vector, rtol=1e-6)

    def test_miss_returns_none(self):
        loaded = load_cached_vector(self.root, "non_existent_key_1234")
        self.assertIsNone(loaded)

    def test_garbage_bytes_treated_as_miss_and_auto_healed(self):
        key = "aabbccddeeff0011" * 4
        # 手动写入非法的垃圾字节
        path = Path(self.root) / key[:2] / f"{key}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a valid npy file content")

        loaded = load_cached_vector(self.root, key)
        self.assertIsNone(loaded)
        # A2策略断言：损坏文件已被自动删除
        self.assertFalse(path.exists())

    def test_wrong_dimension_treated_as_miss_and_auto_healed(self):
        key = "1122334455667788" * 4
        wrong_dim_vec = np.zeros(512, dtype=np.float32)  # 512 != 1024
        path = Path(self.root) / key[:2] / f"{key}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, wrong_dim_vec, allow_pickle=False)

        loaded = load_cached_vector(self.root, key)
        self.assertIsNone(loaded)
        # A2策略断言：维度不符的文件已被自动删除
        self.assertFalse(path.exists())

    def test_tmp_files_do_not_interfere(self):
        key = "9988776655443322" * 4
        # 模拟写入中断留下的临时文件
        path = Path(self.root) / key[:2] / f".{key}.12345.tmp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial content")

        loaded = load_cached_vector(self.root, key)
        self.assertIsNone(loaded)


class BuildWithCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cold_start_embeds_all_and_populates_cache(self):
        calls = []

        def mock_embed(texts):
            calls.append(list(texts))
            return [np.ones(EMBEDDING_DIM, dtype=np.float32) * (i + 1) for i, _ in enumerate(texts)]

        docs = [("c1", "文本一"), ("c2", "文本二")]
        store, stats = build_with_cache(docs, self.root, embed_fn=mock_embed)

        self.assertEqual(stats, {"total": 2, "reused": 0, "embedded": 2})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["文本一", "文本二"])
        self.assertEqual(store.doc_ids, ["c1", "c2"])
        self.assertEqual(store.matrix.shape, (2, EMBEDDING_DIM))

    def test_warm_cache_reuses_all_with_zero_api_calls(self):
        calls = []

        def mock_embed(texts):
            calls.append(list(texts))
            return [np.ones(EMBEDDING_DIM, dtype=np.float32) for _ in texts]

        docs = [("c1", "文本一"), ("c2", "文本二")]
        # 第一次构建（冷启动）
        build_with_cache(docs, self.root, embed_fn=mock_embed)
        self.assertEqual(len(calls), 1)

        # 第二次构建（全命中）
        store, stats = build_with_cache(docs, self.root, embed_fn=mock_embed)
        self.assertEqual(stats, {"total": 2, "reused": 2, "embedded": 0})
        self.assertEqual(len(calls), 1)  # 未增加 API 调用
        self.assertEqual(store.doc_ids, ["c1", "c2"])

    def test_single_chunk_edit_reuses_unmodified_chunks(self):
        calls = []

        def mock_embed(texts):
            calls.append(list(texts))
            return [np.ones(EMBEDDING_DIM, dtype=np.float32) * len(t) for t in texts]

        # 初始版本：3 个 chunks
        v1_docs = [("v1#1", "第一章未修改"), ("v1#2", "第二章旧内容"), ("v1#3", "第三章未修改")]
        build_with_cache(v1_docs, self.root, embed_fn=mock_embed)
        self.assertEqual(calls, [["第一章未修改", "第二章旧内容", "第三章未修改"]])

        # 更新版本：修改第二章，chunk_id 全变
        v2_docs = [("v2#1", "第一章未修改"), ("v2#2", "第二章新内容！"), ("v2#3", "第三章未修改")]
        store, stats = build_with_cache(v2_docs, self.root, embed_fn=mock_embed)

        self.assertEqual(stats, {"total": 3, "reused": 2, "embedded": 1})
        self.assertEqual(len(calls), 2)
        # 关键验证：第二次只把真正修改的文本传给 embed_fn
        self.assertEqual(calls[1], ["第二章新内容！"])
        self.assertEqual(store.doc_ids, ["v2#1", "v2#2", "v2#3"])

    def test_batch_duplicate_texts_deduplicated_before_api_call(self):
        calls = []

        def mock_embed(texts):
            calls.append(list(texts))
            return [np.ones(EMBEDDING_DIM, dtype=np.float32) for _ in texts]

        # 传入包含重复文本的 chunks
        docs = [("c1", "重复免责声明"), ("c2", "正文内容"), ("c3", "重复免责声明")]
        store, stats = build_with_cache(docs, self.root, embed_fn=mock_embed)

        self.assertEqual(stats, {"total": 3, "reused": 1, "embedded": 2})
        self.assertEqual(len(calls), 1)
        # 验证只发送了 2 条唯一文本
        self.assertEqual(calls[0], ["重复免责声明", "正文内容"])
        self.assertEqual(store.matrix.shape, (3, EMBEDDING_DIM))


if __name__ == "__main__":
    unittest.main(verbosity=2)
