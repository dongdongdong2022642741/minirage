"""Stage 5 ops tests: retry, audit log, build jobs, single-flight, opstats."""

import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from app.audit import AuditLog
from app.kb import KnowledgeBase
from index.embeddings import EMBEDDING_DIM


def unit_embed(texts):
    return [[1.0] * EMBEDDING_DIM for _ in texts]


def _fake_urlopen_response(texts):
    payload = {"data": [{"embedding": [1.0] * EMBEDDING_DIM} for _ in texts]}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json
            return json.dumps(payload).encode("utf-8")

    # urlopen 的返回值被当作上下文管理器，json.load(response) 需要可迭代行？
    # 实际实现是 json.load(response)，因此提供 read 之外还要兼容 json.load。
    import io
    return io.StringIO(__import__("json").dumps(payload))


class RetryEmbeddingTests(unittest.TestCase):
    def test_transient_errors_are_retried_then_succeed(self):
        attempts = {"n": 0}

        def flaky_urlopen(request, timeout=60):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise urllib.error.URLError("transient network blip")
            return _fake_urlopen_response(["任意文本"])

        with mock.patch("index.embeddings.urlopen", side_effect=flaky_urlopen), \
                mock.patch("index.embeddings.time.sleep") as fake_sleep:
            from index.embeddings import _embed_batch
            vectors = _embed_batch("fake-key", ["任意文本"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(fake_sleep.call_count, 2)  # 指数退避发生两次

    def test_http_400_is_not_retried(self):
        attempts = {"n": 0}

        def bad_request_urlopen(request, timeout=60):
            attempts["n"] += 1
            raise urllib.error.HTTPError(
                url="x", code=400, msg="bad request",
                hdrs=None, fp=__import__("io").BytesIO(b"{}"))

        with mock.patch("index.embeddings.urlopen", side_effect=bad_request_urlopen), \
                mock.patch("index.embeddings.time.sleep") as fake_sleep:
            from index.embeddings import _embed_batch
            with self.assertRaises(RuntimeError):
                _embed_batch("fake-key", ["任意文本"])

        self.assertEqual(attempts["n"], 1)
        fake_sleep.assert_not_called()


class AuditLogTests(unittest.TestCase):
    def test_round_trip_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = AuditLog(Path(tmp) / "audit.jsonl")
            log.record("upload", name="a.md", ok=True)
            log.record("ask", user_id="u1", latency_ms=12.5)
            events = log.tail(10)
            self.assertEqual([e["event"] for e in events], ["upload", "ask"])
            self.assertNotIn("query", events[1])

    def test_corrupted_tail_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            path.write_text('{"event":"ok"}\n{"broken\n', encoding="utf-8")
            events = AuditLog(path).tail(10)
            self.assertEqual(len(events), 1)


class BuildJobAndSingleFlightTests(unittest.TestCase):
    def test_second_build_hits_cache_and_job_records_zero_embedded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("index.embeddings.embed_texts", unit_embed):
                kb = KnowledgeBase(Path(tmp))
                kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
                kb.rebuild(force=True)
                jobs = kb.catalog.recent_build_jobs(5)
                self.assertEqual(jobs[0]["state"], "done")
                self.assertGreaterEqual(jobs[0]["embedded"], 1)

                kb.rebuild(force=True)
                jobs = kb.catalog.recent_build_jobs(5)
                self.assertEqual(jobs[0]["embedded"], 0)
                self.assertGreaterEqual(jobs[0]["reused"], 1)

    def test_start_rebuild_is_single_flight(self):
        release = threading.Event()

        def blocking_embed(texts):
            release.wait(timeout=5)
            return unit_embed(texts)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("index.embeddings.embed_texts", blocking_embed):
                kb = KnowledgeBase(Path(tmp))
                kb.add_uploads([("handbook.md", b"# Handbook\nStable")])

                job = kb.start_rebuild()
                self.assertEqual(job["state"], "queued")
                with self.assertRaises(RuntimeError):
                    kb.start_rebuild()  # 单飞锁生效

                release.set()
                for _ in range(200):  # 最多等 2 秒
                    if kb.current_build() is None or \
                            kb.current_build().get("state") == "done":
                        break
                    threading.Event().wait(0.01)

                self.assertIsNotNone(kb.current_build())
                self.assertEqual(kb.current_build()["state"], "done")


class OpstatsTests(unittest.TestCase):
    def test_ask_records_latency_audit_without_query_text_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("index.embeddings.embed_texts", unit_embed):
                kb = KnowledgeBase(Path(tmp))
                kb.add_uploads([("handbook.md", "# Handbook\nStable 内容".encode("utf-8"))])
                kb.rebuild(force=True)

            with mock.patch("app.kb.ask_deepseek",
                            return_value="[1] Stable 内容相关回答。"):
                result = kb.ask("Stable 有什么内容", user_id=DEFAULT_USER_ID_KB())

            self.assertIn("latency_ms", result)
            stats = kb.opstats()
            self.assertGreaterEqual(stats["asks"], 1)
            self.assertIsNotNone(stats["p95_ms"])
            self.assertGreaterEqual(stats["llm_calls"], 1)
            self.assertGreater(stats["estimated_cost_yuan"], 0)

            events = kb.audit.tail(20)
            ask_events = [e for e in events if e["event"] == "ask"]
            self.assertTrue(ask_events)
            latest = ask_events[-1]
            self.assertNotIn("query", latest)
            raw_lines = (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("有什么内容", raw_lines)  # 问题原文绝不落盘


def DEFAULT_USER_ID_KB():
    from app.document_catalog import DEFAULT_USER_ID
    return DEFAULT_USER_ID


class RelevanceGateTests(unittest.TestCase):
    def test_gate_blocks_below_threshold_and_passes_above(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("index.embeddings.embed_texts", unit_embed):
                kb = KnowledgeBase(Path(tmp))
                kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
                kb.rebuild(force=True, embed_fn=unit_embed)
            doc_id = kb.list_docs()[0]["id"]

            import os
            # unit_embed 全同向量 -> 相似度恒为 1.0
            with mock.patch.dict(os.environ, {"KB_RELEVANCE_GATE": "0.9"}):
                hits, _f = kb._retrieve("Stable", allowed_ids={doc_id})
                self.assertTrue(hits)
            with mock.patch.dict(os.environ, {"KB_RELEVANCE_GATE": "1.5"}):
                hits, _f = kb._retrieve("Stable", allowed_ids={doc_id})
                self.assertEqual(hits, [])
            # 默认关闭
            hits, _f = kb._retrieve("Stable", allowed_ids={doc_id})
            self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
