import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.document_catalog import (
    DocumentCatalog,
    DocumentNotFoundError,
    DocumentStateError,
)
from app.kb import KnowledgeBase
from index.embeddings import EMBEDDING_DIM


class DocumentCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = DocumentCatalog(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_same_content_is_idempotent(self):
        first = self.catalog.ingest("policy.md", b"# Policy\nVersion one")
        second = self.catalog.ingest("policy.md", b"# Policy\nVersion one")

        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(second["version_number"], 1)
        self.assertEqual(len(self.catalog.versions(first["document_id"])), 1)

    def test_changed_content_creates_new_current_version(self):
        first = self.catalog.ingest("policy.md", b"# Policy\nVersion one")
        second = self.catalog.ingest("policy.md", b"# Policy\nVersion two")

        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(second["version_number"], 2)
        versions = self.catalog.versions(first["document_id"])
        self.assertEqual([item["version_number"] for item in versions], [2, 1])
        self.assertNotEqual(versions[0]["content_hash"], versions[1]["content_hash"])

    def test_failed_replacement_does_not_promote_version(self):
        first = self.catalog.ingest("policy.md", b"# Policy\nValid")
        with self.assertRaises(ValueError):
            self.catalog.ingest("policy.md", b"\x81")

        current = self.catalog.get(first["document_id"])
        self.assertEqual(current["current_version_id"], first["current_version_id"])
        self.assertEqual(current["status"], "ready")
        self.assertTrue(current["last_error"])
        self.assertEqual(self.catalog.versions(first["document_id"])[0]["status"], "failed")

    def test_soft_delete_excludes_document_and_keeps_versions(self):
        document = self.catalog.ingest("policy.md", b"# Policy\nValid")

        self.assertTrue(self.catalog.delete(document["document_id"]))
        self.assertEqual(self.catalog.list_documents(), [])
        self.assertEqual(len(self.catalog.versions(document["document_id"])), 1)
        self.assertFalse(self.catalog.delete(document["document_id"]))

    def test_reupload_after_delete_restores_same_document(self):
        document = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.catalog.delete(document["document_id"])

        restored = self.catalog.ingest("policy.md", b"# Policy\nValid")

        self.assertEqual(restored["document_id"], document["document_id"])
        self.assertEqual(restored["version_number"], 1)
        self.assertEqual(restored["status"], "ready")
        self.assertIsNone(restored["deleted_at"])

    def test_restore_deleted_document_succeeds(self):
        document = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.assertTrue(self.catalog.delete(document["document_id"]))

        restored = self.catalog.restore(document["document_id"])

        self.assertEqual(restored["status"], "ready")
        self.assertIsNone(restored["deleted_at"])
        self.assertEqual(restored["current_version_id"], document["current_version_id"])
        self.assertEqual(len(self.catalog.versions(document["document_id"])), 1)

    def test_restore_twice_second_call_conflicts(self):
        document = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.assertTrue(self.catalog.delete(document["document_id"]))
        self.catalog.restore(document["document_id"])

        with self.assertRaises(DocumentStateError):
            self.catalog.restore(document["document_id"])

    def test_restore_non_deleted_document_conflicts(self):
        document = self.catalog.ingest("policy.md", b"# Policy\nValid")

        with self.assertRaises(DocumentStateError):
            self.catalog.restore(document["document_id"])
        self.assertEqual(self.catalog.get(document["document_id"])["status"], "ready")

    def test_restore_unknown_document_not_found(self):
        with self.assertRaises(DocumentNotFoundError):
            self.catalog.restore("nonexistent-document-id")

    def test_restore_document_without_ready_version_conflicts(self):
        with self.assertRaises(ValueError):
            self.catalog.ingest("broken.md", b"\x81")
        record = self.catalog.get_by_name("broken.md")
        self.assertTrue(self.catalog.delete(record["document_id"]))

        with self.assertRaises(DocumentStateError):
            self.catalog.restore(record["document_id"])
        self.assertEqual(
            self.catalog.get(record["document_id"])["status"], "deleted"
        )


class KnowledgeBaseLifecycleTests(unittest.TestCase):
    def test_repeated_upload_updates_one_logical_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("handbook.md", b"# Handbook\nFirst")])
            kb.add_uploads([("handbook.md", b"# Handbook\nSecond")])

            docs = kb.list_docs()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["name"], "handbook.md")
            self.assertEqual(docs[0]["version"], 2)

    def test_html_upload_enters_the_same_versioned_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("manual.html", b"<h1>Manual</h1><p>Power off first.</p>")])

            docs = kb.list_docs()
            chunks = kb._load_chunks()
            self.assertEqual(docs[0]["version"], 1)
            self.assertEqual(docs[0]["status"], "ready")
            self.assertEqual(chunks[0].label, "manual.html \u00b7 Manual")

    def test_chunks_use_version_ids_and_preserve_document_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("handbook.md", b"# Handbook\nStable evidence")])

            doc = kb.list_docs()[0]
            chunks = kb._load_chunks()
            self.assertTrue(chunks)
            self.assertEqual(chunks[0].document_id, doc["id"])
            self.assertIn(chunks[0].version_id, chunks[0].chunk_id)
            self.assertEqual(chunks[0].label, "handbook.md · Handbook")

    def test_public_version_history_hides_storage_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("handbook.md", b"# Handbook\nStable evidence")])
            document_id = kb.list_docs()[0]["id"]

            version = kb.document_versions(document_id)[0]
            self.assertNotIn("stored_path", version)
            self.assertNotIn("source_uri", version)
            self.assertEqual(version["version_number"], 1)

    def test_restore_returns_to_built_state_without_reembedding(self):
        calls = []

        def fake_embed(texts):
            calls.append(list(texts))
            return [[1.0] * EMBEDDING_DIM for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
            doc_id = kb.list_docs()[0]["id"]

            kb.rebuild(force=True, embed_fn=fake_embed)
            self.assertEqual(len(calls), 1)
            self.assertTrue(kb.status()["built"])

            self.assertTrue(kb.delete_doc(doc_id))
            self.assertFalse(kb.status()["built"])

            kb.restore_document(doc_id)
            self.assertTrue(kb.status()["built"])

            kb.rebuild(force=True, embed_fn=fake_embed)
            self.assertEqual(len(calls), 1)

    def test_rebuild_failure_preserves_existing_index_and_keeps_serving(self):
        def good_embed(texts):
            return [[1.0] * EMBEDDING_DIM for _ in texts]

        def broken_embed(texts):
            raise RuntimeError("API timeout simulation")

        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads([("handbook.md", b"# Handbook\nInitial content")])
            
            # 1. 成功构建一代
            kb.rebuild(force=True, embed_fn=good_embed)
            self.assertTrue(kb.status()["built"])
            self.assertIsNotNone(kb._bm25)
            self.assertIsNotNone(kb._vector)
            
            # 2. 上传新文档，造成指纹变化
            kb.add_uploads([("new_doc.md", b"# New\nSomething new")])
            self.assertFalse(kb.status()["built"])
            
            # 3. 模拟第二代构建时抛出异常 (A1 策略)
            with self.assertRaises(RuntimeError):
                kb.rebuild(force=True, embed_fn=broken_embed)
            
            # 4. 关键验证：构建虽然失败，但原内存中的第一代索引依然完好无损，且没有残留脏目录
            self.assertIsNotNone(kb._bm25)
            self.assertIsNotNone(kb._vector)
            hits = kb._retrieve("Handbook")
            self.assertTrue(len(hits) > 0)
            self.assertEqual(hits[0][0].label, "handbook.md · Handbook")

    def test_generations_pruned_to_two_most_recent(self):
        def fake_embed(texts):
            return [[1.0] * EMBEDDING_DIM for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(Path(tmp))
            # 构造 3 个版本触发 3 次代际构建
            for i in range(1, 4):
                kb.add_uploads([("handbook.md", f"# Handbook\nVersion {i}".encode("utf-8"))])
                kb.rebuild(force=True, embed_fn=fake_embed)
            
            # B1 策略验证：generations 目录下仅保留最新的 2 代
            gen_dirs = list(kb.generations_dir.glob("gen_*"))
            self.assertEqual(len(gen_dirs), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
