import tempfile
import unittest
from pathlib import Path

from app.document_catalog import DocumentCatalog
from app.kb import KnowledgeBase


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
