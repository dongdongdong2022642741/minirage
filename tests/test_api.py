import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import main as api
from app.kb import KnowledgeBase


class RestoreEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(api, "kb", KnowledgeBase(Path(self.temp.name)))
        self.kb = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)
        self.client = TestClient(api.app)

    def test_restore_endpoint_maps_errors(self):
        missing = self.client.post("/api/documents/deadbeef/restore")
        self.assertEqual(missing.status_code, 404)

        self.kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
        doc_id = self.kb.list_docs()[0]["id"]
        not_deleted = self.client.post(f"/api/documents/{doc_id}/restore")
        self.assertEqual(not_deleted.status_code, 409)
        self.assertIn("detail", not_deleted.json())

    def test_restore_endpoint_success_shape(self):
        self.kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
        doc_id = self.kb.list_docs()[0]["id"]
        self.assertTrue(self.kb.delete_doc(doc_id))

        response = self.client.post(f"/api/documents/{doc_id}/restore")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], doc_id)
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["version"], 1)
        self.assertNotIn("stored_path", body)
        self.assertNotIn("source_uri", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
