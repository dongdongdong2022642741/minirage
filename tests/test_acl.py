"""Stage 4 ACL tests: C1 whitelist / E1 default-deny / F2 filtered reporting."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.document_catalog import DEFAULT_USER_ID, DocumentCatalog
from app.kb import KnowledgeBase, RAW_K
from app.main import app
from fastapi.testclient import TestClient
from index.embeddings import EMBEDDING_DIM


def unit_embed(texts):
    return [[1.0] * EMBEDDING_DIM for _ in texts]


def zero_embed(texts):
    """All-zero vectors make the vector channel silent -> pure BM25 scenarios."""
    return [[0.0] * EMBEDDING_DIM for _ in texts]


class CatalogAclTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = DocumentCatalog(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_user_allow_set_is_empty(self):
        self.assertEqual(self.catalog.allowed_document_ids("ghost"), set())

    def test_ingest_auto_grants_admin(self):
        doc = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.assertIn(doc["document_id"],
                      self.catalog.allowed_document_ids(DEFAULT_USER_ID))

    def test_grant_revoke_update_allow_set_with_validation(self):
        doc = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.catalog.ensure_user("u1", "普通用户")

        self.assertNotIn(doc["document_id"], self.catalog.allowed_document_ids("u1"))
        self.catalog.grant("u1", doc["document_id"])
        self.assertIn(doc["document_id"], self.catalog.allowed_document_ids("u1"))
        self.assertTrue(self.catalog.revoke("u1", doc["document_id"]))
        self.assertNotIn(doc["document_id"], self.catalog.allowed_document_ids("u1"))

        with self.assertRaises(ValueError):
            self.catalog.grant("no-such-user", doc["document_id"])
        with self.assertRaises(ValueError):
            self.catalog.grant("u1", "no-such-document")

    def test_admin_grants_resync_on_startup(self):
        doc = self.catalog.ingest("policy.md", b"# Policy\nValid")
        self.assertTrue(self.catalog.revoke(DEFAULT_USER_ID, doc["document_id"]))

        fresh = DocumentCatalog(self.catalog.root)

        self.assertIn(doc["document_id"],
                      fresh.allowed_document_ids(DEFAULT_USER_ID))


class KnowledgeBaseAclTests(unittest.TestCase):
    def _build_kb(self, tmp, files, embed_fn=unit_embed):
        with mock.patch("index.embeddings.embed_texts", embed_fn):
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads(files)
            kb.rebuild(force=True)
        return kb

    def test_unauthorized_user_never_sees_denied_document_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp, [
                ("public.md", "# Public\n入职流程说明".encode("utf-8")),
                ("secret.md", "# Secret\n薪酬保密信息".encode("utf-8")),
            ])
            by_name = {d["name"]: d["id"] for d in kb.list_docs()}
            kb.catalog.ensure_user("u1", "普通员工")
            kb.catalog.grant("u1", by_name["public.md"])

            with mock.patch("app.kb.ask_deepseek",
                            return_value="[1] 入职流程如下。"):
                result = kb.ask("薪酬保密信息", user_id="u1")

            doc_ids = {e["document_id"] for e in result["evidence"]}
            self.assertNotIn(by_name["secret.md"], doc_ids)
            self.assertLessEqual(doc_ids, {by_name["public.md"]})
            self.assertEqual(result["acl"]["user_id"], "u1")
            self.assertGreaterEqual(result["acl"]["filtered"], 1)

    def test_overfetch_rescues_allowed_doc_from_denied_crowding(self):
        files = [("public.md", "# Public\n报销批准流程 需要主管签字".encode("utf-8"))]
        for i in range(12):
            files.append((
                f"denied{i}.md",
                f"# D{i}\n报销批准流程 批准 批准 流程 流程 主管 签字 签字".encode("utf-8"),
            ))

        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp, files, embed_fn=zero_embed)
            public_id = {d["name"]: d["id"] for d in kb.list_docs()}["public.md"]
            kb.catalog.ensure_user("u1", "普通员工")
            kb.catalog.grant("u1", public_id)

            with mock.patch("app.kb.ask_deepseek", return_value="[1] 流程如下。"):
                result = kb.ask("报销批准流程 主管 签字", user_id="u1")

            self.assertTrue(result["evidence"])
            for e in result["evidence"]:
                self.assertEqual(e["document_id"], public_id)
            # 12 个无权文档全部被过滤（旧 RAW_K=10 会把 public 挤出候选池）
            self.assertGreaterEqual(result["acl"]["filtered"], RAW_K)

    def test_empty_evidence_short_circuits_llm(self):
        """零证据必须确定性拒答，绝不调用 LLM（权限矩阵实测缺陷的回归测试）。"""
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp, [
                ("public.md", "# Public\n报销批准流程 需要主管签字".encode("utf-8")),
            ], embed_fn=zero_embed)
            kb.catalog.ensure_user("u1", "员工")
            kb.catalog.grant("u1", kb.list_docs()[0]["id"])

            with mock.patch("app.kb.ask_deepseek") as fake_chat:
                result = kb.ask("薪酬保密信息有哪些", user_id="u1")

            fake_chat.assert_not_called()
            self.assertTrue(result["refusal"])
            self.assertEqual(result["retried"], False)

    def test_deleted_document_hidden_even_if_still_granted(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp, [
                ("public.md", "# Public\n公开内容".encode("utf-8")),
                ("secret.md", "# Secret\n薪酬保密信息".encode("utf-8")),
            ])
            secret_id = {d["name"]: d["id"] for d in kb.list_docs()}["secret.md"]
            kb.catalog.ensure_user("u1", "员工")
            kb.catalog.grant("u1", secret_id)
            kb.delete_doc(secret_id)  # 软删除后未重建索引——内存索引仍是旧的

            with mock.patch("app.kb.ask_deepseek", return_value="[1] 内容。"):
                result = kb.ask("薪酬保密信息", user_id="u1")

            self.assertEqual(result["evidence"], [])
            self.assertGreaterEqual(result["acl"]["filtered"], 1)


class ApiAclTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        patcher = mock.patch(
            "app.main.kb", KnowledgeBase(Path(self.temp.name)))
        self.kb = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)
        self.client = TestClient(app)

    def test_ask_without_user_id_is_rejected(self):
        response = self.client.post("/api/ask", json={"query": "任何问题"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_user_defaults_to_deny(self):
        with mock.patch("index.embeddings.embed_texts", unit_embed):
            self.kb.add_uploads([("handbook.md", b"# Handbook\nStable")])
            self.kb.rebuild(force=True)
        with mock.patch("app.kb.ask_deepseek", return_value="资料不足"):
            response = self.client.post(
                "/api/ask", json={"query": "Stable", "user_id": "ghost"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["evidence"], [])
        self.assertEqual(body["acl"]["user_id"], "ghost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
