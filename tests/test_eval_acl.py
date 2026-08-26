"""Offline unit tests for eval_acl.run_acl_matrix (no real API)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb import KnowledgeBase  # noqa: E402
from eval_acl import run_acl_matrix  # noqa: E402
from index.embeddings import EMBEDDING_DIM  # noqa: E402


def unit_embed(texts):
    return [[1.0] * EMBEDDING_DIM for _ in texts]


def zero_embed(texts):
    """全零向量使向量检索通道静默：用例退化为纯 BM25 场景，行为确定。"""
    return [[0.0] * EMBEDDING_DIM for _ in texts]


def canned_chat(prompt: str, **_kwargs) -> str:
    """有证据时给可引用回答（含关键词），无证据时拒答——模拟真实模型行为。"""
    if "\n[" in prompt:
        return "[1] 资料确认：报销批准流程需要主管签字后方可执行。"
    return "资料不足"


class RunAclMatrixTests(unittest.TestCase):
    def _build_kb(self, tmp, files=None, embed_fn=unit_embed):
        if files is None:
            files = [
                ("public.md", "# Public\n报销批准流程 需要主管签字".encode("utf-8")),
                ("secret.md", "# Secret\n薪酬保密信息".encode("utf-8")),
            ]
        with mock.patch("index.embeddings.embed_texts", embed_fn):
            kb = KnowledgeBase(Path(tmp))
            kb.add_uploads(files)
            kb.rebuild(force=True)
        return kb

    def test_matrix_passes_with_zero_leak(self):
        cfg = {
            "users": {
                "u1": {"display_name": "员工", "grant": ["public.md"]},
            },
            "cases": [
                {"user_id": "u1", "query": "报销流程",
                 "expected": "answer", "keywords": ["签字"]},
                {"user_id": "u1", "query": "薪酬保密信息",
                 "expected": "refuse"},
                {"user_id": "ghost", "query": "任何内容",
                 "expected": "refuse"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.kb.ask_deepseek", side_effect=canned_chat):
                report = run_acl_matrix(self._build_kb(tmp, embed_fn=zero_embed), cfg,
                                        include_state_probe=False)

        s = report["summary"]
        self.assertEqual(s["passed"], s["total"])
        self.assertEqual(s["leak_count"], 0)
        for row in report["rows"]:
            self.assertEqual(row["leak_docs"], [])

    def test_state_probe_covers_delete_and_restore(self):
        cfg = {
            "users": {
                "u1": {"grant": ["public.md"]},
            },
            "cases": [],
            "state_probe": {
                "user_id": "u1",
                "doc_name": "public.md",
                "query": "报销批准流程",
                "keywords": ["签字"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp)
            scenarios = []
            orig_ask = KnowledgeBase.ask

            def spy_ask(self, query, user_id=None):
                res = orig_ask(self, query, user_id=user_id)
                scenarios.append((user_id, res["refusal"]))
                return res

            with mock.patch("app.kb.ask_deepseek", side_effect=canned_chat), \
                    mock.patch.object(KnowledgeBase, "ask", spy_ask):
                report = run_acl_matrix(kb, cfg, include_state_probe=True)

            names = [r["scenario"] for r in report["rows"]]
            self.assertEqual(names,
                             ["granted_visible", "deleted_hidden", "restored_visible"])
            # 删除期间必须拒答（中间行 refusal=True），恢复后重新可答
            refusals = [r["refusal"] for r in report["rows"]]
            self.assertEqual(refusals, [False, True, False])
            self.assertEqual(report["summary"]["leak_count"], 0)
            # 探针结束后文档应已恢复
            restored = kb.catalog.get_by_name("public.md")
            self.assertEqual(restored["status"], "ready")

    def test_unknown_grant_doc_fails_fast(self):
        cfg = {"users": {"u1": {"grant": ["不存在.md"]}}, "cases": []}
        with tempfile.TemporaryDirectory() as tmp:
            kb = self._build_kb(tmp)
            with self.assertRaises(SystemExit):
                run_acl_matrix(kb, cfg, include_state_probe=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
