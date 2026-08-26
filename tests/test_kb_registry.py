"""多知识库注册表测试：挂载、校验与实例隔离。"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb_registry import DEFAULT_KB_ID, KBRegistry  # noqa: E402


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name) / "kbs"
        legacy = Path(self.tmp.name) / "legacy_kb"
        legacy.mkdir(parents=True, exist_ok=True)
        self.reg = KBRegistry(base, legacy)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_kb_auto_registered(self):
        entries = {e["kb_id"] for e in self.reg.list()}
        self.assertIn(DEFAULT_KB_ID, entries)

    def test_create_lists_and_duplicate_rejected(self):
        self.reg.create("school", "校园库", "测试")
        ids = [e["kb_id"] for e in self.reg.list()]
        self.assertIn("school", ids)
        with self.assertRaises(ValueError):
            self.reg.create("school", "again")

    def test_invalid_kb_id_rejected(self):
        for bad in ("A_upper", "1start", "a", "含中文", ""):
            with self.assertRaises(ValueError):
                self.reg.create(bad, "x")

    def test_unknown_kb_get_raises(self):
        with self.assertRaises(KeyError):
            self.reg.get("nope")

    def test_instances_are_fully_isolated(self):
        def unit_embed(texts):
            from index.embeddings import EMBEDDING_DIM
            return [[1.0] * EMBEDDING_DIM for _ in texts]

        self.reg.create("alpha", "A 库")
        self.reg.create("beta", "B 库")
        ka = self.reg.get("alpha")
        kbeta = self.reg.get("beta")

        ka.add_uploads([("note.md", "# Alpha\n只有甲库有的秘密口令".encode("utf-8"))])
        kbeta.add_uploads([("note.md", "# Beta\n乙库的完全不同内容".encode("utf-8"))])
        ka.rebuild(force=True, embed_fn=unit_embed)
        kbeta.rebuild(force=True, embed_fn=unit_embed)

        docs_a = {d["name"]: d["id"] for d in ka.list_docs()}
        docs_b = {d["name"]: d["id"] for d in kbeta.list_docs()}
        # 同名文件在不同库是不同文档身份
        self.assertNotEqual(docs_a["note.md"], docs_b["note.md"])
        # 各自目录只装得下自己的文档
        chunks_a = {c.document_id for c in ka._load_chunks()}
        chunks_b = {c.document_id for c in kbeta._load_chunks()}
        self.assertEqual(chunks_a, {docs_a["note.md"]})
        self.assertEqual(chunks_b, {docs_b["note.md"]})

        # 核心不变量：在乙库里检索甲库专属词，证据中永远不可能出现甲库文档
        hits_b, _f = kbeta._retrieve("只有甲库有的秘密口令")
        returned_ids = {chunk.document_id for chunk, _s in hits_b}
        self.assertNotIn(docs_a["note.md"], returned_ids)
        # 反向同理
        hits_a, _f2 = ka._retrieve("乙库的完全不同内容")
        self.assertNotIn(docs_b["note.md"],
                         {c.document_id for c, _s in hits_a})


if __name__ == "__main__":
    unittest.main(verbosity=2)
