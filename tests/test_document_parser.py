"""Tests for the docparser module. Run with:  python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path

from docparser import Document, load_documents, parse_file

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


class ParseFileTests(unittest.TestCase):
    def test_parses_markdown(self):
        doc = parse_file(DOCS_DIR / "sample_01.md")
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.filename, "sample_01.md")
        self.assertEqual(doc.suffix, "md")
        self.assertIn("RAG", doc.text)
        self.assertGreater(doc.char_count, 0)
        self.assertTrue(doc.doc_id)

    def test_parses_txt(self):
        doc = parse_file(DOCS_DIR / "sample_02.txt")
        self.assertEqual(doc.filename, "sample_02.txt")
        self.assertEqual(doc.suffix, "txt")

    def test_parses_gbk_file_via_fallback(self):
        doc = parse_file(DOCS_DIR / "sample_03.txt")
        self.assertIn("GBK 编码示例", doc.text)
        self.assertEqual(doc.encoding, "gb18030")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_file(DOCS_DIR / "nope.md")

    def test_unsupported_suffix_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "data.bin"
            bad.write_bytes(b"\x00\x01")
            with self.assertRaises(ValueError):
                parse_file(bad)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.md"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_file(empty)


class LoadDocumentsTests(unittest.TestCase):
    def test_loads_md_and_txt(self):
        result = load_documents(DOCS_DIR)
        docs = {d.filename: d for d in result.documents}
        self.assertIn("sample_01.md", docs)
        self.assertIn("sample_02.txt", docs)
        self.assertIn("sample_03.txt", docs)

    def test_empty_file_recorded_as_issue(self):
        result = load_documents(DOCS_DIR)
        reasons = {i.reason for i in result.issues}
        self.assertIn("empty_file", reasons)

    def test_undecodable_file_recorded_as_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.txt").write_bytes(b"\x81")
            result = load_documents(root)
            self.assertEqual(result.documents, [])
            self.assertEqual(len(result.issues), 1)
            self.assertEqual(result.issues[0].reason, "decode_error")

    def test_doc_id_is_stable_across_calls(self):
        first = load_documents(DOCS_DIR).documents
        second = load_documents(DOCS_DIR).documents
        first_ids = {d.filename: d.doc_id for d in first}
        second_ids = {d.filename: d.doc_id for d in second}
        self.assertEqual(first_ids, second_ids)

    def test_doc_id_differs_between_documents(self):
        result = load_documents(DOCS_DIR)
        ids = [d.doc_id for d in result.documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(Path(tempfile.gettempdir()) / "definitely-not-here")

    def test_ignores_non_doc_files_and_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello", encoding="utf-8")
            (root / "image.png").write_bytes(b"\x89PNG")
            (root / "sub").mkdir()
            (root / "sub" / "inner.md").write_text("inner", encoding="utf-8")
            result = load_documents(root)
            names = [d.filename for d in result.documents]
            self.assertEqual(names, ["notes.md"])


class DocumentShapeTests(unittest.TestCase):
    def test_required_fields_present(self):
        doc = parse_file(DOCS_DIR / "sample_01.md")
        self.assertTrue(hasattr(doc, "doc_id"))
        self.assertTrue(hasattr(doc, "filename"))
        self.assertTrue(hasattr(doc, "text"))

    def test_frozen_immutability(self):
        doc = parse_file(DOCS_DIR / "sample_01.md")
        with self.assertRaises(Exception):
            doc.text = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main(verbosity=2)
