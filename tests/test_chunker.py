"""Tests for the chunking module (W2). Run:  python -m unittest discover tests"""

import unittest

from chunking import ChunkResult, chunk_fixed, chunk_structured
from docparser import Document


def make_doc(text: str, doc_id: str = "doc-test") -> Document:
    return Document(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        path=f"{doc_id}.md",
        text=text,
        suffix="md",
        encoding="utf-8",
        char_count=len(text),
        line_count=text.count("\n") + 1,
        size_bytes=len(text.encode("utf-8")),
        modified_at="2026-08-07T00:00:00+00:00",
    )


def top_level(result: ChunkResult):
    return [c for c in result.chunks if c.parent_id is None]


class FixedChunkTests(unittest.TestCase):
    def test_empty_doc_returns_empty_result(self):
        result = chunk_fixed(make_doc(""), size=10)
        self.assertEqual(result.chunks, [])

    def test_exact_multiple_splits_evenly(self):
        doc = make_doc("abcdefghijklmnopqrst")  # 20 chars
        result = chunk_fixed(doc, size=10)
        self.assertEqual([c.text for c in result.chunks], ["abcdefghij", "klmnopqrst"])
        self.assertEqual(result.chunks[0].start_char, 0)
        self.assertEqual(result.chunks[1].start_char, 10)
        self.assertEqual(result.chunks[1].end_char, 20)

    def test_ragged_tail_kept(self):
        doc = make_doc("a" * 23)
        result = chunk_fixed(doc, size=10)
        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(result.chunks[-1].text, "a" * 3)
        self.assertEqual(result.chunks[-1].end_char, 23)

    def test_overlap_zero_reassembles_original(self):
        text = "今天天气很好。明天会下雨。后天放晴。数据是干净的。"
        doc = make_doc(text)
        result = chunk_fixed(doc, size=6)
        self.assertEqual("".join(c.text for c in result.chunks), text)

    def test_overlap_keeps_boundary_fragment_complete(self):
        # 答案"北京是中国的首都"骑在 5 字符边界上。
        text = "中国的首都是北京是中国的首都吗？"
        doc = make_doc(text)
        result = chunk_fixed(doc, size=6, overlap=2)
        texts = [c.text for c in result.chunks]
        self.assertTrue(any("北京是" in t for t in texts))

    def test_overlap_ge_size_raises(self):
        doc = make_doc("x" * 50)
        with self.assertRaises(ValueError):
            chunk_fixed(doc, size=10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_fixed(doc, size=10, overlap=15)

    def test_size_zero_raises(self):
        doc = make_doc("x")
        with self.assertRaises(ValueError):
            chunk_fixed(doc, size=0)

    def test_single_char_size_and_doc(self):
        # 单字符文档：size=1 产生 1 块
        one = chunk_fixed(make_doc("a"), size=1)
        self.assertEqual(len(one.chunks), 1)
        self.assertEqual(one.chunks[0].text, "a")
        self.assertEqual((one.chunks[0].start_char, one.chunks[0].end_char), (0, 1))
        # 长文档 size=1：每字符一块，偏移连续，拼回 == 原文
        text = "中文abc"
        many = chunk_fixed(make_doc(text), size=1)
        self.assertEqual(len(many.chunks), len(text))
        for i, c in enumerate(many.chunks):
            self.assertEqual(c.text, text[i])
            self.assertEqual((c.start_char, c.end_char), (i, i + 1))
        self.assertEqual("".join(c.text for c in many.chunks), text)

    def test_overlap_increases_total_text_amount(self):
        # 步长 = size - overlap，overlap 越大步长越小、块数越多、总文本量越大。
        doc = make_doc("x" * 100)
        plain = chunk_fixed(doc, size=30)
        with_ov = chunk_fixed(doc, size=30, overlap=10)
        plain_total = sum(len(c.text) for c in plain.chunks)
        ov_total = sum(len(c.text) for c in with_ov.chunks)
        self.assertGreater(len(with_ov.chunks), len(plain.chunks))
        self.assertGreater(ov_total, plain_total)

    def test_chunk_ids_unique_and_stable(self):
        doc = make_doc("z" * 25)
        ids = [c.chunk_id for c in chunk_fixed(doc, size=10).chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))


class StructuredChunkTests(unittest.TestCase):
    def test_empty_doc_returns_empty_result(self):
        result = chunk_structured(make_doc(""))
        self.assertEqual(result.chunks, [])

    def test_no_headings_becomes_single_chunk(self):
        result = chunk_structured(make_doc("只有正文，没有标题。"))
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].heading_path, ())
        self.assertIsNone(result.chunks[0].parent_id)

    def test_intro_before_first_heading_is_separate_chunk(self):
        text = "这是引言，概述全文。\n\n# 第一章\n正文内容。"
        result = chunk_structured(make_doc(text))
        chunks = {c.heading_path: c for c in result.chunks}
        self.assertIn((), chunks)
        self.assertIn(("第一章",), chunks)
        self.assertTrue(chunks[()].text.startswith("这是引言"))

    def test_heading_path_nesting(self):
        text = "# 一\n甲\n## 1.1\n乙\n### 1.1.1\n丙\n## 1.2\n丁"
        result = chunk_structured(make_doc(text))
        paths = {c.heading_path for c in top_level(result)}
        self.assertEqual(
            paths,
            {("一",), ("一", "1.1"), ("一", "1.1", "1.1.1"), ("一", "1.2")},
        )

    def test_level_jump_does_not_fabricate_headings(self):
        # # 一 直接跳到 ### 1.1.1（跳过 ##）。
        text = "# 一\n甲\n### 1.1.1\n乙"
        result = chunk_structured(make_doc(text))
        paths = {c.heading_path for c in top_level(result)}
        self.assertEqual(paths, {("一",), ("一", "1.1.1")})

    def test_sibling_section_after_deeper_heading(self):
        text = "# 一\n甲\n## 1.1\n乙\n# 二\n丙"
        result = chunk_structured(make_doc(text))
        paths = {c.heading_path for c in top_level(result)}
        self.assertEqual(paths, {("一",), ("一", "1.1"), ("二",)})

    def test_long_section_triggers_parent_child_split(self):
        text = "# 超长章节\n" + "内容" * 1000  # 2000 chars > max_section_size
        result = chunk_structured(
            make_doc(text), max_section_size=500, child_size=200, child_overlap=20
        )
        stats = result.stats()
        self.assertEqual(stats["parents"], 1)
        self.assertGreater(stats["children"], 1)
        parent = top_level(result)[0]
        for child in result.chunks:
            if child.parent_id is not None:
                self.assertEqual(child.parent_id, parent.chunk_id)
                self.assertEqual(child.heading_path, parent.heading_path)

    def test_children_offsets_inside_parent(self):
        text = "# 一\n" + "内容" * 500
        result = chunk_structured(
            make_doc(text), max_section_size=100, child_size=60, child_overlap=5
        )
        parent = top_level(result)[0]
        for child in result.chunks:
            if child.parent_id is not None:
                self.assertGreaterEqual(child.start_char, parent.start_char)
                self.assertLessEqual(child.end_char, parent.end_char)

    def test_children_concat_equals_parent_text(self):
        text = "# 一\n" + "内容" * 500
        result = chunk_structured(
            make_doc(text), max_section_size=100, child_size=60, child_overlap=0
        )
        parent = top_level(result)[0]
        children = sorted(
            (c for c in result.chunks if c.parent_id is not None),
            key=lambda c: c.start_char,
        )
        self.assertEqual("".join(c.text for c in children), parent.text)

    def test_top_level_reassembles_original(self):
        text = "引言部分。\n\n# 一\n甲\n## 1.1\n乙\n# 二\n丙"
        result = chunk_structured(make_doc(text))
        top = sorted(top_level(result), key=lambda c: c.start_char)
        self.assertEqual("".join(c.text for c in top), text)

    def test_short_section_has_no_children(self):
        result = chunk_structured(make_doc("# 短\n内容"), max_section_size=500)
        self.assertEqual(result.stats()["children"], 0)

    def test_heading_in_body_text_not_detected(self):
        # ATX 标题必须在行首；正文里的 # 不算。
        text = "# 标题\n正文 # 不是标题\n| # | 表头 |"
        result = chunk_structured(make_doc(text))
        paths = {c.heading_path for c in top_level(result)}
        self.assertEqual(paths, {("标题",)})

    def test_bad_params_raise(self):
        doc = make_doc("x")
        with self.assertRaises(ValueError):
            chunk_structured(doc, max_section_size=0)
        with self.assertRaises(ValueError):
            chunk_structured(doc, child_size=0)
        with self.assertRaises(ValueError):
            chunk_structured(doc, child_overlap=-1)
        with self.assertRaises(ValueError):
            chunk_structured(doc, child_size=10, child_overlap=10)


class ChunkResultStatsTests(unittest.TestCase):
    def test_stats_counts(self):
        doc = make_doc("# 长\n" + "内容" * 1000)
        result = chunk_structured(doc, max_section_size=100, child_size=50, child_overlap=5)
        stats = result.stats()
        self.assertEqual(stats["parents"] + stats["children"], stats["total"])
        self.assertEqual(stats["total"], len(result.chunks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
