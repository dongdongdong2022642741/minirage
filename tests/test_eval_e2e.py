"""Tests for the pure, API-free logic in eval_e2e.py.

Run with:  python -m unittest discover tests -t .
"""

import unittest

from eval_e2e import TOP_FOR_ANSWER, citation_ok, is_meta, sample_queries, split_statements


class CitationOkTests(unittest.TestCase):
    def test_zero_citations_is_invalid(self):
        # Empty-set == empty-set used to make "all valid" trivially true.
        self.assertFalse(citation_ok([]))

    def test_single_valid_citation_is_valid(self):
        self.assertTrue(citation_ok([1]))

    def test_all_in_range_is_valid(self):
        self.assertTrue(citation_ok([1, 5, 3]))

    def test_out_of_range_is_invalid(self):
        self.assertFalse(citation_ok([TOP_FOR_ANSWER + 1]))
        self.assertFalse(citation_ok([0]))
        self.assertFalse(citation_ok([-1]))

    def test_mixed_valid_and_invalid_is_invalid(self):
        self.assertFalse(citation_ok([1, TOP_FOR_ANSWER + 1]))


class SplitStatementsTests(unittest.TestCase):
    def test_splits_on_chinese_and_ascii_punctuation(self):
        parts = split_statements("甲乙丙丁。戊己庚辛！壬癸子丑？寅卯辰巳；午未申酉")
        self.assertEqual(parts, ["甲乙丙丁", "戊己庚辛", "壬癸子丑", "寅卯辰巳", "午未申酉"])

    def test_drops_parts_shorter_than_four_chars(self):
        self.assertEqual(split_statements("是。甲乙丙丁"), ["甲乙丙丁"])

    def test_empty_answer_yields_no_statements(self):
        self.assertEqual(split_statements(""), [])


class IsMetaTests(unittest.TestCase):
    def test_known_markers(self):
        self.assertTrue(is_meta("资料不足"))
        self.assertTrue(is_meta("根据资料，无法确定答案"))
        self.assertTrue(is_meta("没有提供相关信息"))

    def test_normal_statement_is_not_meta(self):
        self.assertFalse(is_meta("中国的首都是北京"))


class SampleQueriesTests(unittest.TestCase):
    QUERIES = [(str(i), f"q{i}", set()) for i in range(100)]

    def test_deterministic_stride_sampling(self):
        self.assertEqual(sample_queries(self.QUERIES, 10), self.QUERIES[::10][:10])

    def test_size_larger_than_input_clamps(self):
        result = sample_queries(self.QUERIES, 1000)
        self.assertEqual(len(result), 100)

    def test_size_one_keeps_first_query(self):
        self.assertEqual(sample_queries(self.QUERIES, 1), [self.QUERIES[0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
