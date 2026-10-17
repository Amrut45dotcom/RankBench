"""Run from the repo root:  python -m unittest level1.test_validator -v
or from inside level1/:     python -m unittest test_validator -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator import validate, clean_citations, clean_for_display  # noqa: E402

N = 5


class TestValidate(unittest.TestCase):
    def test_plain_valid(self):
        r = validate("Paris is the capital of France. [P1]", N)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["cited"], [1])

    def test_fullwidth_brackets_accepted(self):
        r = validate("Canada has three levels of government【P1】【P2】【P3】.", N)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["cited"], [1, 2, 3])

    def test_no_final_period(self):
        r = validate("Olivia Taylor Dudley plays Alice on The Magicians【P1】", N)
        self.assertEqual(r["status"], "ok")

    def test_non_breaking_hyphen_and_cite_after_period(self):
        r = validate("The number is (631) 454\u20118059. [P1]", N)
        self.assertEqual(r["status"], "ok")

    def test_two_sentences_cite_after_period_style(self):
        r = validate("A is true. [P1] B is also true. [P2]", N)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["cited"], [1, 2])

    def test_second_sentence_uncited(self):
        r = validate("A is true. [P1] B is also true.", N)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("uncited_sentences:1", r["reasons"])

    def test_no_citation_at_all(self):
        r = validate("Paris is the capital of France.", N)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("no_citations", r["reasons"])

    def test_unknown_passage_number(self):
        r = validate("Foo is bar [P9].", N)
        self.assertEqual(r["status"], "invalid")
        self.assertTrue(any(x.startswith("unknown_citation") for x in r["reasons"]))

    def test_zero_is_unknown(self):
        r = validate("Foo is bar [P0].", N)
        self.assertEqual(r["status"], "invalid")

    def test_exact_refusal(self):
        r = validate("INSUFFICIENT_CONTEXT", N)
        self.assertEqual(r["status"], "refused")

    def test_refusal_with_whitespace(self):
        r = validate("  INSUFFICIENT_CONTEXT.\n", N)
        self.assertEqual(r["status"], "refused")

    def test_refusal_mixed_with_answer_is_invalid(self):
        r = validate("INSUFFICIENT_CONTEXT, but it might be X [P1].", N)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("refusal_mixed_with_answer", r["reasons"])

    def test_empty_and_none(self):
        self.assertEqual(validate("", N)["status"], "invalid")
        self.assertEqual(validate(None, N)["status"], "invalid")

    def test_multi_cite_in_one_bracket(self):
        r = validate("Both passages agree on this [P1, P3].", N)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["cited"], [1, 3])

    def test_decimal_number_not_split(self):
        r = validate("The rate is 3.5 percent [P2].", N)
        self.assertEqual(r["status"], "ok")

    def test_citation_only_is_not_an_answer(self):
        r = validate("[P1]", N)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("no_content", r["reasons"])


class TestSentenceSplitting(unittest.TestCase):
    def test_lga_jfk_regression(self):
        raw = ("The driving distance between LaGuardia Airport (LGA) and John F. Kennedy "
               "International Airport (JFK) is 12 miles (19 kilometers)【P1】")
        self.assertEqual(validate(raw, N)["status"], "ok")

    def test_us_abbreviation(self):
        self.assertEqual(validate("It was passed by the U.S. Congress in 1950 [P2].", N)["status"], "ok")

    def test_dr_abbreviation(self):
        self.assertEqual(validate("Dr. Smith wrote the paper [P1].", N)["status"], "ok")

    def test_eg_abbreviation(self):
        self.assertEqual(validate("Hand tools, e.g. a trowel, are used [P1].", N)["status"], "ok")

    def test_initial_then_real_uncited_sentence(self):
        r = validate("John F. Kennedy Airport is big [P1]. It is old.", N)
        self.assertEqual(r["status"], "invalid")
        self.assertIn("uncited_sentences:1", r["reasons"])


class TestCleanCitations(unittest.TestCase):
    def test_brackets_fixed(self):
        self.assertEqual(clean_citations("x【P1】"), "x[P1]")

    def test_other_text_untouched(self):
        self.assertEqual(clean_citations("a\u2011b [P1]"), "a\u2011b [P1]")


class TestCleanForDisplay(unittest.TestCase):
    def test_non_breaking_hyphen_becomes_plain(self):
        self.assertEqual(clean_for_display("(631) 454\u20118059. [P1]"), "(631) 454-8059. [P1]")

    def test_unicode_hyphen_becomes_plain(self):
        self.assertEqual(clean_for_display("a\u2010b"), "a-b")

    def test_brackets_still_fixed(self):
        self.assertEqual(clean_for_display("x【P1】"), "x[P1]")

    def test_en_and_em_dashes_kept(self):
        self.assertEqual(clean_for_display("2010\u20132015 \u2014 done"), "2010\u20132015 \u2014 done")


if __name__ == "__main__":
    unittest.main()