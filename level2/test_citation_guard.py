import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level1"))
from citation_guard import blocked_citations

PASSAGES = [{"pid": "a"}, {"pid": "b"}, {"pid": "c"}]


def allow(*ok):
    return lambda p: p in ok


class TestCitationGuard(unittest.TestCase):
    def test_no_citations(self):
        self.assertEqual(blocked_citations([], PASSAGES, allow("a")), [])

    def test_none_citations(self):
        self.assertEqual(blocked_citations(None, PASSAGES, allow("a")), [])

    def test_all_allowed(self):
        self.assertEqual(blocked_citations([1, 3], PASSAGES, allow("a", "c")), [])

    def test_one_forbidden(self):
        self.assertEqual(blocked_citations([1, 2], PASSAGES, allow("a")), ["b"])

    def test_forced_forbidden_id_is_blocked(self):
        # the attack case: a forbidden passage ends up in the cited list
        forced = [{"pid": "a"}, {"pid": "FORBIDDEN"}, {"pid": "c"}]
        self.assertEqual(blocked_citations([2], forced, allow("a", "c")), ["FORBIDDEN"])

    def test_order_preserved_and_unique(self):
        self.assertEqual(blocked_citations([3, 1, 3], PASSAGES, allow()), ["c", "a"])

    def test_out_of_range_ignored(self):
        self.assertEqual(blocked_citations([0, 4, -1], PASSAGES, allow()), [])

    def test_string_markers(self):
        self.assertEqual(blocked_citations(["P2"], PASSAGES, allow("a")), ["b"])

    def test_lookup_error_fails_closed(self):
        def boom(p):
            raise KeyError(p)
        self.assertEqual(blocked_citations([1], PASSAGES, boom), ["a"])


if __name__ == "__main__":
    unittest.main()