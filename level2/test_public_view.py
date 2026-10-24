import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "level1"))
from pipeline import REFUSAL_TEXT          # noqa: E402
from public_view import public_view        # noqa: E402

LEAKY = {"reasons": ["x"], "trace": {"final_ids": ["SECRET"]}, "attempt_log": [{"raw": "SECRET"}],
         "raw_answer": "SECRET", "blocked_citations": ["SECRET"], "retrieval_ms": 1.0,
         "generation_ms": 2.0, "attempts": 2, "cited": [], "passages": []}


def refusal(status):
    return {**LEAKY, "status": status, "refused": True, "answer": REFUSAL_TEXT}


class TestPublicView(unittest.TestCase):
    def test_all_refusals_identical(self):
        outs = [json.dumps(public_view(refusal(s)), sort_keys=True) for s in
                ("refused_no_passages", "refused_by_model",
                 "refused_validation_failed", "refused_citation_blocked")]
        self.assertEqual(len(set(outs)), 1)

    def test_absent_equals_forbidden(self):
        a = public_view(refusal("refused_no_passages"))
        f = public_view(refusal("refused_citation_blocked"))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(f, sort_keys=True))

    def test_no_internal_fields_in_refusal(self):
        out = json.dumps(public_view(refusal("refused_citation_blocked")))
        self.assertNotIn("SECRET", out)
        self.assertEqual(set(public_view(refusal("refused_by_model"))), {"answer", "refused", "citations"})

    def test_answered_exposes_only_cited_passages(self):
        res = {**LEAKY, "status": "answered", "refused": False, "answer": "Yes [P2].",
               "cited": [2], "passages": [{"pid": "a", "text": "ta"}, {"pid": "b", "text": "tb"}]}
        out = public_view(res)
        self.assertEqual(out["citations"], [{"n": 2, "pid": "b", "text": "tb"}])
        self.assertEqual(set(out), {"answer", "refused", "citations"})
        self.assertNotIn("SECRET", json.dumps(out))


if __name__ == "__main__":
    unittest.main()