import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "level1"))
from pipeline import Pipeline, REFUSAL_TEXT   # noqa: E402
from level2.audit import AuditLog              # noqa: E402


@dataclass(frozen=True)
class U:
    name: str
    is_admin: bool = False
    tenant: Optional[int] = None
    clearance: int = 0


PIDS = ["a", "b", "c"]
MASK = np.array([True, False, True])      # passage "b" is forbidden


def make_pipe(llm_text, audit_path):
    p = Pipeline.__new__(Pipeline)         # skip __init__: no index, no GPU, no Groq
    p.L1 = {"n_passages": 3}
    p.qrels = {}
    p.pid_list = PIDS
    p.mask_fn = lambda user: MASK
    p._masks = {}
    p._pid_idx = lambda pid: PIDS.index(pid)
    p.audit = AuditLog(audit_path)
    p.retrieve_traced = lambda q, n, user=None, strategy="none", **kw: (
        [{"pid": x, "text": f"text {x}"} for x in PIDS],
        {"rerank_input_ids": PIDS, "final_ids": PIDS})
    p.call_llm = lambda q, passages, reminder=None: llm_text
    return p


class TestAnswerGuard(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")

    def last_audit(self):
        with open(self.path, encoding="utf-8") as f:
            return json.loads(f.readlines()[-1])

    def test_forbidden_citation_is_blocked(self):
        p = make_pipe("The plan is confidential [P2].", self.path)
        res = p.answer("q", user=U("guest"), strategy="pre")
        self.assertEqual(res["status"], "refused_citation_blocked")
        self.assertTrue(res["refused"])
        self.assertEqual(res["answer"], REFUSAL_TEXT)
        self.assertEqual(res["cited"], [])
        self.assertEqual(res["raw_answer"], "")
        self.assertEqual(res["attempt_log"], [])
        self.assertEqual(res["passages"], [])
        self.assertEqual(res["blocked_citations"], ["b"])
        row = self.last_audit()
        self.assertEqual(row["status"], "refused_citation_blocked")
        self.assertEqual(row["blocked_citations"], ["b"])

    def test_allowed_citation_is_answered(self):
        p = make_pipe("Revenue grew last year [P1].", self.path)
        res = p.answer("q", user=U("guest"), strategy="pre")
        self.assertEqual(res["status"], "answered")
        self.assertNotIn("blocked_citations", res)
        self.assertEqual(self.last_audit()["blocked_citations"], [])

    def test_no_user_skips_guard(self):
        # documents current behaviour: without a user there is no mask, so nothing is checked
        p = make_pipe("The plan is confidential [P2].", self.path)
        res = p.answer("q", user=None, strategy="none")
        self.assertEqual(res["status"], "answered")


if __name__ == "__main__":
    unittest.main()