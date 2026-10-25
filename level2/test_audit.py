import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

from level2.audit import AuditLog


@dataclass(frozen=True)
class U:
    name: str
    is_admin: bool = False
    tenant: Optional[int] = None
    clearance: int = 0


RES = {
    "qid": "q1", "status": "answered", "refused": False, "cited": ["P1", "P3"],
    "passages": [{"pid": "a"}, {"pid": "b"}, {"pid": "c"}],
    "trace": {"rerank_input_ids": ["a", "b", "c", "d"], "final_ids": ["a", "b", "c"]},
    "retrieval_ms": 10.0, "generation_ms": 20.0,
}


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "sub", "audit.jsonl")
        self.log = AuditLog(self.path)

    def lines(self):
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    def test_one_valid_line_per_call(self):
        self.log.log(U("guest"), "pre", "q one", RES, 31.0)
        self.log.log(U("admin", is_admin=True), "none", "q two", RES, 12.0)
        rows = self.lines()
        self.assertEqual(len(rows), 2)
        for k in ("ts", "user", "role", "strategy", "query", "retrieved_ids", "reranked_ids",
                  "cited_ids", "refused", "total_ms"):
            self.assertIn(k, rows[0])

    def test_roles(self):
        self.log.log(U("guest"), "pre", "q", RES, 1.0)
        self.log.log(U("hr", tenant=0, clearance=2), "pre", "q", RES, 1.0)
        self.log.log(U("root", is_admin=True), "pre", "q", RES, 1.0)
        self.assertEqual([r["role"] for r in self.lines()], ["guest", "tenant0/clearance2", "admin"])

    def test_cited_ids_map_to_pids(self):
        self.log.log(U("guest"), "pre", "q", RES, 1.0)
        self.assertEqual(self.lines()[0]["cited_ids"], ["a", "c"])

    def test_error_request_is_logged(self):
        self.log.log(U("guest"), "pre", "q", None, 5.0, error="AssertionError: LEAK before rerank")
        row = self.lines()[0]
        self.assertEqual(row["status"], "error")
        self.assertIn("LEAK", row["error"])

    def test_unicode_query_stays_one_line(self):
        self.log.log(U("guest"), "pre", "what is \u00e9t\u00e9?\nsecond line", RES, 1.0)
        self.assertEqual(len(self.lines()), 1)

    def test_logging_failure_does_not_raise(self):
        bad = AuditLog(self.path)
        bad.path = self.dir          # a directory, so open() fails
        bad.log(U("guest"), "pre", "q", RES, 1.0)


if __name__ == "__main__":
    unittest.main()
