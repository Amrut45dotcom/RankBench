import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "level1"))

from fastapi.testclient import TestClient      # noqa: E402
import main                                     # noqa: E402
from public_view import public_view             # noqa: E402
from pipeline import REFUSAL_TEXT               # noqa: E402
from level2.permissions import PERSONAS         # noqa: E402


class Fake:
    def __init__(self, status="answered"):
        self.status, self.seen = status, None

    def answer(self, q, user=None, strategy="none"):
        self.seen = (user, strategy)
        if self.status == "boom":
            raise AssertionError("LEAK before rerank: ['SECRET']")
        return {"status": self.status, "refused": self.status != "answered", "answer": "ok [P1]",
                "cited": [1], "passages": [{"pid": "a", "text": "ta"}], "reasons": ["x"],
                "trace": {"final_ids": ["SECRET"]}, "attempt_log": [{"raw": "SECRET"}],
                "raw_answer": "SECRET", "blocked_citations": ["SECRET"]}


def setup(fake, perms=True):
    main.state.clear()
    main.state.update({"l1": fake, "l1_perms": perms, "public_view": public_view,
                       "personas": {n: u for n, u in PERSONAS.items() if n != "admin"}})
    return TestClient(main.app)       # no 'with': startup (model loading) does not run


class TestApiAnswer(unittest.TestCase):
    def test_unknown_persona_becomes_guest(self):
        f = Fake()
        setup(f).post("/answer", json={"query": "q", "persona": "nobody"})
        self.assertEqual(f.seen[0].name, "guest")
        self.assertEqual(f.seen[1], "pre")

    def test_admin_over_http_becomes_guest(self):
        f = Fake()
        setup(f).post("/answer", json={"query": "q", "persona": "admin"})
        self.assertEqual(f.seen[0].name, "guest")

    def test_known_persona_is_used(self):
        f = Fake()
        setup(f).post("/answer", json={"query": "q", "persona": "hr_manager"})
        self.assertEqual(f.seen[0].name, "hr_manager")

    def test_response_has_only_public_fields(self):
        r = setup(Fake()).post("/answer", json={"query": "q", "persona": "guest"})
        self.assertEqual(set(r.json()), {"answer", "refused", "citations"})
        self.assertNotIn("SECRET", r.text)

    def test_refusals_identical_for_absent_and_forbidden(self):
        a = setup(Fake("refused_no_passages")).post("/answer", json={"query": "q"})
        b = setup(Fake("refused_citation_blocked")).post("/answer", json={"query": "q"})
        self.assertEqual(a.json(), b.json())
        self.assertEqual(a.json()["answer"], REFUSAL_TEXT)

    def test_error_does_not_leak_detail(self):
        r = setup(Fake("boom")).post("/answer", json={"query": "q"})
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("SECRET", r.text)

    def test_disabled_without_permissions(self):
        r = setup(Fake(), perms=False).post("/answer", json={"query": "q"})
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()