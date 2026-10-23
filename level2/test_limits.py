import json
import os
import tempfile
import unittest

from level2.limits import is_rate_limit, is_daily_limit, parse_wait_seconds, load_resume

# approximate wording of Groq 429 messages
TPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
       "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199284, Requested 1000. "
       "Please try again in 9m10.5s. Need more tokens? Upgrade to Dev Tier.'}}")
TPM = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
       "service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7500, Requested 1000. "
       "Please try again in 3.75s. Need more tokens? Upgrade to Dev Tier.'}}")


class RateLimitError(Exception):
    pass


class Fake429(Exception):
    status_code = 429


class TestLimits(unittest.TestCase):
    def test_parse_wait(self):
        self.assertAlmostEqual(parse_wait_seconds(TPD), 550.5)
        self.assertAlmostEqual(parse_wait_seconds(TPM), 3.75)
        self.assertAlmostEqual(parse_wait_seconds("Please try again in 350ms."), 0.35)
        self.assertAlmostEqual(parse_wait_seconds("Please try again in 1h2m3s."), 3723)
        self.assertIsNone(parse_wait_seconds("no hint here"))
        self.assertIsNone(parse_wait_seconds(None))

    def test_daily(self):
        self.assertTrue(is_daily_limit(TPD))
        self.assertFalse(is_daily_limit(TPM))
        self.assertFalse(is_daily_limit(None))

    def test_is_rate_limit(self):
        self.assertTrue(is_rate_limit(RateLimitError("x")))
        self.assertTrue(is_rate_limit(Fake429("x")))
        self.assertTrue(is_rate_limit(Exception(TPM)))
        self.assertFalse(is_rate_limit(ValueError("boom")))

    def test_load_resume(self):
        rows = [
            {"persona": "guest", "query": "q1", "strategy": "pre", "answer_leak": False,
             "cited_forbidden": [], "text_overlap": False, "status": "answered"},
            {"persona": "guest", "query": "q2", "strategy": "pre", "answer_leak": None,
             "answer_error": "RateLimitError: x"},
            {"persona": "guest", "query": "q1", "strategy": "post", "answer_leak": False,
             "cited_forbidden": [], "text_overlap": False, "status": "answered"},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            json.dump({"summary": {"answer_level": True}, "rows": rows}, open(p, "w"))
            done = load_resume(p, ("pre",))
            self.assertEqual(list(done), [("guest", "q1", "pre")])
            self.assertEqual(done[("guest", "q1", "pre")]["status"], "answered")
            self.assertEqual(len(load_resume(p, ("pre", "post"))), 2)
            p2 = os.path.join(d, "b.json")
            json.dump({"summary": {"answer_level": False}, "rows": rows}, open(p2, "w"))
            with self.assertRaises(ValueError):
                load_resume(p2, ("pre",))


if __name__ == "__main__":
    unittest.main()