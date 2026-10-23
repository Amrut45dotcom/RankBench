"""Helpers for level2/run_leakage.py: Groq rate-limit handling and resuming a partial run.
Pure Python (no model loading), so the unit tests run in a second."""
import json
import os
import re

_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def is_rate_limit(exc):
    """True for an HTTP 429 from the LLM provider."""
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ == "RateLimitError" or "Error code: 429" in str(exc)


def is_daily_limit(msg):
    """True when the 429 text says the per-day quota (tokens or requests) is used up."""
    m = (msg or "").lower()
    return "per day" in m or "(tpd)" in m or "(rpd)" in m


def parse_wait_seconds(msg):
    """Seconds in 'try again in 2m39.936s' / '7.5s' / '350ms' / '1h2m3s'; None if absent."""
    m = re.search(r"[Tt]ry again in\s+([0-9hms.\s]+)", msg or "")
    if not m:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1))
    if not parts:
        return None
    return sum(float(v) * _UNIT_S[u] for v, u in parts)


def load_resume(path, strategies):
    """Answered runs from an earlier answer-level result file, keyed (persona, query, strategy).
    Rows with an answer error or no answer are not reused, so they get redone."""
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("summary", {}).get("answer_level"):
        raise ValueError("resume file is not an answer-level run")
    done = {}
    for r in data.get("rows", []):
        if r.get("strategy") in strategies and r.get("answer_leak") is not None:
            done[(r["persona"], r["query"], r["strategy"])] = {
                k: r.get(k) for k in ("answer_leak", "cited_forbidden", "text_overlap", "status")}
    return done