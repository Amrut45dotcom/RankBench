from pipeline import REFUSAL_TEXT


def public_view(res):
    """What a caller may see. Everything else stays in the audit log.

    All refusals (no passages, model refusal, validation failure, blocked citation)
    return the exact same dict. Status, reasons, trace, attempt_log, raw_answer,
    blocked_citations and timings are never exposed.
    """
    if res.get("status") != "answered":
        return {"answer": REFUSAL_TEXT, "refused": True, "citations": []}
    cited = set(res.get("cited") or [])
    cites = []
    for i, p in enumerate(res.get("passages") or [], start=1):
        if i in cited:
            cites.append({"n": i, "pid": p.get("pid"), "text": p.get("text")})
    return {"answer": res["answer"], "refused": False, "citations": cites}