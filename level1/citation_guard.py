import re


def blocked_citations(cited, passages, is_allowed):
    """Passage ids cited in an answer that the user may not see.

    cited: numbers from the validator ([1, 3]) or markers like "P2"; 1-based.
    passages: the list sent to the LLM, each with a "pid".
    is_allowed: pid -> bool. If it raises, the citation counts as blocked (fail closed).
    Numbers outside the passage list are ignored; the validator already rejects them.
    """
    out = []
    for c in cited or []:
        if isinstance(c, int) and not isinstance(c, bool):
            n = c
        else:
            m = re.findall(r"\d+", str(c))
            if not m:
                continue
            n = int(m[0])
        if not 1 <= n <= len(passages):
            continue
        pid = passages[n - 1]["pid"]
        try:
            ok = bool(is_allowed(pid))
        except Exception:
            ok = False
        if not ok and pid not in out:
            out.append(pid)
    return out