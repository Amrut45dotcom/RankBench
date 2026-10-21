import datetime
import json
import os
import re
import threading


class AuditLog:
    """Appends one JSON line per request. Logging must never break a request."""

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _role(user):
        if user is None:
            return "anonymous"
        if getattr(user, "is_admin", False):
            return "admin"
        tenant = getattr(user, "tenant", None)
        if tenant is None:
            return "guest"
        return f"tenant{tenant}/clearance{getattr(user, 'clearance', '?')}"

    @staticmethod
    def _cited_ids(res):
        """Map cited [P#] markers back to passage ids."""
        passages = res.get("passages") or []
        out = []
        for c in res.get("cited") or []:
            m = re.findall(r"\d+", str(c))
            if m and 1 <= int(m[0]) <= len(passages):
                out.append(passages[int(m[0]) - 1]["pid"])
        return out

    def log(self, user, strategy, query, res=None, total_ms=None, error=None):
        res = res or {}
        trace = res.get("trace") or {}
        rec = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user": getattr(user, "name", None),
            "role": self._role(user),
            "strategy": strategy,
            "query": query,
            "qid": res.get("qid"),
            "retrieved_ids": trace.get("rerank_input_ids", []),   # candidates that reached the reranker
            "reranked_ids": trace.get("final_ids", []),           # top passages sent to the LLM
            "cited_ids": self._cited_ids(res),
            "status": res.get("status", "error" if error else None),
            "refused": res.get("refused"),
            "error": error,
            "retrieval_ms": res.get("retrieval_ms"),
            "generation_ms": res.get("generation_ms"),
            "total_ms": total_ms,
        }
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:   # never take down a request because of logging
            print(f"[audit] write failed: {e}", flush=True)
