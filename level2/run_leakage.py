import os, sys, re, json, time, argparse, datetime
H = os.path.expanduser("~")
sys.path[:0] = [f"{H}/rankbench_v2/level1", f"{H}/rankbench_v2"]

import numpy as np
from pipeline import load_pipeline
from level2.permissions import PERSONAS, load_labels, allowed_mask
from level2.audit import AuditLog
from level2.limits import is_rate_limit, is_daily_limit, parse_wait_seconds, load_resume

STAGES = ("bm25_ids", "dense_ids", "rerank_input_ids", "final_ids")
STRATS = ("post", "oversample", "pre")
NON_ADMIN = [n for n in PERSONAS if n != "admin"]
TOPICAL_PERSONAS = ("guest", "hr_intern", "finance_analyst", "legal_manager")
MAX_RATE_RETRIES = 3
MAX_WAIT_S = 600


class DailyLimit(Exception):
    pass


def shingles(text, n=8):
    w = re.findall(r"\w+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


ap = argparse.ArgumentParser()
ap.add_argument("--answer", action="store_true", help="also run the LLM and check cited ids and answer text")
ap.add_argument("--topical-per-persona", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-tests", type=int, default=0, help="debug: only the first N tests")
ap.add_argument("--strategies", default=",".join(STRATS), help="comma list, subset of post,oversample,pre")
ap.add_argument("--resume", default=None, help="earlier *_answer.json: reuse its answered rows (skips those Groq calls)")
ap.add_argument("--attacks", default="attacks.json",
                help="attack file inside level2/ (default: the frozen attacks.json)")
args = ap.parse_args()

STRATS_RUN = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
if not STRATS_RUN or any(s not in STRATS for s in STRATS_RUN):
    ap.error(f"--strategies must be a comma list from {','.join(STRATS)}")
if args.resume and not args.answer:
    ap.error("--resume needs --answer")

done = {}
if args.resume:
    done = load_resume(args.resume, STRATS_RUN)
    print(f"resume: {len(done)} answered runs reused from {args.resume}", flush=True)

pipe = load_pipeline(f"{H}/rankbench_v2/level1/config.yaml")
tenant, level = load_labels(f"{H}/RankBench/level2", f"{H}/RankBench/embeddings/pid_list.npy")
pipe.set_permissions(lambda u: allowed_mask(u, tenant, level))
pidx = {p: i for i, p in enumerate(pipe.pid_list)}
masks = {n: allowed_mask(u, tenant, level) for n, u in PERSONAS.items()}

out_dir = f"{H}/RankBench/results/level2"
os.makedirs(out_dir, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.answer:
    pipe.set_audit(AuditLog(f"{out_dir}/audit_leakage_{stamp}.jsonl"))


def answer_with_retry(q, qid, user, s):
    """Returns (result, error_text). Waits out per-minute limits. Raises DailyLimit when the daily
    quota is used up. AssertionError (the leak guard) passes through to the caller."""
    for attempt in range(MAX_RATE_RETRIES + 1):
        try:
            return pipe.answer(q, qid=qid, user=user, strategy=s), None
        except AssertionError:
            raise
        except Exception as e:
            msg = str(e)
            if is_rate_limit(e):
                if is_daily_limit(msg):
                    raise DailyLimit(msg[:300])
                wait = parse_wait_seconds(msg)
                wait = 30.0 if wait is None else wait
                if attempt < MAX_RATE_RETRIES and wait <= MAX_WAIT_S:
                    print(f"  rate limit, waiting {wait + 1:.0f}s (retry {attempt + 1}/{MAX_RATE_RETRIES})", flush=True)
                    time.sleep(wait + 1)
                    continue
            return None, f"{type(e).__name__}: {msg}"[:300]
    return None, "RateLimitError: retries exhausted"


# ---------------- build the tests ----------------
tests = []
attacks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.attacks)
with open(attacks_path, encoding="utf-8") as f:
    attacks = json.load(f)
for cat, qs in attacks.items():
    for q in qs:
        for name in NON_ADMIN:
            tests.append({"category": cat, "persona": name, "qid": None, "query": q})

dev_path = next(p for p in (f"{H}/rankbench_v2/splits/dev_200.txt", f"{H}/RankBench/splits/dev_200.txt")
                if os.path.exists(p))
dev_qids = [l.strip() for l in open(dev_path) if l.strip()]
rng = np.random.default_rng(args.seed)
for name in TOPICAL_PERSONAS:
    m = masks[name]
    hidden = []
    for qid in dev_qids:
        gold = [p for p, r in pipe.qrels.get(qid, {}).items() if r > 0 and p in pidx]
        if gold and not any(m[pidx[p]] for p in gold) and qid in pipe.qtext:
            hidden.append(qid)
    pick = rng.choice(hidden, size=min(args.topical_per_persona, len(hidden)), replace=False)
    for qid in pick:
        tests.append({"category": "gold_hidden", "persona": name, "qid": str(qid), "query": pipe.qtext[str(qid)]})
if args.max_tests:
    tests = tests[:args.max_tests]
print(f"\n{len(tests)} tests (query, persona) x {len(STRATS_RUN)} strategies ({', '.join(STRATS_RUN)})", flush=True)

# ---------------- run ----------------
rows = []
control_leaks = 0
stopped = None
for n, t in enumerate(tests, 1):
    user, mask, q = PERSONAS[t["persona"]], masks[t["persona"]], t["query"]

    # control: unfiltered search judged against this persona's rules
    _, tr0 = pipe.retrieve_traced(q, 5, strategy="none")
    forbidden0 = [p for p in tr0["final_ids"] if not mask[pidx[p]]]
    control_leaks += bool(forbidden0)
    pool = shingles(" ".join(pipe.get_texts(forbidden0))) if (args.answer and forbidden0) else set()

    for s in STRATS_RUN:
        row = {**t, "strategy": s, "guard_fired": False, "leaks": {}, "n_final": 0,
               "answer_leak": None}
        try:
            out, tr = pipe.retrieve_traced(q, 5, user=user, strategy=s)
            row["n_final"] = len(out)
            for k in STAGES:
                bad = [p for p in tr[k] if not mask[pidx[p]]]
                if bad:
                    row["leaks"][k] = bad
        except AssertionError as e:          # the in-pipeline guard caught something: that is a finding
            row["guard_fired"] = True
            row["error"] = str(e)
        if args.answer and not row["guard_fired"]:
            key = (t["persona"], q, s)
            if key in done:                  # answered in the earlier run: reuse, no Groq call
                row.update(done[key])
                row["resumed"] = True
            else:
                res, err = None, None
                try:
                    res, err = answer_with_retry(q, t["qid"], user, s)
                except AssertionError as e:  # the leak guard fired inside answer(): a finding, not a network error
                    row["guard_fired"], row["error"] = True, str(e)
                except DailyLimit as e:      # quota gone: save what we have and stop
                    stopped = str(e)
                    break
                if err:
                    row["answer_error"] = err
                if res is not None:
                    cited = AuditLog._cited_ids(res)
                    cited_bad = [p for p in cited if not mask[pidx[p]]]
                    text = (res.get("answer") or "") + " " + (res.get("raw_answer") or "")
                    overlap = bool(pool & shingles(text))
                    blocked = res.get("blocked_citations") or []
                    row["answer_leak"] = bool(cited_bad or overlap or blocked)
                    row["blocked_citations"] = blocked
                    row["cited_forbidden"], row["text_overlap"], row["status"] = cited_bad, overlap, res["status"]
        row["leaked"] = bool(row["leaks"]) or row["guard_fired"] or bool(row["answer_leak"])
        rows.append(row)
    if stopped:
        print(f"\nDAILY LIMIT reached at test {n}/{len(tests)}: stopping and saving the partial run", flush=True)
        break
    if n % 25 == 0:
        print(f"  {n}/{len(tests)} tests done", flush=True)

# ---------------- summary ----------------
expected = len(tests) * len(STRATS_RUN)
answered = sum(1 for r in rows if r["answer_leak"] is not None) if args.answer else 0
summary = {"tests": len(tests), "strategies": {}, "by_category": {}, "answer_level": args.answer,
           "strategies_run": list(STRATS_RUN),
           "control_unfiltered_leaking_tests": control_leaks, "dev_split": dev_path,
           "seed": args.seed,
           "answer_errors": sum(1 for r in rows if r.get("answer_error")),
           "resumed_from": args.resume, "resumed_rows": sum(1 for r in rows if r.get("resumed")),
           "stopped_early": stopped,
           "answer_coverage": {"answered": answered, "expected": expected} if args.answer else None}
for s in STRATS_RUN:
    rs = [r for r in rows if r["strategy"] == s]
    summary["strategies"][s] = {
        "tests": len(rs),
        "leaking_tests": sum(r["leaked"] for r in rs),
        "guard_fired": sum(r["guard_fired"] for r in rs),
        "avg_final_results": round(float(np.mean([r["n_final"] for r in rs])), 2) if rs else None,
        "empty_results": sum(r["n_final"] == 0 for r in rs),
    }
for cat in sorted({r["category"] for r in rows}):
    for s in STRATS_RUN:
        rs = [r for r in rows if r["category"] == cat and r["strategy"] == s]
        summary["by_category"].setdefault(cat, {})[s] = {"tests": len(rs), "leaking_tests": sum(r["leaked"] for r in rs)}

path = f"{out_dir}/leakage_{stamp}{'_answer' if args.answer else ''}.json"
json.dump({"summary": summary, "rows": rows}, open(path, "w"), indent=1)
print("\n" + json.dumps(summary, indent=1))
print(f"\nCONTROL: unfiltered search leaked in {control_leaks}/{len(tests)} tests (expect > 0)")
total_leaks = sum(v["leaking_tests"] for v in summary["strategies"].values())
print(f"TOTAL LEAKING TESTS (filtered strategies): {total_leaks}")
print(f"saved: {path}")
if stopped:
    print(f"STOPPED EARLY (daily Groq limit). Rerun with --resume {path}")
if args.answer:
    print(f"ANSWER-LEVEL COVERAGE: {answered} of {expected}  (answer errors: {summary['answer_errors']})")
incomplete = args.answer and (stopped is not None or answered < expected)
sys.exit(1 if total_leaks else (2 if incomplete else 0))