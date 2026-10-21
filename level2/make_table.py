"""L2.6 tables. Run from anywhere:  python level2/make_table.py [--test]

Reads results/level2/comparison_retrieval.jsonl and comparison_rerank.jsonl (from run_comparison.py).
Metrics are averaged over gold-visible (query, persona) pairs only. Gold-hidden pairs are counted
and reported as "should refuse", not as retrieval failures.
Retention = strategy NDCG@10 / unfiltered NDCG@10 on the same gold-visible queries.
"""
import os, sys, json
from collections import defaultdict
import numpy as np

H = os.path.expanduser("~")
D = f"{H}/RankBench/results/level2"
suffix = "_test" if "--test" in sys.argv else ""
STRATS = ("post", "oversample", "pre")
lines_out = []


def load(mode):
    p = f"{D}/comparison_{mode}{suffix}.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def table(mode, rows, title):
    none = {r["qid"]: r for r in rows if r["persona"] == "none"}
    personas = sorted({r["persona"] for r in rows if r["persona"] != "none"})
    n_q = len(none)
    out = [f"### {title} ({n_q} queries)", ""]

    # sanity line: unfiltered baseline on all queries
    vis = [r for r in none.values() if r["visible"]]
    out.append(f"Unfiltered baseline (all queries): NDCG@10 {np.mean([r['ndcg10'] for r in vis]):.4f}, "
               f"MRR@10 {np.mean([r['mrr10'] for r in vis]):.4f}, Recall@100 {np.mean([r['recall100'] for r in vis]):.4f}, "
               f"p50 {np.percentile([r['ms_total'] for r in none.values()], 50):.0f} ms, "
               f"p95 {np.percentile([r['ms_total'] for r in none.values()], 95):.0f} ms")
    out.append("")
    out.append("| Persona | Strategy | Gold visible | Should refuse | NDCG@10 | Retention | MRR@10 | Recall@100 | Avg candidates | p50 ms | p95 ms |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for persona in personas:
        for s in STRATS:
            rs = [r for r in rows if r["persona"] == persona and r["strategy"] == s]
            if not rs:
                continue
            v = [r for r in rs if r["visible"]]
            hidden = len(rs) - len(v)
            base = np.mean([none[r["qid"]]["ndcg10"] for r in v if r["qid"] in none and none[r["qid"]]["visible"]])
            nd = np.mean([r["ndcg10"] for r in v])
            lat = [r["ms_total"] for r in rs]
            out.append(f"| {persona} | {s} | {len(v)} ({100*len(v)/len(rs):.0f}%) | {hidden} ({100*hidden/len(rs):.0f}%) | "
                       f"{nd:.4f} | {100*nd/base:.1f}% | {np.mean([r['mrr10'] for r in v]):.4f} | "
                       f"{np.mean([r['recall100'] for r in v]):.4f} | {np.mean([r['n_returned'] for r in rs]):.1f} | "
                       f"{np.percentile(lat, 50):.0f} | {np.percentile(lat, 95):.0f} |")
    out.append("")
    lines_out.extend(out)


for mode, title in (("retrieval", "Retrieval-only (after RRF, no reranker), all final queries"),
                    ("rerank", "Full pipeline with reranker, fixed sample")):
    rows = load(mode)
    if rows:
        table(mode, rows, title)
    else:
        lines_out.append(f"(no data for mode {mode})\n")

lines_out += ["Notes: gold-visible = the persona may see at least one passage MS MARCO marks relevant; "
              "NDCG/MRR/Recall use only the visible gold passages. Latency is end-to-end per query on a shared server, "
              "compare within a run only. Avg candidates = size of the fused list before reranking (up to 100).", ""]
text = "\n".join(lines_out)
print(text)
open(f"{D}/comparison_tables{suffix}.md", "w").write(text)
print(f"saved: {D}/comparison_tables{suffix}.md")
