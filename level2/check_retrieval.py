"""L2.3 check: run from ~/rankbench_v2 (see run command). Loads all models, so check nvidia-smi first."""
import os, sys, json
H = os.path.expanduser("~")
sys.path[:0] = [f"{H}/rankbench_v2/level1", f"{H}/rankbench_v2"]

import numpy as np
from pipeline import load_pipeline
from level2.permissions import PERSONAS, load_labels, allowed_mask

N_QUERIES = 8
pipe = load_pipeline(f"{H}/rankbench_v2/level1/config.yaml")
tenant, level = load_labels(f"{H}/RankBench/level2", f"{H}/RankBench/embeddings/pid_list.npy")
pipe.set_permissions(lambda u: allowed_mask(u, tenant, level))
pidx = {p: i for i, p in enumerate(pipe.pid_list)}

rows = [json.loads(l) for l in open(f"{H}/RankBench/results/level1/answer_eval_run1.jsonl")]
rows = [r for r in rows if r.get("passages")][:N_QUERIES]
queries = [(r["qid"], r["query"]) for r in rows]

# ---- Test 1: strategy none must match the frozen Level 1 run ----
print("\n== Test 1: strategy none vs frozen Level 1 run ==")
same = 0
for r in rows:
    old = [p["pid"] for p in r["passages"]]
    new = [p["pid"] for p in pipe.retrieve(r["query"], len(old), strategy="none")]
    same += (old == new)
    if old != new:
        print(f"  differs: {r['qid']} overlap {len(set(old) & set(new))}/{len(old)}")
print(f"  identical order: {same}/{len(rows)}")

# ---- Test 2: admin must equal none for every strategy ----
print("\n== Test 2: admin gives the same result as none ==")
admin = PERSONAS["admin"]
for qid, q in queries[:5]:
    base = [p["pid"] for p in pipe.retrieve(q, 5, strategy="none")]
    for s in ("post", "oversample", "pre"):
        got = [p["pid"] for p in pipe.retrieve(q, 5, user=admin, strategy=s)]
        print(f"  {qid} {s:10s}", "OK" if got == base else f"DIFF (overlap {len(set(got) & set(base))}/5)")

# ---- Test 3: no forbidden id at any stage, for every persona and strategy ----
print("\n== Test 3: leak check (bm25, dense, rerank input, final) ==")
STAGES = ("bm25_ids", "dense_ids", "rerank_input_ids", "final_ids")
total_bad = 0
for name, user in PERSONAS.items():
    mask = allowed_mask(user, tenant, level)
    for s in ("post", "oversample", "pre"):
        bad = n_fin = n_dense = 0
        for qid, q in queries:
            out, tr = pipe.retrieve_traced(q, 5, user=user, strategy=s)
            for k in STAGES:
                bad += sum(not mask[pidx[p]] for p in tr[k])
            n_fin += len(out)
            n_dense += len(tr["dense_ids"])
        total_bad += bad
        print(f"  {name:16s} {s:10s} leaks={bad}  avg final={n_fin/len(queries):.1f}  avg dense cands={n_dense/len(queries):.0f}")
print(f"\nTOTAL LEAKS (filtered strategies): {total_bad}")

# ---- Control: unfiltered search as guest MUST leak, or the leak check is blind ----
guest_mask = allowed_mask(PERSONAS["guest"], tenant, level)
ctrl = 0
for qid, q in queries:
    _, tr = pipe.retrieve_traced(q, 5, strategy="none")
    ctrl += sum(not guest_mask[pidx[p]] for p in tr["final_ids"])
print(f"CONTROL: strategy none judged against guest permissions shows {ctrl} forbidden passages in final results (expect > 0)")