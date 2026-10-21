"""L2.6 comparison run. Run from ~/rankbench_v2.

  python level2/run_comparison.py --mode retrieval          # all 6,780 final queries, stops after RRF (no reranker)
  python level2/run_comparison.py --mode rerank             # fixed 1,000-query sample, full pipeline to the reranker

For each query it runs: unfiltered (none), then every persona x {post, oversample, pre}.
Raw per-run rows are appended to results/level2/comparison_<mode>.jsonl (resumable: finished
(query, persona, strategy) triples are skipped on restart). Tables come from make_table.py.
Run the two modes one after the other, not at the same time, or the latency numbers are polluted.
"""
import os, sys, json, time, argparse
H = os.path.expanduser("~")
sys.path[:0] = [f"{H}/rankbench_v2/level1", f"{H}/rankbench_v2"]

import numpy as np
from pipeline import load_pipeline
from level2.permissions import PERSONAS, load_labels, allowed_mask

STRATS = ("post", "oversample", "pre")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("retrieval", "rerank"), required=True)
ap.add_argument("--personas", nargs="+", default=["guest", "hr_intern", "hr_manager"])
ap.add_argument("--n-sample", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N queries, writes a *_test file")
args = ap.parse_args()
for p in args.personas:
    assert p in PERSONAS and p != "admin", f"bad persona {p}"

pipe = load_pipeline(f"{H}/rankbench_v2/level1/config.yaml")
tenant, level = load_labels(f"{H}/RankBench/level2", f"{H}/RankBench/embeddings/pid_list.npy")
pipe.set_permissions(lambda u: allowed_mask(u, tenant, level))
pidx = {p: i for i, p in enumerate(pipe.pid_list)}
masks = {n: allowed_mask(PERSONAS[n], tenant, level) for n in args.personas}

out_dir = f"{H}/RankBench/results/level2"
os.makedirs(out_dir, exist_ok=True)

# ---------------- queries ----------------
final_path = next(p for p in (f"{H}/rankbench_v2/splits/final.txt", f"{H}/RankBench/splits/final.txt")
                  if os.path.exists(p))
qids = [l.strip() for l in open(final_path) if l.strip()]
if args.mode == "rerank":
    sample_path = f"{out_dir}/rerank_sample_qids.txt"
    if os.path.exists(sample_path):                      # frozen once created
        qids = [l.strip() for l in open(sample_path) if l.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        qids = [str(q) for q in rng.choice(qids, size=args.n_sample, replace=False)]
        open(sample_path, "w").write("\n".join(qids) + "\n")
if args.limit:
    qids = qids[:args.limit]
suffix = "_test" if args.limit else ""
out_path = f"{out_dir}/comparison_{args.mode}{suffix}.jsonl"

done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l)
        done.add((r["qid"], r["persona"], r["strategy"]))
print(f"mode={args.mode} queries={len(qids)} personas={args.personas} already done={len(done)}", flush=True)


def metrics(ranked, fused, rel):
    top = ranked[:10]
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top) if p in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    mrr = next((1.0 / (i + 1) for i, p in enumerate(top) if p in rel), 0.0)
    return {"ndcg10": dcg / idcg, "mrr10": mrr, "recall100": len(set(fused[:100]) & rel) / len(rel)}


rerank = args.mode == "rerank"
n_out = 10 if rerank else 100
combos = [("none", "none")] + [(p, s) for p in args.personas for s in STRATS]

# warm-up (not recorded): first calls load kernels and caches
for qid in qids[:3]:
    q = pipe.qtext[qid]
    pipe.retrieve_traced(q, n_out, strategy="none", rerank=rerank)
    pipe.retrieve_traced(q, n_out, user=PERSONAS[args.personas[0]], strategy="pre", rerank=rerank)

t_run = time.time()
with open(out_path, "a", encoding="utf-8") as fout:
    for n, qid in enumerate(qids, 1):
        q = pipe.qtext[qid]
        gold = {p for p, r in pipe.qrels.get(qid, {}).items() if r > 0 and p in pidx}
        if not gold:
            continue
        for persona, strat in combos:
            if (qid, persona, strat) in done:
                continue
            if persona == "none":
                user, rel = None, gold
            else:
                user = PERSONAS[persona]
                rel = {p for p in gold if masks[persona][pidx[p]]}
            t0 = time.perf_counter()
            out, tr = pipe.retrieve_traced(q, n_out, user=user, strategy=strat, rerank=rerank)
            ms = (time.perf_counter() - t0) * 1000
            row = {"qid": qid, "persona": persona, "strategy": strat, "visible": bool(rel),
                   "n_gold": len(gold), "n_returned": len(tr["fused_ids"]),
                   "ms_total": round(ms, 2), "ms_retrieval": round(tr["ms_retrieval"], 2),
                   "ms_rerank": round(tr["ms_rerank"], 2)}
            if rel:
                row.update({k: round(v, 6) for k, v in metrics(tr["final_ids"], tr["fused_ids"], rel).items()})
            fout.write(json.dumps(row) + "\n")
            fout.flush()
        if n % 100 == 0:
            el = time.time() - t_run
            print(f"  {n}/{len(qids)} queries, {el/60:.1f} min elapsed, ~{el/n*(len(qids)-n)/60:.0f} min left", flush=True)

print(f"finished -> {out_path}")
