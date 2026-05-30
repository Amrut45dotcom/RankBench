import os
import time
import csv
import numpy as np
import ir_datasets
from sentence_transformers import CrossEncoder
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "msmarco-passage/dev/small"
RRF_RUN_PATH   = "results/rrf_run.json"
MODEL_NAME     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_INPUT    = 100   # candidates from RRF
TOP_K_OUTPUT   = 10    # final reranked output
GPU_ID         = 1
RESULTS_DIR    = "results"
RESULTS_FILE   = os.path.join(RESULTS_DIR, "reranker_baseline.csv")
RERANKER_RUN_PATH = os.path.join(RESULTS_DIR, "reranker_run.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# ── 1. Load corpus texts ─────────────────────────────────────────────────────
print("[1/5] Loading corpus texts ...")
t0 = time.time()
dataset = ir_datasets.load(DATASET_NAME)
corpus = {}
for doc in tqdm(dataset.docs_iter(), desc="Loading docs", total=8_841_823):
    corpus[doc.doc_id] = doc.text
print(f"     Corpus size: {len(corpus):,}  [{time.time()-t0:.1f}s]")

# ── 2. Load queries ──────────────────────────────────────────────────────────
print("[2/5] Loading queries ...")
queries = {q.query_id: q.text for q in dataset.queries_iter()}
print(f"     Total queries: {len(queries):,}")

# ── 3. Load RRF run ──────────────────────────────────────────────────────────
print("[3/5] Loading RRF run ...")
rrf_run = Run.from_file(RRF_RUN_PATH, name="rrf_hybrid")
print(f"     Queries in run: {len(rrf_run.run):,}")

# ── 4. Load cross-encoder ────────────────────────────────────────────────────
print("[4/5] Loading cross-encoder model ...")
model = CrossEncoder(MODEL_NAME, device="cuda", max_length=512)

# ── 5. Rerank ────────────────────────────────────────────────────────────────
print("[5/5] Reranking ...")
run_dict        = {}
per_query_times = []

for qid, qtext in tqdm(queries.items(), desc="Reranking"):
    qt0 = time.time()

    # get top-100 candidate pids from RRF
    if qid not in rrf_run.run:
        continue

    candidate_pids = list(rrf_run.run[qid].keys())[:TOP_K_INPUT]

    # build query-passage pairs
    pairs = [
        (qtext, corpus[pid])
        for pid in candidate_pids
        if pid in corpus
    ]
    valid_pids = [
        pid for pid in candidate_pids
        if pid in corpus
    ]

    # score with cross-encoder
    scores = model.predict(pairs, batch_size=64, show_progress_bar=False)

    # sort by score, keep top-10
    scored = sorted(zip(valid_pids, scores), key=lambda x: x[1], reverse=True)
    run_dict[qid] = {
        pid: float(score)
        for pid, score in scored[:TOP_K_OUTPUT]
    }

    per_query_times.append(time.time() - qt0)

avg_ms  = (sum(per_query_times) / len(per_query_times)) * 1000
total_s = sum(per_query_times)

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("Evaluating ...")
qrels_dict = {}
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

qrels = Qrels(qrels_dict)
run   = Run(run_dict, name="reranker")

metrics = evaluate(qrels, run, ["ndcg@10", "mrr@10"])

ndcg_10 = metrics["ndcg@10"]
mrr_10  = metrics["mrr@10"]

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  NDCG@10      : {ndcg_10:.4f}       │")
print(f"  │  MRR@10       : {mrr_10:.4f}       │")
print(f"  │  Avg latency  : {avg_ms:.1f} ms/q    │")
print(f"  │  Total time   : {total_s:.1f}s          │")
print(f"  └─────────────────────────────┘\n")

# ── Save run ──────────────────────────────────────────────────────────────────
run.save(RERANKER_RUN_PATH)
print(f"Run saved to {RERANKER_RUN_PATH}")

# ── Log to CSV ────────────────────────────────────────────────────────────────
write_header = not os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model", "dataset", "ndcg@10", "mrr@10",
            "avg_latency_ms", "total_time_s", "num_queries"
        ])
    writer.writerow([
        MODEL_NAME, DATASET_NAME,
        f"{ndcg_10:.4f}", f"{mrr_10:.4f}",
        f"{avg_ms:.1f}", f"{total_s:.1f}",
        len(run_dict)
    ])

print(f"Results saved to {RESULTS_FILE}")