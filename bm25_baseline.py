import time
import csv
import os
import ir_datasets
import bm25s
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_NAME = "msmarco-passage/dev/small"
TOP_K        = 1000   # retrieve top-1000 per query (standard MS MARCO eval)
RESULTS_DIR  = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "bm25_baseline.csv")

# ── 1. Load dataset ─────────────────────────────────────────────────────────
print(f"[1/5] Loading dataset: {DATASET_NAME}")
dataset = ir_datasets.load(DATASET_NAME)

print("     Building corpus dict {doc_id: text} ...")
t0 = time.time()
corpus = {}
for doc in tqdm(dataset.docs_iter(), desc="Loading docs", total=8_841_823):
    corpus[doc.doc_id] = doc.text

print(f"     Corpus size: {len(corpus):,} passages  [{time.time()-t0:.1f}s]")

# ── 2. Build BM25 index ─────────────────────────────────────────────────────
print("[2/5] Building bm25s index ...")
t0 = time.time()

doc_ids = list(corpus.keys())

if os.path.exists("bm25_index"):
    print("     Loading index from disk ...")
    bm25 = bm25s.BM25.load("bm25_index", load_corpus=False)
    print(f"     Loaded  [{time.time()-t0:.1f}s]")
else:
    corpus_texts = [corpus[did] for did in tqdm(doc_ids, desc="     Preparing", unit=" docs")]
    tokenized = bm25s.tokenize(corpus_texts, show_progress=True)
    bm25 = bm25s.BM25(method="robertson", idf_method="robertson")
    bm25.index(tokenized, show_progress=True)
    bm25.save("bm25_index")
    print(f"     Index built and saved  [{time.time()-t0:.1f}s]")

# ── 3. Run all queries ──────────────────────────────────────────────────────
print("[3/5] Running queries ...")
queries = {q.query_id: q.text for q in dataset.queries_iter()}
print(f"     Total queries: {len(queries):,}")

run_dict          = {}   # {query_id: {doc_id: score}}
per_query_times   = []

for qid, qtext in tqdm(queries.items(), desc="Running queries"):
    qt0 = time.time()
    query_tokens = bm25s.tokenize([qtext])
    results, scores = bm25.retrieve(query_tokens, k=TOP_K)
    run_dict[qid] = {
        doc_ids[results[0][i]]: float(scores[0][i])
        for i in range(len(results[0]))
    }
    per_query_times.append(time.time() - qt0)


avg_ms  = (sum(per_query_times) / len(per_query_times)) * 1000
total_s = sum(per_query_times)
print(f"     Done  |  avg {avg_ms:.1f} ms/query  |  total {total_s:.1f}s")

# ── 4. Evaluate with ranx ───────────────────────────────────────────────────
print("[4/5] Evaluating ...")
qrels_dict = {}
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

qrels = Qrels(qrels_dict)
run   = Run(run_dict, name="bm25_baseline")
run.save("results/bm25_run.json")

metrics = evaluate(qrels, run, ["ndcg@10", "mrr@10", "recall@1000"])

ndcg_10    = metrics["ndcg@10"]
mrr_10     = metrics["mrr@10"]
recall_1k  = metrics["recall@1000"]

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  NDCG@10      : {ndcg_10:.4f}       │")
print(f"  │  MRR@10       : {mrr_10:.4f}       │")
print(f"  │  Recall@1000  : {recall_1k:.4f}       │")
print(f"  │  Avg latency  : {avg_ms:.1f} ms/q    │")
print(f"  │  Total time   : {total_s:.1f}s          │")
print(f"  └─────────────────────────────┘\n")

# ── 5. Log to CSV ───────────────────────────────────────────────────────────
print(f"[5/5] Logging results to {RESULTS_FILE} ...")
os.makedirs(RESULTS_DIR, exist_ok=True)

write_header = not os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model", "dataset", "ndcg@10", "mrr@10",
            "recall@1000", "avg_latency_ms", "total_time_s",
            "num_queries", "corpus_size"
        ])
    writer.writerow([
        "bm25s", DATASET_NAME,
        f"{ndcg_10:.4f}", f"{mrr_10:.4f}", f"{recall_1k:.4f}",
        f"{avg_ms:.1f}", f"{total_s:.1f}",
        len(queries), len(corpus)
    ])

print("Done. bm25s baseline logged.")
