import os
import time
import csv
import ir_datasets
from ranx import Qrels, Run, evaluate, fuse
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_NAME  = "msmarco-passage/dev/small"
BM25_RUN_PATH = "results/bm25_run.json"
DENSE_RUN_PATH = "results/dense_run.json"
TOP_K         = 100
RESULTS_DIR   = "results"
RESULTS_FILE  = os.path.join(RESULTS_DIR, "rrf_baseline.csv")
RRF_RUN_PATH  = os.path.join(RESULTS_DIR, "rrf_run.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load runs ─────────────────────────────────────────────────────────────
print("[1/4] Loading BM25 and Dense runs ...")
bm25_run  = Run.from_file(BM25_RUN_PATH,  name="bm25")
dense_run = Run.from_file(DENSE_RUN_PATH, name="dense")
print(f"     BM25  queries: {len(bm25_run.run):,}")
print(f"     Dense queries: {len(dense_run.run):,}")

# ── 2. RRF Fusion ─────────────────────────────────────────────────────────────
print("[2/4] Fusing with RRF ...")
t0 = time.time()

rrf_run = fuse(
    runs=[bm25_run, dense_run],
    norm="min-max",
    method="rrf",
)
rrf_run.name = "rrf_hybrid"

# cut to top-100
for qid in rrf_run.run:
    sorted_docs = sorted(rrf_run.run[qid].items(), key=lambda x: x[1], reverse=True)
    rrf_run.run[qid] = dict(sorted_docs[:TOP_K])

fusion_time = time.time() - t0
print(f"     Fusion done  [{fusion_time:.2f}s]")

# ── 3. Save RRF run ───────────────────────────────────────────────────────────
print("[3/4] Saving RRF run ...")
rrf_run.save(RRF_RUN_PATH)
print(f"     Saved to {RRF_RUN_PATH}")

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("[4/4] Evaluating ...")
dataset = ir_datasets.load(DATASET_NAME)

qrels_dict = {}
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

qrels   = Qrels(qrels_dict)
metrics = evaluate(qrels, rrf_run, ["ndcg@10", "mrr@10", "recall@100"])

ndcg_10    = metrics["ndcg@10"]
mrr_10     = metrics["mrr@10"]
recall_100 = metrics["recall@100"]

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  NDCG@10      : {ndcg_10:.4f}       │")
print(f"  │  MRR@10       : {mrr_10:.4f}       │")
print(f"  │  Recall@100   : {recall_100:.4f}       │")
print(f"  │  Fusion time  : {fusion_time:.2f}s          │")
print(f"  └─────────────────────────────┘\n")

# ── Log to CSV ────────────────────────────────────────────────────────────────
write_header = not os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model", "dataset", "ndcg@10", "mrr@10",
            "recall@100", "fusion_time_s"
        ])
    writer.writerow([
        "rrf_hybrid", DATASET_NAME,
        f"{ndcg_10:.4f}", f"{mrr_10:.4f}", f"{recall_100:.4f}",
        f"{fusion_time:.2f}"
    ])

print(f"Results saved to {RESULTS_FILE}")