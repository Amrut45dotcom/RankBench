import os
import time
import csv
import numpy as np
import faiss
import ir_datasets
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "msmarco-passage/dev/small"
INDEX_PATH     = "faiss_index/bge_large_1M.index"
PID_LIST_PATH  = "embeddings/pid_list.npy"
TOP_K          = 100
GPU_ID         = 1
RESULTS_DIR    = "results"
RESULTS_FILE   = os.path.join(RESULTS_DIR, "bge_dense.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load FAISS index + pid list ───────────────────────────────────────────
print("[1/5] Loading FAISS index ...")
index    = faiss.read_index(INDEX_PATH)
pid_list = np.load(PID_LIST_PATH)
print(f"     Index size: {index.ntotal:,} vectors")

# ── 2. Load BGE model ────────────────────────────────────────────────────────
print("[2/5] Loading BGE model ...")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# ── 3. Load queries ──────────────────────────────────────────────────────────
print("[3/5] Loading queries ...")
dataset = ir_datasets.load(DATASET_NAME)
queries = {q.query_id: q.text for q in dataset.queries_iter()}
print(f"     Total queries: {len(queries):,}")

# ── 4. Run query pipeline ────────────────────────────────────────────────────
print("[4/5] Running query pipeline ...")
run_dict        = {}
per_query_times = []

for qid, qtext in tqdm(queries.items(), desc="Querying"):
    qt0 = time.time()

    # embed query
    q_vec = model.encode(
        [qtext],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # search FAISS
    scores, indices = index.search(q_vec, TOP_K)

    run_dict[qid] = {
        str(pid_list[indices[0][i]]): float(scores[0][i])
        for i in range(TOP_K)
    }

    per_query_times.append(time.time() - qt0)

avg_ms  = (sum(per_query_times) / len(per_query_times)) * 1000
total_s = sum(per_query_times)

# ── 5. Evaluate with ranx ────────────────────────────────────────────────────
print("[5/5] Evaluating ...")
qrels_dict = {}
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

qrels = Qrels(qrels_dict)
run   = Run(run_dict, name="bge_dense")

metrics = evaluate(qrels, run, ["ndcg@10", "mrr@10", "recall@100"])

ndcg_10   = metrics["ndcg@10"]
mrr_10    = metrics["mrr@10"]
recall_100 = metrics["recall@100"]

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  NDCG@10      : {ndcg_10:.4f}       │")
print(f"  │  MRR@10       : {mrr_10:.4f}       │")
print(f"  │  Recall@100   : {recall_100:.4f}       │")
print(f"  │  Avg latency  : {avg_ms:.1f} ms/q    │")
print(f"  │  Total time   : {total_s:.1f}s          │")
print(f"  └─────────────────────────────┘\n")

# ── Log to CSV ───────────────────────────────────────────────────────────────
write_header = not os.path.exists(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model", "dataset", "ndcg@10", "mrr@10",
            "recall@100", "avg_latency_ms", "total_time_s",
            "num_queries"
        ])
    writer.writerow([
        "bge-large", DATASET_NAME,
        f"{ndcg_10:.4f}", f"{mrr_10:.4f}", f"{recall_100:.4f}",
        f"{avg_ms:.1f}", f"{total_s:.1f}",
        len(queries)
    ])

print(f"Results saved to {RESULTS_FILE}")