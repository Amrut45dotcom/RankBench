import os
import ir_datasets
from ranx import Qrels, Run, compare
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_NAME     = "msmarco-passage/dev/small"
BM25_RUN_PATH    = "results/bm25_run.json"
DENSE_RUN_PATH   = "results/dense_run.json"
RRF_RUN_PATH     = "results/rrf_run.json"
RERANKER_RUN_PATH = "results/reranker_run.json"
RESULTS_DIR      = "results"
REPORT_PATH      = os.path.join(RESULTS_DIR, "final_comparison.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load qrels ─────────────────────────────────────────────────────────────
print("[1/3] Loading qrels ...")
dataset = ir_datasets.load(DATASET_NAME)

qrels_dict = {}
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

qrels = Qrels(qrels_dict)

# ── 2. Load all runs ──────────────────────────────────────────────────────────
print("[2/3] Loading runs ...")
bm25_run     = Run.from_file(BM25_RUN_PATH,     name="BM25")
dense_run    = Run.from_file(DENSE_RUN_PATH,    name="Dense (BGE-large)")
rrf_run      = Run.from_file(RRF_RUN_PATH,      name="RRF Hybrid")
reranker_run = Run.from_file(RERANKER_RUN_PATH, name="Reranker (MiniLM)")

# ── 3. Compare ────────────────────────────────────────────────────────────────
print("[3/3] Running comparison ...")
report = compare(
    qrels,
    runs=[bm25_run, dense_run, rrf_run, reranker_run],
    metrics=["ndcg@10", "mrr@10", "recall@100"],
    max_p=0.05,
)

print("\n" + str(report))

# ── Save report ───────────────────────────────────────────────────────────────
with open(REPORT_PATH, "w") as f:
    f.write(str(report))

print(f"\nReport saved to {REPORT_PATH}")