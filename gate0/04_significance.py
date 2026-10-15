import os, yaml
import ir_datasets
from ranx import Qrels, Run, compare

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
ROOT = os.path.expanduser(CFG["data_root"])
SPLITS = os.path.expanduser(CFG["paths"]["splits_dir"])
RUNS = os.path.join(ROOT, CFG["paths"]["results_dir"], "runs_matched")

final = [l.strip() for l in open(os.path.join(SPLITS, "final.txt")) if l.strip()]
qd = {}
for qr in ir_datasets.load(CFG["dataset"]).qrels_iter():
    qd.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
qrels = Qrels({q: qd[q] for q in final})

names = [("bm25", "BM25"), ("dense", "Dense (BGE-large)"),
         ("rrf", "RRF Hybrid"), ("reranker", "Reranker (MiniLM)")]
runs = [Run.from_file(os.path.join(RUNS, f"{f}.json"), name=n) for f, n in names]

report = compare(qrels, runs=runs, metrics=["ndcg@10", "mrr@10", "recall@100"], max_p=0.05)
print(report)
open(os.path.join(ROOT, CFG["paths"]["results_dir"], "final_comparison_matched.txt"), "w").write(str(report))