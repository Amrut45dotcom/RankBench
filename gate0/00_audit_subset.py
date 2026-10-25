"""
Gate 0.1 - audit the existing 1M subset (read-only).

Checks that every eval query still has at least one relevant passage inside
embeddings/pid_list.npy, and records the subset's hash so it can be treated
as frozen. Writes splits/subset_audit.json and splits/subset_pids.txt.gz.

Usage (from the RankBench v2 root):
    python gate0/00_audit_subset.py
"""
import gzip
import hashlib
import json
import os

import ir_datasets
import numpy as np

DATASET_NAME  = "msmarco-passage/dev/small"
PID_LIST_PATH = "embeddings/pid_list.npy"
OUT_DIR       = "splits"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pid_list = [str(p) for p in np.load(PID_LIST_PATH)]
    subset = set(pid_list)
    print(f"pid_list entries: {len(pid_list):,}  unique: {len(subset):,}")

    dataset = ir_datasets.load(DATASET_NAME)
    qrels = {}  # qid -> set of relevant pids
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    all_qids = [q.query_id for q in dataset.queries_iter()]

    relevant_pids = set().union(*qrels.values()) if qrels else set()
    fully, partial, none_ = 0, 0, []
    for qid in all_qids:
        rel = qrels.get(qid, set())
        inside = len(rel & subset)
        if rel and inside == len(rel):
            fully += 1
        elif inside > 0:
            partial += 1
        else:
            none_.append(qid)

    audit = {
        "dataset": DATASET_NAME,
        "pid_list_path": PID_LIST_PATH,
        "pid_list_sha256": sha256_file(PID_LIST_PATH),
        "subset_size": len(pid_list),
        "subset_unique": len(subset),
        "num_queries": len(all_qids),
        "num_queries_with_qrels": len(qrels),
        "num_relevant_pids": len(relevant_pids),
        "relevant_pids_in_subset": len(relevant_pids & subset),
        "queries_fully_covered": fully,
        "queries_partially_covered": partial,
        "queries_not_covered": len(none_),
        "not_covered_query_ids": none_[:50],
    }
    with open(os.path.join(OUT_DIR, "subset_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    pids_path = os.path.join(OUT_DIR, "subset_pids.txt.gz")
    if not os.path.exists(pids_path):
        with gzip.open(pids_path, "wt") as f:
            f.write("\n".join(pid_list))

    for k, v in audit.items():
        if k != "not_covered_query_ids":
            print(f"{k}: {v}")
    if none_:
        print(f"WARNING: {len(none_)} queries have no relevant passage in the subset")


if __name__ == "__main__":
    main()
