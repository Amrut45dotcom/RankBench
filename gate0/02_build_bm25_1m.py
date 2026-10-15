"""
Gate 0.2 - build a BM25 index over exactly the 1M subset.

Same bm25s settings as scripts/bm25_baseline.py, so only the corpus changes.
Writes to bm25_index_1M/ (new directory; the existing bm25_index/ used by the
deployed app is NOT touched). Refuses to run if the directory exists.

Usage:
    python gate0/02_build_bm25_1m.py
"""
import os
import sys
import time

import bm25s
import ir_datasets
import numpy as np
from tqdm import tqdm

DATASET_NAME  = "msmarco-passage/dev/small"
PID_LIST_PATH = "embeddings/pid_list.npy"
INDEX_DIR     = "bm25_index_1M"


def main():
    if os.path.exists(INDEX_DIR):
        print(f"{INDEX_DIR}/ already exists. Not overwriting.")
        sys.exit(1)

    pid_list = [str(p) for p in np.load(PID_LIST_PATH)]
    wanted = set(pid_list)
    print(f"Subset size: {len(pid_list):,}")

    dataset = ir_datasets.load(DATASET_NAME)
    t0 = time.time()
    texts_by_pid = {}
    for doc in tqdm(dataset.docs_iter(), desc="Scanning corpus", total=8_841_823):
        if doc.doc_id in wanted:
            texts_by_pid[doc.doc_id] = doc.text
    missing = len(wanted) - len(texts_by_pid)
    print(f"Found {len(texts_by_pid):,} texts, missing {missing}  [{time.time()-t0:.1f}s]")
    assert missing == 0, "some subset pids were not found in the corpus"

    texts = [texts_by_pid[p] for p in pid_list]  # same order as FAISS rows
    del texts_by_pid

    t0 = time.time()
    tokenized = bm25s.tokenize(texts, show_progress=True)
    bm25 = bm25s.BM25(method="robertson", idf_method="robertson")
    bm25.index(tokenized, show_progress=True)
    bm25.save(INDEX_DIR)
    np.save(os.path.join(INDEX_DIR, "doc_ids.npy"), np.array(pid_list))
    print(f"Index built and saved to {INDEX_DIR}/  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
