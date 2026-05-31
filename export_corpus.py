"""
Run this ONCE on the server before starting the FastAPI app.
Exports two files that main.py needs at startup:

  corpus.npy          — dict {pid_str: passage_text} for all 8.8M passages
  bm25_index/doc_ids.npy — ordered doc_id list matching the BM25 index

Usage:
    python export_corpus.py
"""
import time
import numpy as np
import ir_datasets
from tqdm import tqdm

DATASET_NAME = "msmarco-passage/dev/small"

print("Loading ir_datasets corpus (this takes a few minutes)...")
dataset = ir_datasets.load(DATASET_NAME)

t0 = time.time()
corpus  = {}
doc_ids = []

for doc in tqdm(dataset.docs_iter(), desc="Loading", total=8_841_823):
    corpus[doc.doc_id]  = doc.text
    doc_ids.append(doc.doc_id)

print(f"Loaded {len(corpus):,} passages in {time.time()-t0:.1f}s")

print("Saving corpus.npy ...")
np.save("corpus.npy", corpus)

print("Saving bm25_index/doc_ids.npy ...")
np.save("bm25_index/doc_ids.npy", np.array(doc_ids))

print("Done.")
print("  corpus.npy             — for reranker text lookup")
print("  bm25_index/doc_ids.npy — for BM25 result mapping")
