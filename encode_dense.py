import os
import time
import random
import numpy as np
import ir_datasets
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_NAME   = "msmarco-passage/dev/small"
SUBSET_SIZE    = 1_000_000
GPU_ID         = 1            
BATCH_SIZE     = 512
EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── 1. Load qrels pids ───────────────────────────────────────────────────────
print("[1/4] Loading qrels ...")
dataset = ir_datasets.load(DATASET_NAME)

qrels_pids = set()
for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels"):
    qrels_pids.add(qrel.doc_id)

print(f"     Qrels pids: {len(qrels_pids):,}")

# ── 2. Filter corpus to 1M ───────────────────────────────────────────────────
print("[2/4] Filtering corpus to 1M passages ...")
t0 = time.time()

guaranteed = {}
pool       = []

for doc in tqdm(dataset.docs_iter(), desc="Scanning corpus", total=8_841_823):
    if doc.doc_id in qrels_pids:
        guaranteed[doc.doc_id] = doc.text
    else:
        pool.append((doc.doc_id, doc.text))

print(f"     Guaranteed (qrels): {len(guaranteed):,}")
print(f"     Pool size: {len(pool):,}")

# sample remaining to reach 1M
n_sample = SUBSET_SIZE - len(guaranteed)
sampled  = random.sample(pool, n_sample)
del pool  # free memory

subset = {**guaranteed}
for pid, text in sampled:
    subset[pid] = text

print(f"     Final subset size: {len(subset):,}  [{time.time()-t0:.1f}s]")

# save pid list for later (needed to map embedding index → pid)
pid_list = list(subset.keys())
np.save(os.path.join(EMBEDDINGS_DIR, "pid_list.npy"), np.array(pid_list))
print(f"     pid_list saved.")

# ── 3. Encode with BGE ───────────────────────────────────────────────────────
print(f"[3/4] Encoding with BGE on GPU {GPU_ID} ...")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

texts = [subset[pid] for pid in pid_list]

t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"     Encoded {len(texts):,} passages  [{time.time()-t0:.1f}s]")

# ── 4. Save embeddings ───────────────────────────────────────────────────────
print("[4/4] Saving embeddings ...")
np.save(os.path.join(EMBEDDINGS_DIR, "bge_large_1M.npy"), embeddings)
print(f"     Shape: {embeddings.shape}")
print("Done.")