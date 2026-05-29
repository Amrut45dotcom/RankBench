import os
import numpy as np
import faiss
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "embeddings"
INDEX_PATH     = "faiss_index/bge_large_1M.index"
os.makedirs("faiss_index", exist_ok=True)

# ── 1. Load embeddings ───────────────────────────────────────────────────────
print("[1/3] Loading embeddings ...")
embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "bge_large_1M.npy"))
print(f"     Shape: {embeddings.shape}")

# ensure float32
embeddings = embeddings.astype(np.float32)

# ── 2. Build FAISS index ─────────────────────────────────────────────────────
print("[2/3] Building FAISS index ...")
dim   = embeddings.shape[1]  # 1024

# IndexFlatIP = exact inner product search (cosine equiv since vectors are normalized)
index = faiss.IndexFlatIP(dim)

# add in batches to show progress
BATCH = 100_000
for start in tqdm(range(0, len(embeddings), BATCH), desc="Adding vectors"):
    end = min(start + BATCH, len(embeddings))
    index.add(embeddings[start:end])

print(f"     Total vectors in index: {index.ntotal:,}")

# ── 3. Save index ────────────────────────────────────────────────────────────
print("[3/3] Saving FAISS index ...")
faiss.write_index(index, INDEX_PATH)
print(f"     Saved to {INDEX_PATH}")
print("Done.")