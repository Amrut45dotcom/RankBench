import numpy as np, faiss, json, hashlib, os, sys

EMB_PATH = "embeddings/bge_large_1M.npy"   
OUT_DIR = "level2"
K, SEED = 4, 42
LEVEL_P = [0.30, 0.40, 0.30]             # public, internal, restricted
TRAIN_N = 200_000

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(f"{OUT_DIR}/tenant.npy"):
    sys.exit("tenant.npy exists, refusing to overwrite")

emb = np.load(EMB_PATH, mmap_mode="r")
pids = np.load("embeddings/pid_list.npy")
assert emb.shape[0] == len(pids) == 1_000_000

rng = np.random.default_rng(SEED)
train_idx = np.sort(rng.choice(len(pids), TRAIN_N, replace=False))
train = np.ascontiguousarray(emb[train_idx], dtype="float32")

km = faiss.Kmeans(train.shape[1], K, niter=25, seed=SEED, spherical=True, verbose=True)
km.train(train)

tenant = np.empty(len(pids), dtype=np.uint8)
for s in range(0, len(pids), 100_000):
    chunk = np.ascontiguousarray(emb[s:s+100_000], dtype="float32")
    tenant[s:s+len(chunk)] = km.index.search(chunk, 1)[1][:, 0]

level = rng.choice(3, size=len(pids), p=LEVEL_P).astype(np.uint8)

np.save(f"{OUT_DIR}/tenant.npy", tenant)
np.save(f"{OUT_DIR}/level.npy", level)
meta = {
  "k": K, "seed": SEED, "level_p": LEVEL_P, "train_n": TRAIN_N,
  "pid_list_sha256": sha("embeddings/pid_list.npy"),
  "tenant_sha256": sha(f"{OUT_DIR}/tenant.npy"),
  "level_sha256": sha(f"{OUT_DIR}/level.npy"),
  "tenant_sizes": np.bincount(tenant, minlength=K).tolist(),
  "level_sizes": np.bincount(level, minlength=3).tolist(),
  "cross_tab": [[int(((tenant==t)&(level==l)).sum()) for l in range(3)] for t in range(K)],
}
json.dump(meta, open(f"{OUT_DIR}/tenants_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))