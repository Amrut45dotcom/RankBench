import os, sys, json, time, random, argparse, subprocess, platform

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
ap.add_argument("--mode", choices=["check", "quality", "latency", "all"], required=True)
args = ap.parse_args()

with open(args.config) as f:
    CFG = yaml.safe_load(f)

# GPU must be chosen before torch is imported
os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG["hardware"]["gpu_id"])

import numpy as np
import faiss
import torch
import bm25s
import ir_datasets
from ranx import Qrels, Run, evaluate, fuse
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = os.path.expanduser(CFG["data_root"])
REPO = os.path.expanduser(CFG["repo_root"])
SPLITS = os.path.expanduser(CFG["paths"]["splits_dir"])
RES = os.path.join(ROOT, CFG["paths"]["results_dir"])
os.makedirs(os.path.join(RES, "runs_matched"), exist_ok=True)
faiss.omp_set_num_threads(CFG["hardware"]["faiss_threads"])
METRICS = ["ndcg@10", "mrr@10", "recall@100"]
dev_device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- data ----------
def read_ids(name):
    with open(os.path.join(SPLITS, name)) as f:
        return [l.strip() for l in f if l.strip()]

dataset = ir_datasets.load(CFG["dataset"])
ALL_QTEXT = {q.query_id: q.text for q in dataset.queries_iter()}
ALL_QRELS = {}
for qr in dataset.qrels_iter():
    ALL_QRELS.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
docstore = dataset.docs_store()
_text_cache = {}

def get_texts(pids):
    missing = [p for p in set(pids) if p not in _text_cache]
    if missing:
        for d in docstore.get_many(missing).values():
            _text_cache[d.doc_id] = d.text
    return [_text_cache[p] for p in pids]


# ---------- indexes and models ----------
pid_list = np.load(os.path.join(ROOT, CFG["paths"]["pid_list"])).astype(str)
bm25_dir = os.path.join(ROOT, CFG["paths"]["bm25_index"])
bm25_doc_ids = np.load(os.path.join(bm25_dir, "doc_ids.npy")).astype(str)
assert len(pid_list) == len(bm25_doc_ids) == 1_000_000, "unexpected subset size"
assert np.array_equal(pid_list, bm25_doc_ids), "pid_list and bm25 doc_ids differ"

index = faiss.read_index(os.path.join(ROOT, CFG["paths"]["faiss_index"]))
assert index.ntotal == len(pid_list), "FAISS rows != pid_list"

bm25 = bm25s.BM25.load(bm25_dir)
_stem = None
if CFG["bm25_tokenize"]["stemmer"]:
    import Stemmer
    _stem = Stemmer.Stemmer(CFG["bm25_tokenize"]["stemmer"])

_dense_model = None
_reranker = None

def dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(CFG["models"]["dense"], device=dev_device)
    return _dense_model

def reranker():
    global _reranker
    if _reranker is None:
                _reranker = CrossEncoder(CFG["models"]["reranker"], device=dev_device, max_length=512)
    return _reranker


# ---------- retrieval primitives (all take lists of qids) ----------
def bm25_search(qids, k):
    texts = [ALL_QTEXT[q] for q in qids]
    try:
        toks = bm25s.tokenize(texts, stopwords=CFG["bm25_tokenize"]["stopwords"],
                              stemmer=_stem, return_as="string", show_progress=False)
    except TypeError:   # older bm25s without return_as
        toks = bm25s.tokenize(texts, stopwords=CFG["bm25_tokenize"]["stopwords"],
                              stemmer=_stem, return_ids=False, show_progress=False)
    idx, sc = bm25.retrieve(toks, k=k, n_threads=CFG["hardware"]["bm25_threads"],
                            show_progress=False)
    return {q: [(pid_list[i], float(s)) for i, s in zip(idx[r], sc[r])]
            for r, q in enumerate(qids)}

def dense_search(qids, k):
    prefix = CFG["models"]["query_prefix"]
    texts = [prefix + ALL_QTEXT[q] for q in qids]
    emb = dense_model().encode(texts, batch_size=CFG["batching"]["encode_batch"],
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float32)
    out = {}
    sb = CFG["batching"]["search_batch"]
    for s in range(0, len(qids), sb):
        D, I = index.search(emb[s:s + sb], k)
        for r, q in enumerate(qids[s:s + sb]):
            out[q] = [(pid_list[i], float(d)) for i, d in zip(I[r], D[r])]
    return out

def rrf_fuse(bm, de, k_const, out_k):
    runs = [
        Run({q: {p: s for p, s in lst} for q, lst in bm.items()}, name="bm25"),
        Run({q: {p: s for p, s in lst} for q, lst in de.items()}, name="dense"),
    ]
    fused = fuse(runs=runs, norm="min-max", method="rrf", params={"k": k_const})
    return {q: sorted(docs.items(), key=lambda x: -x[1])[:out_k]
            for q, docs in fused.to_dict().items()}

def rerank(qids, cands, depth):
    pairs, owners = [], []
    for q in qids:
        for pid, _ in cands[q][:depth]:
            owners.append((q, pid))
    texts = get_texts([p for _, p in owners])
    pairs = [(ALL_QTEXT[q], t) for (q, _), t in zip(owners, texts)]
    scores = reranker().predict(pairs, batch_size=CFG["batching"]["rerank_batch"],
                                show_progress_bar=False)
    out = {q: [] for q in qids}
    for (q, pid), s in zip(owners, scores):
        out[q].append((pid, float(s)))
    for q in out:
        out[q].sort(key=lambda x: -x[1])
    return out


# ---------- evaluation ----------
def to_run(res, name):
    return Run({q: {p: s for p, s in lst} for q, lst in res.items()}, name=name)

def evaluate_run(res, qids, name):
    qrels = Qrels({q: ALL_QRELS[q] for q in qids})
    run = to_run(res, name)
    run.save(os.path.join(RES, "runs_matched", f"{name}.json"))
    return evaluate(qrels, run, METRICS)

def fmt(m):
    return {k: round(float(v), 4) for k, v in m.items()}


# ---------- modes ----------
def mode_check():
    qids = read_ids("dev_200.txt") + read_ids("final.txt")
    assert len(qids) == 6980, len(qids)
    t = time.time()
    res = dense_search(qids, CFG["topk"]["dense"])
    m = fmt(evaluate(Qrels({q: ALL_QRELS[q] for q in qids}), to_run(res, "dense_check"), METRICS))
    print(f"Dense on all {len(qids)} queries ({time.time()-t:.0f}s): {m}")
    print("Expected ndcg@10 about 0.7118")
    return m

def mode_quality():
    qids = read_ids("final.txt")
    print(f"{len(qids)} final queries")
    out = {}
    t = time.time(); bm = bm25_search(qids, CFG["topk"]["bm25"]); print(f"bm25 {time.time()-t:.0f}s")
    t = time.time(); de = dense_search(qids, CFG["topk"]["dense"]); print(f"dense {time.time()-t:.0f}s")
    fu = rrf_fuse(bm, de, CFG["topk"]["rrf_k"], CFG["topk"]["rrf_out"])
    t = time.time(); rr = rerank(qids, fu, CFG["topk"]["rerank_depth"]); print(f"rerank {time.time()-t:.0f}s")
    for name, res in [("bm25", bm), ("dense", de), ("rrf", fu), ("reranker", rr)]:
        out[name] = fmt(evaluate_run(res, qids, name))
        print(name, out[name])
    return out

def pct(a, p):
    return float(np.percentile(a, p))

def mode_latency():
    final = read_ids("final.txt")
    sample = random.Random(CFG["seed"]).sample(final, CFG["latency"]["sample_size"])
    k = CFG["topk"]
    methods = {
        "bm25":     lambda q: bm25_search([q], k["bm25"]),
        "dense":    lambda q: dense_search([q], k["dense"]),
        "rrf":      lambda q: rrf_fuse(bm25_search([q], k["bm25"]), dense_search([q], k["dense"]),
                                       k["rrf_k"], k["rrf_out"]),
        "reranker": lambda q: rerank([q], rrf_fuse(bm25_search([q], k["bm25"]),
                                                   dense_search([q], k["dense"]),
                                                   k["rrf_k"], k["rrf_out"]), k["rerank_depth"]),
    }
    L = CFG["latency"]
    out = {}
    for name, fn in methods.items():
        per_repeat = []
        for rep in range(L["warmup_repeats"] + L["repeats"]):
            times = []
            for q in sample:
                t = time.perf_counter(); fn(q)
                if dev_device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t) * 1000)
            if rep >= L["warmup_repeats"]:
                per_repeat.append({"p50": pct(times, 50), "p95": pct(times, 95)})
        out[name] = {
            "p50_ms": round(float(np.median([r["p50"] for r in per_repeat])), 2),
            "p95_ms": round(float(np.median([r["p95"] for r in per_repeat])), 2),
            "per_repeat": per_repeat,
        }
        print(name, out[name]["p50_ms"], out[name]["p95_ms"])
    return out

def env_info():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, cwd=REPO).strip()
        except Exception as e:
            return f"unavailable: {e}"
    return {
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": sh("git status --porcelain") != "",
        "python": platform.python_version(),
        "torch": torch.__version__, "faiss": faiss.__version__,
        "loadavg": os.getloadavg(),
        "gpu": sh("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader"),
        "gpu_id_used": CFG["hardware"]["gpu_id"],
    }

if args.mode == "check":
    mode_check()
elif args.mode == "quality":
    mode_quality()
elif args.mode == "latency":
    print(json.dumps(mode_latency(), indent=2))
else:
    check = mode_check()
    env_before = env_info()
    quality = mode_quality()
    lat = mode_latency()
    result = {"config": CFG, "env_before": env_before, "env_after": env_info(),
              "dense_sanity_all_6980": check, "quality_final_6780": quality, "latency": lat}
    path = os.path.join(RES, "baseline_matched.json")
    assert not os.path.exists(path), "baseline_matched.json already exists; refusing to overwrite"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("saved", path)