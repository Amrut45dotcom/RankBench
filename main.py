import os
import time
import numpy as np
import faiss
import bm25s
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Run, fuse

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_PATH      = "faiss_index/bge_large_1M.index"
PID_LIST_PATH   = "embeddings/pid_list.npy"
BM25_INDEX_DIR  = "bm25_index"
CORPUS_PKL      = "corpus.npy"          # see note below
BGE_MODEL       = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE          = "cuda"
TOP_K_RETRIEVE  = 1000   # first-stage retrieval
TOP_K_RRF       = 100    # RRF cut before reranking
TOP_K_FINAL     = 5      # results returned per strategy

LATENCY_WINDOW  = 100    # rolling window size for /metrics

# ── Globals populated at startup ──────────────────────────────────────────────
state = {}

# ── Latency store ─────────────────────────────────────────────────────────────
# { strategy_name: deque of latency floats (ms) }
latency_store: dict[str, deque] = {
    "bm25":     deque(maxlen=LATENCY_WINDOW),
    "dense":    deque(maxlen=LATENCY_WINDOW),
    "rrf":      deque(maxlen=LATENCY_WINDOW),
    "reranker": deque(maxlen=LATENCY_WINDOW),
}


# ── Lifespan: load all models/indexes once at startup ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading BM25 index ...")
    bm25 = bm25s.BM25.load(BM25_INDEX_DIR, load_corpus=False)

    # We need doc_ids in same order as the BM25 index was built.
    # The original script built the index from list(corpus.keys()).
    # Store that mapping alongside the bm25_index as doc_ids.npy
    # (see README for one-time export command).
    doc_ids = np.load(os.path.join(BM25_INDEX_DIR, "doc_ids.npy"), allow_pickle=True)
    state["bm25"]    = bm25
    state["doc_ids"] = doc_ids
    print(f"       BM25 vocab size: {len(doc_ids):,}")

    print("[startup] Loading FAISS index ...")
    index    = faiss.read_index(INDEX_PATH)
    pid_list = np.load(PID_LIST_PATH)
    state["faiss"]    = index
    state["pid_list"] = pid_list
    print(f"       FAISS vectors: {index.ntotal:,}")

    print("[startup] Loading BGE model ...")
    bge = SentenceTransformer(BGE_MODEL, device=DEVICE)
    state["bge"] = bge

    print("[startup] Loading cross-encoder ...")
    ce = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
    state["ce"] = ce

    # corpus text needed by reranker
    # Load as dict {pid_str: text}.  Export once from ir_datasets — see README.
    print("[startup] Loading corpus texts ...")
    corpus = np.load("corpus.npy", allow_pickle=True).item()
    state["corpus"] = corpus
    print(f"       Corpus passages: {len(corpus):,}")

    print("[startup] All components loaded. Ready.")
    yield
    state.clear()


app = FastAPI(title="RankBench API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K_FINAL   # override per-request if needed


class Passage(BaseModel):
    pid:   str
    score: float
    text:  str


class StrategyResult(BaseModel):
    results:      list[Passage]
    latency_ms:   float


class QueryResponse(BaseModel):
    query:    str
    bm25:     StrategyResult
    dense:    StrategyResult
    rrf:      StrategyResult
    reranker: StrategyResult


# ── Helpers ───────────────────────────────────────────────────────────────────
def _passages_from_run(run_dict: dict[str, float], corpus: dict, top_k: int) -> list[Passage]:
    """Convert {pid: score} dict → list[Passage], looking up text from corpus."""
    sorted_items = sorted(run_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        Passage(pid=pid, score=score, text=corpus.get(pid, "[text not in corpus]"))
        for pid, score in sorted_items
    ]


def _run_bm25(query: str) -> tuple[dict, float]:
    t0 = time.time()
    tokens  = bm25s.tokenize([query])
    results, scores = state["bm25"].retrieve(tokens, k=TOP_K_RETRIEVE)
    doc_ids = state["doc_ids"]
    run_dict = {
        str(doc_ids[results[0][i]]): float(scores[0][i])
        for i in range(len(results[0]))
    }
    return run_dict, (time.time() - t0) * 1000


def _run_dense(query: str) -> tuple[dict, float]:
    t0 = time.time()
    q_vec = state["bge"].encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores, indices = state["faiss"].search(q_vec, TOP_K_RETRIEVE)
    pid_list = state["pid_list"]
    run_dict = {
        str(pid_list[indices[0][i]]): float(scores[0][i])
        for i in range(TOP_K_RETRIEVE)
    }
    return run_dict, (time.time() - t0) * 1000


def _run_rrf(bm25_dict: dict, dense_dict: dict) -> tuple[dict, float]:
    t0 = time.time()
    # ranx fuse expects Run objects
    # Use a dummy qid since we're doing single-query fusion
    QID = "q0"
    bm25_run  = Run({QID: bm25_dict},  name="bm25")
    dense_run = Run({QID: dense_dict}, name="dense")
    fused = fuse(runs=[bm25_run, dense_run], norm="min-max", method="rrf")
    run_dict = dict(fused.run[QID])
    return run_dict, (time.time() - t0) * 1000


def _run_reranker(query: str, rrf_dict: dict) -> tuple[dict, float]:
    t0 = time.time()
    corpus = state["corpus"]
    candidate_pids = sorted(rrf_dict, key=rrf_dict.get, reverse=True)[:TOP_K_RRF]
    valid_pids = [p for p in candidate_pids if p in corpus]
    pairs  = [(query, corpus[p]) for p in valid_pids]
    scores = state["ce"].predict(pairs, batch_size=64, show_progress_bar=False)
    scored = sorted(zip(valid_pids, scores.tolist()), key=lambda x: x[1], reverse=True)
    run_dict = {pid: float(score) for pid, score in scored}
    return run_dict, (time.time() - t0) * 1000


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k  = min(req.top_k, TOP_K_FINAL)
    corpus = state["corpus"]

    # ── BM25 ──
    bm25_dict, bm25_ms = _run_bm25(req.query)
    latency_store["bm25"].append(bm25_ms)

    # ── Dense ──
    dense_dict, dense_ms = _run_dense(req.query)
    latency_store["dense"].append(dense_ms)

    # ── RRF (uses BM25 + Dense results computed above) ──
    rrf_dict, rrf_ms = _run_rrf(bm25_dict, dense_dict)
    latency_store["rrf"].append(rrf_ms)

    # ── Reranker ──
    reranker_dict, reranker_ms = _run_reranker(req.query, rrf_dict)
    latency_store["reranker"].append(reranker_ms)

    return QueryResponse(
        query=req.query,
        bm25=StrategyResult(
            results=_passages_from_run(bm25_dict, corpus, top_k),
            latency_ms=round(bm25_ms, 2),
        ),
        dense=StrategyResult(
            results=_passages_from_run(dense_dict, corpus, top_k),
            latency_ms=round(dense_ms, 2),
        ),
        rrf=StrategyResult(
            results=_passages_from_run(rrf_dict, corpus, top_k),
            latency_ms=round(rrf_ms, 2),
        ),
        reranker=StrategyResult(
            results=_passages_from_run(reranker_dict, corpus, top_k),
            latency_ms=round(reranker_ms, 2),
        ),
    )


@app.get("/metrics")
def metrics():
    """Return p50 / p95 latency (ms) per strategy over last 100 requests."""
    out = {}
    for strategy, dq in latency_store.items():
        if not dq:
            out[strategy] = {"p50": None, "p95": None, "n": 0}
            continue
        arr = np.array(dq)
        out[strategy] = {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "n":   len(arr),
        }
    return out


@app.get("/health")
def health():
    return {"status": "ok", "components": list(state.keys())}
