"""Level 1, step 1: standalone answer(query). No server, no validator yet.

Run from the data folder:
    cd ~/RankBench
    python ~/rankbench_v2/level1/answer.py --n 5
    python ~/rankbench_v2/level1/answer.py --query "what is the capital of france"
"""
import os, sys, json, time, argparse
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
ap.add_argument("--n", type=int, default=5, help="how many queries from the split to run")
ap.add_argument("--split", default="dev_200.txt", help="file in splits/ to take queries from")
ap.add_argument("--query", default=None, help="free-text query instead of the split")
args = ap.parse_args()

with open(args.config) as f:
    L1 = yaml.safe_load(f)
with open(os.path.expanduser(L1["gate0_config"])) as f:
    G0 = yaml.safe_load(f)

# GPU must be chosen before torch is imported
os.environ["CUDA_VISIBLE_DEVICES"] = str(G0["hardware"]["gpu_id"])

import numpy as np
import faiss
import torch
import bm25s
import ir_datasets
from ranx import Run, fuse
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

ROOT = os.path.expanduser(G0["data_root"])
SPLITS = os.path.expanduser(G0["paths"]["splits_dir"])
OUT = os.path.join(ROOT, L1["out_dir"])
os.makedirs(OUT, exist_ok=True)
faiss.omp_set_num_threads(G0["hardware"]["faiss_threads"])
device = "cuda" if torch.cuda.is_available() else "cpu"
K = G0["topk"]


# ---------- data ----------
dataset = ir_datasets.load(G0["dataset"])
QTEXT = {q.query_id: q.text for q in dataset.queries_iter()}
QRELS = {}
for qr in dataset.qrels_iter():
    QRELS.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
docstore = dataset.docs_store()


def get_texts(pids):
    docs = docstore.get_many(list(set(pids)))
    return [docs[p].text for p in pids]


# ---------- indexes and models (same files and settings as the Gate 0 harness) ----------
pid_list = np.load(os.path.join(ROOT, G0["paths"]["pid_list"])).astype(str)
bm25_dir = os.path.join(ROOT, G0["paths"]["bm25_index"])
bm25_doc_ids = np.load(os.path.join(bm25_dir, "doc_ids.npy")).astype(str)
assert np.array_equal(pid_list, bm25_doc_ids), "pid_list and bm25 doc_ids differ"
index = faiss.read_index(os.path.join(ROOT, G0["paths"]["faiss_index"]))
assert index.ntotal == len(pid_list), "FAISS rows != pid_list"
bm25 = bm25s.BM25.load(bm25_dir)

_stem = None
if G0["bm25_tokenize"]["stemmer"]:
    import Stemmer
    _stem = Stemmer.Stemmer(G0["bm25_tokenize"]["stemmer"])

dense = SentenceTransformer(G0["models"]["dense"], device=device)
reranker = CrossEncoder(G0["models"]["reranker"], device=device, max_length=512)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ---------- retrieval: BM25 + dense -> RRF -> rerank (single query) ----------
def retrieve(query, n_out):
    texts = [query]
    try:
        toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                              stemmer=_stem, return_as="string", show_progress=False)
    except TypeError:  # older bm25s
        toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                              stemmer=_stem, return_ids=False, show_progress=False)
    idx, sc = bm25.retrieve(toks, k=K["bm25"], n_threads=G0["hardware"]["bm25_threads"],
                            show_progress=False)
    bm = {pid_list[i]: float(s) for i, s in zip(idx[0], sc[0])}

    emb = dense.encode([G0["models"]["query_prefix"] + query], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    D, I = index.search(emb, K["dense"])
    de = {pid_list[i]: float(d) for i, d in zip(I[0], D[0])}

    fused = fuse(runs=[Run({"q": bm}, name="bm25"), Run({"q": de}, name="dense")],
                 norm="min-max", method="rrf", params={"k": K["rrf_k"]}).to_dict()["q"]
    cands = sorted(fused.items(), key=lambda x: -x[1])[:K["rrf_out"]]
    pids = [p for p, _ in cands[:K["rerank_depth"]]]
    ptexts = get_texts(pids)
    scores = reranker.predict([(query, t) for t in ptexts],
                              batch_size=G0["batching"]["rerank_batch"], show_progress_bar=False)
    ranked = sorted(zip(pids, ptexts, scores), key=lambda x: -x[2])
    return [{"pid": p, "text": t, "score": float(s)} for p, t, s in ranked[:n_out]]


# ---------- generation ----------
SYSTEM_PROMPT = """You answer questions using only the numbered passages provided.
Rules:
1. Use only information stated in the passages. Do not use outside knowledge.
2. After every sentence of your answer, cite the supporting passage(s), like [P1] or [P1][P3].
3. Only cite passage numbers that appear in the list.
4. If the passages do not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT
5. Keep the answer short: 1 to 3 sentences."""


def build_user_message(query, passages):
    block = "\n\n".join(f"[P{i+1}] {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n\n{block}\n\nQuestion: {query}"


def call_llm(query, passages):
    cfg = L1["llm"]
    kwargs = dict(
        model=cfg["model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_user_message(query, passages)}],
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
    )
    try:
        resp = groq_client.chat.completions.create(
            **kwargs, reasoning_effort=cfg["reasoning_effort"], include_reasoning=False)
    except TypeError:  # older groq SDK without these arguments
        resp = groq_client.chat.completions.create(
            **kwargs, extra_body={"reasoning_effort": cfg["reasoning_effort"],
                                  "include_reasoning": False})
    return (resp.choices[0].message.content or "").strip()


def answer(query, qid=None):
    t0 = time.perf_counter()
    passages = retrieve(query, L1["n_passages"])
    t1 = time.perf_counter()
    text = call_llm(query, passages)
    t2 = time.perf_counter()
    rel = QRELS.get(qid, {}) if qid else {}
    for p in passages:
        p["is_gold"] = rel.get(p["pid"], 0) > 0 if qid else None
    return {
        "qid": qid, "query": query, "answer": text,
        "refused": "INSUFFICIENT_CONTEXT" in text,
        "passages": passages,
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": round((t2 - t1) * 1000, 1),
    }


def show(r):
    print("=" * 80)
    print(f"Q [{r['qid']}]: {r['query']}")
    for i, p in enumerate(r["passages"]):
        gold = "" if p["is_gold"] is None else ("  <-- relevant" if p["is_gold"] else "")
        print(f"  [P{i+1}] score={p['score']:.2f}{gold}  {p['text'][:140]}...")
    print(f"ANSWER: {r['answer']}")
    print(f"refused={r['refused']}  retrieval={r['retrieval_ms']}ms  generation={r['generation_ms']}ms")


if __name__ == "__main__":
    if args.query:
        jobs = [(None, args.query)]
    else:
        with open(os.path.join(SPLITS, args.split)) as f:
            qids = [l.strip() for l in f if l.strip()][:args.n]
        jobs = [(q, QTEXT[q]) for q in qids]

    out_path = os.path.join(OUT, "samples.jsonl")
    with open(out_path, "a") as fout:
        for qid, qtext in jobs:
            r = answer(qtext, qid)
            show(r)
            fout.write(json.dumps(r) + "\n")
    print(f"\nappended {len(jobs)} result(s) to {out_path}")
