import os
import time
import threading
import numpy as np
import yaml
from validator import validate, clean_for_display

SYSTEM_PROMPT = """You answer questions using only the numbered passages provided.
Rules:
1. Use only information stated in the passages. Do not use outside knowledge.
2. After every sentence of your answer, cite the supporting passage(s) using plain square brackets exactly like [P1] or [P1][P3]. Never use any other bracket characters.
3. Only cite passage numbers that appear in the list.
4. If the passages do not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT
5. Keep the answer short: 1 to 3 sentences."""

RETRY_NOTE = ("\n\nYour previous reply was rejected by an automatic format check ({problems}). "
              "Reply again following the rules exactly: cite every sentence with plain [P#] brackets, "
              "cite only the listed passages, or reply with exactly INSUFFICIENT_CONTEXT.")
REFUSAL_TEXT = "I can't answer this from the available documents."


class Pipeline:
    def __init__(self, G0, L1, *, pid_list, index, bm25, dense, reranker, groq_client,
                 get_texts, stemmer=None, qrels=None):
        self.G0, self.L1 = G0, L1
        self.pid_list, self.index, self.bm25 = pid_list, index, bm25
        self.dense, self.reranker, self.groq = dense, reranker, groq_client
        self.get_texts, self.stemmer = get_texts, stemmer
        self.qrels = qrels or {}
        self.qtext = {}                      # filled by load_pipeline (CLI only)
        self._lock = threading.Lock()        # one retrieval at a time: FAISS and GPU models are shared

    # ---------- retrieval: BM25 + dense -> RRF -> rerank (single query) ----------
    def retrieve(self, query, n_out):
        import bm25s                         # imported here, not at the top, so the CLI can set
        from ranx import Run, fuse           # CUDA_VISIBLE_DEVICES before anything heavy loads
        G0 = self.G0
        K = G0["topk"]
        with self._lock:
            texts = [query]
            try:
                toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                                      stemmer=self.stemmer, return_as="string", show_progress=False)
            except TypeError:  # older bm25s
                toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                                      stemmer=self.stemmer, return_ids=False, show_progress=False)
            idx, sc = self.bm25.retrieve(toks, k=K["bm25"], n_threads=G0["hardware"]["bm25_threads"],
                                         show_progress=False)
            bm = {self.pid_list[i]: float(s) for i, s in zip(idx[0], sc[0])}

            emb = self.dense.encode([G0["models"]["query_prefix"] + query], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            D, I = self.index.search(emb, K["dense"])
            de = {self.pid_list[i]: float(d) for i, d in zip(I[0], D[0])}

            fused = fuse(runs=[Run({"q": bm}, name="bm25"), Run({"q": de}, name="dense")],
                         norm="min-max", method="rrf", params={"k": K["rrf_k"]}).to_dict()["q"]
            cands = sorted(fused.items(), key=lambda x: -x[1])[:K["rrf_out"]]
            pids = [p for p, _ in cands[:K["rerank_depth"]]]
            ptexts = self.get_texts(pids)
            scores = self.reranker.predict([(query, t) for t in ptexts],
                                           batch_size=G0["batching"]["rerank_batch"],
                                           show_progress_bar=False)
            ranked = sorted(zip(pids, ptexts, scores), key=lambda x: -x[2])
            return [{"pid": p, "text": t, "score": float(s)} for p, t, s in ranked[:n_out]]

    # ---------- generation ----------
    @staticmethod
    def build_user_message(query, passages, reminder=None):
        block = "\n\n".join(f"[P{i+1}] {p['text']}" for i, p in enumerate(passages))
        msg = f"Passages:\n\n{block}\n\nQuestion: {query}"
        return msg + reminder if reminder else msg

    def call_llm(self, query, passages, reminder=None):
        cfg = self.L1["llm"]
        kwargs = dict(
            model=cfg["model"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": self.build_user_message(query, passages, reminder)}],
            temperature=cfg["temperature"],
            max_completion_tokens=cfg["max_completion_tokens"],
        )
        try:
            resp = self.groq.chat.completions.create(
                **kwargs, reasoning_effort=cfg["reasoning_effort"], include_reasoning=False)
        except TypeError:  # older groq SDK without these arguments
            resp = self.groq.chat.completions.create(
                **kwargs, extra_body={"reasoning_effort": cfg["reasoning_effort"],
                                      "include_reasoning": False})
        return (resp.choices[0].message.content or "").strip()

    def answer(self, query, qid=None):
        t0 = time.perf_counter()
        passages = self.retrieve(query, self.L1["n_passages"])
        t1 = time.perf_counter()
        raw = self.call_llm(query, passages)
        v = validate(raw, len(passages))
        attempts = 1
        attempt_log = [{"raw": raw, "status": v["status"], "reasons": v["reasons"]}]
        if v["status"] == "invalid":   # one retry with a note about what was wrong
            raw = self.call_llm(query, passages, RETRY_NOTE.format(problems=", ".join(v["reasons"])))
            v = validate(raw, len(passages))
            attempts = 2
            attempt_log.append({"raw": raw, "status": v["status"], "reasons": v["reasons"]})
        t2 = time.perf_counter()
        if v["status"] == "ok":
            status, text = "answered", clean_for_display(raw).strip()
        elif v["status"] == "refused":
            status, text = "refused_by_model", REFUSAL_TEXT
        else:
            status, text = "refused_validation_failed", REFUSAL_TEXT
        rel = self.qrels.get(qid, {}) if qid else {}
        for p in passages:
            p["is_gold"] = rel.get(p["pid"], 0) > 0 if qid else None
        return {
            "qid": qid, "query": query, "answer": text,
            "status": status, "refused": status != "answered",
            "reasons": v["reasons"], "attempts": attempts,
            "cited": v["cited"], "raw_answer": raw, "attempt_log": attempt_log,
            "passages": passages,
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "generation_ms": round((t2 - t1) * 1000, 1),
        }


def _read_configs(config_path):
    with open(config_path) as f:
        L1 = yaml.safe_load(f)
    with open(os.path.expanduser(L1["gate0_config"])) as f:
        G0 = yaml.safe_load(f)
    return L1, G0


def build_pipeline(config_path, *, index, pid_list, dense, reranker, get_texts, qrels=None):
    """Build a Pipeline around models the caller already loaded.

    Loads only what is not shared: the matched 1M BM25 index and the Groq client.
    The asserts are a guard: if the BM25 index is not the 1M one built in Gate 0, this fails loudly.
    """
    import bm25s
    from groq import Groq
    L1, G0 = _read_configs(config_path)
    root = os.path.expanduser(G0["data_root"])
    pid_list = np.asarray(pid_list).astype(str)
    bm25_dir = os.path.join(root, G0["paths"]["bm25_index"])
    bm25_doc_ids = np.load(os.path.join(bm25_dir, "doc_ids.npy")).astype(str)
    assert np.array_equal(pid_list, bm25_doc_ids), "pid_list and bm25 doc_ids differ"
    assert index.ntotal == len(pid_list), "FAISS rows != pid_list"
    bm25 = bm25s.BM25.load(bm25_dir)
    stem = None
    if G0["bm25_tokenize"]["stemmer"]:
        import Stemmer
        stem = Stemmer.Stemmer(G0["bm25_tokenize"]["stemmer"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return Pipeline(G0, L1, pid_list=pid_list, index=index, bm25=bm25, dense=dense,
                    reranker=reranker, groq_client=groq_client, get_texts=get_texts,
                    stemmer=stem, qrels=qrels)


def load_pipeline(config_path):
    """Standalone: load every model and index (what the old answer.py did at import time)."""
    L1, G0 = _read_configs(config_path)
    # GPU must be chosen before torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = str(G0["hardware"]["gpu_id"])
    import faiss
    import torch
    import ir_datasets
    from sentence_transformers import SentenceTransformer, CrossEncoder

    root = os.path.expanduser(G0["data_root"])
    faiss.omp_set_num_threads(G0["hardware"]["faiss_threads"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ir_datasets.load(G0["dataset"])
    qtext = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels = {}
    for qr in dataset.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
    docstore = dataset.docs_store()

    def get_texts(pids):
        docs = docstore.get_many(list(set(pids)))
        return [docs[p].text for p in pids]

    pid_list = np.load(os.path.join(root, G0["paths"]["pid_list"]))
    index = faiss.read_index(os.path.join(root, G0["paths"]["faiss_index"]))
    dense = SentenceTransformer(G0["models"]["dense"], device=device)
    reranker = CrossEncoder(G0["models"]["reranker"], device=device, max_length=512)
    pipe = build_pipeline(config_path, index=index, pid_list=pid_list, dense=dense,
                          reranker=reranker, get_texts=get_texts, qrels=qrels)
    pipe.qtext = qtext
    return pipe