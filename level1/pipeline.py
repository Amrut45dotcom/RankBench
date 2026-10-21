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

# Level 2: permission strategies
#   none       no filtering (Level 1 behaviour, and the leak-test control)
#   post       retrieve top-k, drop forbidden candidates (may return fewer than k)
#   oversample retrieve 10x k, drop forbidden candidates, cut back to k
#   pre        restrict the search to allowed passages from the start
STRATEGIES = ("none", "post", "oversample", "pre")
OVERSAMPLE = 10


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
        # Level 2 permission state
        self.mask_fn = None                  # maps a user to a bool mask over the 1M rows
        self._masks = {}                     # user -> bool mask
        self._sel = {}                       # user -> (SearchParameters, selector, packed bits)
        self._pid_idx_map = None
        self.audit = None                    # set by set_audit(); an AuditLog or None

    # ---------- Level 2: permission helpers ----------
    def set_permissions(self, mask_fn):
        """mask_fn(user) -> bool array of length 1,000,000 in pid_list order."""
        self.mask_fn = mask_fn
        self._masks.clear()
        self._sel.clear()

    def set_audit(self, audit_logger):
        """audit_logger: an AuditLog (level2/audit.py) or None to switch logging off."""
        self.audit = audit_logger

    def _get_mask(self, user):
        if user not in self._masks:
            m = np.asarray(self.mask_fn(user))
            assert m.dtype == bool and m.shape[0] == len(self.pid_list), "bad permission mask"
            self._masks[user] = m
        return self._masks[user]

    def _get_selector(self, user, mask):
        import faiss
        if user not in self._sel:
            packed = np.packbits(mask, bitorder="little")
            sel = faiss.IDSelectorBitmap(len(mask), faiss.swig_ptr(packed))
            params = faiss.SearchParameters()
            params.sel = sel
            # keep all three alive, or FAISS reads freed memory
            self._sel[user] = (params, sel, packed)
        return self._sel[user][0]

    def _pid_idx(self, pid):
        if self._pid_idx_map is None:
            self._pid_idx_map = {p: i for i, p in enumerate(self.pid_list)}
        return self._pid_idx_map[pid]

    @staticmethod
    def _bm25_pre(bm25, tokens, mask, k):
        """BM25 top-k restricted to allowed rows: score everything, mask, take top-k."""
        k = min(k, int(mask.sum()))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        s = np.where(mask, np.asarray(bm25.get_scores(tokens), dtype=np.float32), -np.inf)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]

    # ---------- retrieval: BM25 + dense -> RRF -> rerank (single query) ----------
    def retrieve(self, query, n_out, user=None, strategy="none"):
        return self.retrieve_traced(query, n_out, user, strategy)[0]

    def retrieve_traced(self, query, n_out, user=None, strategy="none", rerank=True):
        """Returns (passages, trace). trace lists the pids at each stage (for audit log and leak tests).

        rerank=False stops after RRF fusion (used for the retrieval-only comparison): passages then
        carry fusion scores and no text, and nothing is sent to the reranker."""
        import bm25s                         # imported here, not at the top, so the CLI can set
        from ranx import Run, fuse           # CUDA_VISIBLE_DEVICES before anything heavy loads
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        G0 = self.G0
        K = G0["topk"]
        mask = None
        if strategy != "none":
            if user is None or self.mask_fn is None:
                raise ValueError("a user and set_permissions() are required for filtered strategies")
            mask = self._get_mask(user)
        mult = OVERSAMPLE if strategy == "oversample" else 1
        with self._lock:
            t_start = time.perf_counter()
            texts = [query]
            try:
                toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                                      stemmer=self.stemmer, return_as="string", show_progress=False)
            except TypeError:  # older bm25s
                toks = bm25s.tokenize(texts, stopwords=G0["bm25_tokenize"]["stopwords"],
                                      stemmer=self.stemmer, return_ids=False, show_progress=False)

            # ---- BM25 candidates (filtered BEFORE fusion) ----
            if strategy == "pre":
                b_idx, b_sc = self._bm25_pre(self.bm25, toks[0], mask, K["bm25"])
            else:
                idx, sc = self.bm25.retrieve(toks, k=K["bm25"] * mult,
                                             n_threads=G0["hardware"]["bm25_threads"],
                                             show_progress=False)
                b_idx, b_sc = np.asarray(idx[0]), np.asarray(sc[0])
                if mask is not None:
                    keep = mask[b_idx]
                    b_idx, b_sc = b_idx[keep][:K["bm25"]], b_sc[keep][:K["bm25"]]
            bm = {self.pid_list[i]: float(s) for i, s in zip(b_idx, b_sc)}

            # ---- dense candidates (filtered BEFORE fusion) ----
            emb = self.dense.encode([G0["models"]["query_prefix"] + query], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            if strategy == "pre":
                D, I = self.index.search(emb, K["dense"], params=self._get_selector(user, mask))
                d_idx, d_sc = I[0], D[0]
                keep = d_idx >= 0            # FAISS pads with -1 when fewer than k are allowed
                d_idx, d_sc = d_idx[keep], d_sc[keep]
            else:
                D, I = self.index.search(emb, K["dense"] * mult)
                d_idx, d_sc = I[0], D[0]
                if mask is not None:
                    keep = mask[d_idx]
                    d_idx, d_sc = d_idx[keep][:K["dense"]], d_sc[keep][:K["dense"]]
            de = {self.pid_list[i]: float(d) for i, d in zip(d_idx, d_sc)}

            trace = {"bm25_ids": list(bm), "dense_ids": list(de)}
            runs = [Run({"q": r}, name=n) for n, r in (("bm25", bm), ("dense", de)) if r]
            if not runs:
                trace.update(fused_ids=[], rerank_input_ids=[], final_ids=[], ms_retrieval=0.0, ms_rerank=0.0)
                return [], trace

            fused = fuse(runs=runs, norm="min-max", method="rrf", params={"k": K["rrf_k"]}).to_dict()["q"]
            cands = sorted(fused.items(), key=lambda x: -x[1])[:K["rrf_out"]]
            pids = [p for p, _ in cands[:K["rerank_depth"]]]
            if mask is not None:             # last line of defence: nothing forbidden reaches the reranker
                bad = [p for p in pids if not mask[self._pid_idx(p)]]
                assert not bad, f"LEAK before rerank: {bad[:3]}"
            t_fused = time.perf_counter()
            trace["fused_ids"] = pids        # RRF output in fusion order (retrieval-only metrics use this)
            if not rerank:
                out = [{"pid": p, "text": None, "score": float(s)} for p, s in cands[:n_out]]
                trace.update(rerank_input_ids=pids, final_ids=[o["pid"] for o in out],
                             ms_retrieval=(t_fused - t_start) * 1000, ms_rerank=0.0)
                return out, trace
            ptexts = self.get_texts(pids)
            scores = self.reranker.predict([(query, t) for t in ptexts],
                                           batch_size=G0["batching"]["rerank_batch"],
                                           show_progress_bar=False)
            ranked = sorted(zip(pids, ptexts, scores), key=lambda x: -x[2])
            out = [{"pid": p, "text": t, "score": float(s)} for p, t, s in ranked[:n_out]]
            trace.update(rerank_input_ids=pids, final_ids=[o["pid"] for o in out],
                         ms_retrieval=(t_fused - t_start) * 1000,
                         ms_rerank=(time.perf_counter() - t_fused) * 1000)
            return out, trace

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

    def answer(self, query, qid=None, user=None, strategy="none"):
        t0 = time.perf_counter()
        try:
            res = self._answer(query, qid, user, strategy)
        except Exception as e:      # a failed request (for example the leak assert) is logged too
            if self.audit is not None:
                self.audit.log(user, strategy, query, None,
                               round((time.perf_counter() - t0) * 1000, 1), error=f"{type(e).__name__}: {e}")
            raise
        if self.audit is not None:
            self.audit.log(user, strategy, query, res, round((time.perf_counter() - t0) * 1000, 1))
        return res

    def _answer(self, query, qid=None, user=None, strategy="none"):
        t0 = time.perf_counter()
        passages, trace = self.retrieve_traced(query, self.L1["n_passages"], user, strategy)
        t1 = time.perf_counter()
        if not passages:   # nothing allowed: never call the LLM with an empty passage list
            return {
                "qid": qid, "query": query, "answer": REFUSAL_TEXT,
                "status": "refused_no_passages", "refused": True,
                "reasons": [], "attempts": 0, "cited": [], "raw_answer": "",
                "attempt_log": [], "passages": [], "trace": trace,
                "retrieval_ms": round((t1 - t0) * 1000, 1), "generation_ms": 0.0,
            }
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
            "passages": passages, "trace": trace,
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