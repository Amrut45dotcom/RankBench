# Experiment log

One line per experiment. Dates are approximate where file timestamps were not available. Frozen results are never rerun to get a better number.

| Date | What changed | Result | Keep / discard |
|---|---|---|---|
| 2026-10-15 | Subset audit; froze splits (seed 42); built BM25 over the 1M subset | 6,980 of 6,980 queries covered; dev 200, final 6,780; 1M docs indexed in about 41 s | keep (frozen) |
| 2026-10-15 | Dense rerun with BGE query prefix vs without | 0.7145 vs 0.7118 (exact match without prefix) | prefix discarded for the baseline |
| 2026-10-15 | Matched-corpus quality run, latency run, significance tests | BM25 0.4445, dense 0.7108, RRF 0.6060, reranker 0.7538 NDCG@10; reranker > dense > RRF > BM25 | keep (frozen) |
| 2026-10-16 to 10-20 | Level 1 build: prompt, validator, splitter bug fix, dev_200 run, /answer, single answer_eval_200 run, faithfulness labels, off-topic Set B, /query on the 1M BM25 index, tag `level1` | dev 181 answered / 19 refused; eval 189 / 11; faithfulness 37 of 40; Set B 15 of 20 refused | keep |
| 2026-10-21 | L2.1 k-means tenants: first run trained on 1,024 points (FAISS default cap), rerun with 200,000 | final sizes 292,844 / 253,624 / 198,779 / 254,753 | first run discarded, rerun frozen |
| 2026-10-21 | L2.2 permissions.py and 11 tests; FAISS selector check; L2.3 strategies and check_retrieval.py; L2.4 audit logger and 6 tests | all pass; control showed 33 forbidden passages in final results | keep |
| 2026-10-21 | L2.5 retrieval-level leakage suite (original, 248 tests) | 0 leaks in 744 runs; control leaked in 217 of 248 | keep (frozen) |
| 2026-10-21 | L2.5 answer-level full run, 3 strategies | 273 of 744 answered, 471 Groq 429 (daily token cap) | not a result; kept only as a resume base |
| 2026-10-21 | L2.6 30-query smoke test | pre-filter fastest; gaps at noise level | not for the README |
| 2026-10-23 | Full retrieval-only comparison, 6,780 queries x 10 runs | pre fastest; post slightly higher NDCG@10 before reranking; candidate pool shrinks under post-filter | keep (frozen) |
| 2026-10-23 | Rerank comparison, 1,000-query sample | strategies identical on NDCG@10 within 0.01%; latency noisy | keep (frozen) |
| 2026-10-23 | Found the patched run_leakage.py, audit change and guard tests were never committed; rewrote limits.py, test_limits.py and run_leakage.py (strategies, resume, rate-limit handling) and shipped | server run worked end to end | keep |
| 2026-10-23 | Answer-level rerun, pre only, resumed (92 rows reused, 156 new calls) | 248 of 248 covered, 0 errors, 1 flagged | keep (frozen) |
| 2026-10-23 | Inspected the flagged run (reran it 3 times, printed passages) | false positive: near-duplicate allowed passage | keep |
| 2026-10-24 | Runtime citation guard (`citation_guard.py`, check in `_answer`, `blocked_citations` in audit log and leakage suite) | 9 function tests and 3 end-to-end tests pass; negative-number bug found by the tests and fixed | keep |
| 2026-10-24 | `public_view`: identical output for every refusal type, no internal fields | 4 tests pass | keep |
| 2026-10-24 | Extra attack set (40 hand-written queries, 5 new categories), first attempts | crashed: BM25 `IndexError` on a query with no tokens, then fusion with a single run, then a paste error | discard (no result saved) |
| 2026-10-24 | Fixed the empty-token and single-run cases in `pipeline.py`; reran the extra suite at retrieval level | 240 tests x 3 strategies, 0 leaks; control leaked in 213 of 240 | keep (frozen) |
| 2026-10-25 | Extra suite, answer level, pre-filter | 240 of 240 covered, 0 errors, 0 flagged; 92 answered, 138 refused by model, 10 validator refusals, 0 guard blocks | keep (frozen) |
| 2026-10-25 | Permission-aware `/answer` in `main.py` (persona mapped to a fixed user, `pre`, `public_view`, audit log) | 7 route tests pass; live check passed; 44 unit tests in total | keep |
| 2026-10-25 | UI "Ask (permission-aware)" section in `app.py` | guest, legal_manager and hr_intern checked; the cats example verified against the masks | keep |
