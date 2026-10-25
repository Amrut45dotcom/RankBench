### Retrieval-only (after RRF, no reranker), all final queries (6780 queries)

Unfiltered baseline (all queries): NDCG@10 0.6061, MRR@10 0.5513, Recall@100 0.9727, p50 169 ms, p95 230 ms

| Persona | Strategy | Gold visible | Should refuse | NDCG@10 | vs pre | MRR@10 | Recall@100 | Avg candidates | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| guest | post | 2057 (30%) | 4723 (70%) | 0.7348 | 101.6% | 0.6812 | 0.9820 | 53.7 | 168 | 228 |
| guest | oversample | 2057 (30%) | 4723 (70%) | 0.7228 | 99.9% | 0.6748 | 0.9888 | 100.0 | 170 | 230 |
| guest | pre | 2057 (30%) | 4723 (70%) | 0.7234 | 100.0% | 0.6756 | 0.9888 | 100.0 | 110 | 131 |
| hr_intern | post | 2747 (41%) | 4033 (59%) | 0.6992 | 101.1% | 0.6433 | 0.9818 | 72.0 | 170 | 232 |
| hr_intern | oversample | 2747 (41%) | 4033 (59%) | 0.6916 | 100.0% | 0.6391 | 0.9863 | 100.0 | 170 | 230 |
| hr_intern | pre | 2747 (41%) | 4033 (59%) | 0.6917 | 100.0% | 0.6393 | 0.9863 | 100.0 | 137 | 162 |
| hr_manager | post | 3290 (49%) | 3490 (51%) | 0.6723 | 100.7% | 0.6160 | 0.9815 | 77.4 | 170 | 228 |
| hr_manager | oversample | 3290 (49%) | 3490 (51%) | 0.6674 | 99.9% | 0.6129 | 0.9850 | 100.0 | 170 | 230 |
| hr_manager | pre | 3290 (49%) | 3490 (51%) | 0.6677 | 100.0% | 0.6135 | 0.9850 | 100.0 | 154 | 181 |

### Full pipeline with reranker, fixed sample (1000 queries)

Unfiltered baseline (all queries): NDCG@10 0.7537, MRR@10 0.7081, Recall@100 0.9773, p50 643 ms, p95 1039 ms

| Persona | Strategy | Gold visible | Should refuse | NDCG@10 | vs pre | MRR@10 | Recall@100 | Avg candidates | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| guest | post | 319 (32%) | 681 (68%) | 0.8618 | 100.0% | 0.8303 | 0.9859 | 53.5 | 277 | 401 |
| guest | oversample | 319 (32%) | 681 (68%) | 0.8617 | 100.0% | 0.8302 | 0.9906 | 100.0 | 410 | 646 |
| guest | pre | 319 (32%) | 681 (68%) | 0.8617 | 100.0% | 0.8302 | 0.9906 | 100.0 | 155 | 193 |
| hr_intern | post | 418 (42%) | 582 (58%) | 0.8365 | 100.0% | 0.7970 | 0.9868 | 72.6 | 205 | 338 |
| hr_intern | oversample | 418 (42%) | 582 (58%) | 0.8364 | 100.0% | 0.7969 | 0.9904 | 100.0 | 206 | 287 |
| hr_intern | pre | 418 (42%) | 582 (58%) | 0.8364 | 100.0% | 0.7969 | 0.9904 | 100.0 | 176 | 220 |
| hr_manager | post | 509 (51%) | 491 (49%) | 0.8236 | 100.0% | 0.7823 | 0.9872 | 78.1 | 194 | 278 |
| hr_manager | oversample | 509 (51%) | 491 (49%) | 0.8236 | 100.0% | 0.7822 | 0.9892 | 100.0 | 196 | 254 |
| hr_manager | pre | 509 (51%) | 491 (49%) | 0.8236 | 100.0% | 0.7822 | 0.9892 | 100.0 | 191 | 236 |

Notes: gold-visible = the persona may see at least one passage MS MARCO marks relevant; NDCG/MRR/Recall use only the visible gold passages. Latency is end-to-end per query on a shared server, compare within a run only. Avg candidates = size of the fused list before reranking (up to 100). vs pre = NDCG@10 relative to pre-filter for the same persona (pre-filter is the reference at 100%).
