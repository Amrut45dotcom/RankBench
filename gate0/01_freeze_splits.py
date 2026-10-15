"""
Gate 0.3 - freeze the evaluation query files.

Eligible queries = queries with at least one relevant passage inside the
subset. Writes (once, never overwritten):
    splits/dev_200.txt          debugging / prompt tweaks
    splits/final.txt            reporting set (disjoint from dev)
    splits/answer_eval_200.txt  fixed sample of final for answer-level eval
    splits/manifest.json        seed, counts, sha256 of every file

Refuses to run if any of these files already exist.

Usage:
    python gate0/01_freeze_splits.py
"""
import hashlib
import json
import os
import random
import sys

import ir_datasets
import numpy as np

DATASET_NAME     = "msmarco-passage/dev/small"
PID_LIST_PATH    = "embeddings/pid_list.npy"
OUT_DIR          = "splits"
SEED             = 42
DEV_SIZE         = 200
ANSWER_EVAL_SIZE = 200

FILES = {
    "dev": "dev_200.txt",
    "final": "final.txt",
    "answer_eval": "answer_eval_200.txt",
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = [os.path.join(OUT_DIR, n) for n in FILES.values()]
    targets.append(os.path.join(OUT_DIR, "manifest.json"))
    existing = [t for t in targets if os.path.exists(t)]
    if existing:
        print("Split files already exist and are frozen. Not overwriting:")
        for t in existing:
            print("  ", t)
        sys.exit(1)

    subset = {str(p) for p in np.load(PID_LIST_PATH)}
    dataset = ir_datasets.load(DATASET_NAME)

    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    all_qids = {q.query_id for q in dataset.queries_iter()}
    eligible = sorted(
        (q for q in all_qids if qrels.get(q, set()) & subset),
        key=lambda x: int(x),
    )
    dropped = len(all_qids) - len(eligible)
    print(f"queries: {len(all_qids):,}  eligible: {len(eligible):,}  dropped: {dropped:,}")

    rng = random.Random(SEED)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    dev = sorted(shuffled[:DEV_SIZE], key=int)
    final = sorted(shuffled[DEV_SIZE:], key=int)

    rng2 = random.Random(SEED + 1)
    answer_eval = sorted(rng2.sample(final, min(ANSWER_EVAL_SIZE, len(final))), key=int)

    assert not set(dev) & set(final), "dev and final must be disjoint"
    assert set(answer_eval) <= set(final), "answer_eval must be inside final"

    paths = {k: os.path.join(OUT_DIR, v) for k, v in FILES.items()}
    write_ids(paths["dev"], dev)
    write_ids(paths["final"], final)
    write_ids(paths["answer_eval"], answer_eval)

    manifest = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "pid_list_sha256": sha256_file(PID_LIST_PATH),
        "num_queries_total": len(all_qids),
        "num_queries_eligible": len(eligible),
        "num_queries_dropped": dropped,
        "counts": {"dev": len(dev), "final": len(final), "answer_eval": len(answer_eval)},
        "sha256": {k: sha256_file(p) for k, p in paths.items()},
        "rule": "Never edit these files after seeing results.",
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["counts"]))
    print("Frozen. Commit splits/ now.")


if __name__ == "__main__":
    main()
