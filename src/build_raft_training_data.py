"""Build RAFT training data for Dial2Note.

For each of the 8 529 training examples, retrieves the top-1 most similar
*other* training example's gold SOAP note and adds it as a ``retrieved_note``
column.  The result is saved to ``data/processed/shared_task_train_raft.csv``.

Self-match exclusion
--------------------
Retrieving a dialogue against the corpus that contains it will typically return
the example itself as the #1 hit.  To avoid this, we request k=2 and skip any
result whose ``id`` matches the query's own ``id``.  If both top-2 results
match (extremely unlikely with 8 529 examples), we fall back to the top result
and log a warning.

Chunked retrieval
-----------------
To avoid out-of-memory errors when the sentence-transformer encodes large
batches, retrieval is done in chunks of CHUNK=128 dialogues at a time — the
same strategy used by ``inference_pipeline.py --precompute_retrieval``.

Output
------
data/processed/shared_task_train_raft.csv
    id, dialogue, note, retrieved_note

Usage
-----
python -m src.build_raft_training_data [--train_csv PATH] [--output_csv PATH]
                                        [--faiss_index PATH] [--chunk INT]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Default paths / constants
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_CSV    = "data/processed/shared_task_train.csv"
_OUTPUT_CSV   = "data/processed/shared_task_train_raft.csv"
_FAISS_INDEX  = "outputs/hybrid_faiss.index"
_CHUNK        = 128          # dialogues per retrieval batch (avoids MPS/CUDA OOM)
_K            = 2            # retrieve top-2 so we can skip the self-match


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def build_raft_data(
    train_csv: str = _TRAIN_CSV,
    output_csv: str = _OUTPUT_CSV,
    faiss_index: str = _FAISS_INDEX,
    chunk: int = _CHUNK,
) -> pd.DataFrame:
    """Retrieve top-1 non-self similar training note for every training example.

    Parameters
    ----------
    train_csv:
        Path to the training CSV (columns: id, dialogue, note).
    output_csv:
        Path where the RAFT CSV will be saved.
    faiss_index:
        Path to the FAISS index file.  Built automatically if absent.
    chunk:
        Number of dialogues to process per retrieval batch.

    Returns
    -------
    DataFrame with columns: id, dialogue, note, retrieved_note.
    """
    t_start = time.perf_counter()

    # ── load training data ────────────────────────────────────────────────────
    print(f"[RAFT] Loading training data from {train_csv!r} …")
    df = pd.read_csv(train_csv)
    required = {"id", "dialogue", "note"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")

    df["id"]       = df["id"].astype(int)
    df["dialogue"] = df["dialogue"].apply(str)
    df["note"]     = df["note"].apply(str)
    n = len(df)
    print(f"[RAFT] Loaded {n} training examples.")

    # ── initialise retriever ──────────────────────────────────────────────────
    # Import here so the script can be imported without heavy deps if needed.
    from src.hybrid_retriever import HybridRetriever  # noqa: PLC0415

    retriever = HybridRetriever(
        train_csv_path=train_csv,
        faiss_index_path=faiss_index,
    )

    # Pre-build a map: corpus_id → row_index (for fast self-match detection)
    id_to_idx = {int(row_id): i for i, row_id in enumerate(df["id"])}

    # ── chunked retrieval ─────────────────────────────────────────────────────
    dialogues: List[str] = df["dialogue"].tolist()
    ids: List[int]       = df["id"].tolist()

    print(
        f"[RAFT] Retrieving top-{_K} examples for all {n} dialogues "
        f"(chunk_size={chunk}) …"
    )
    t0 = time.perf_counter()
    all_results = []
    for start in range(0, n, chunk):
        chunk_dialogues = dialogues[start : start + chunk]
        chunk_results   = retriever.retrieve_batch(chunk_dialogues, k=_K)
        all_results.extend(chunk_results)
        done    = min(start + chunk, n)
        elapsed = time.perf_counter() - t0
        eta     = elapsed / done * (n - done) if done else 0
        print(
            f"[RAFT]   {done:>5}/{n}  "
            f"({elapsed:.1f}s elapsed, ETA {eta:.1f}s)"
        )

    retrieval_elapsed = time.perf_counter() - t0
    print(
        f"[RAFT] Retrieval complete in {retrieval_elapsed:.1f}s "
        f"({retrieval_elapsed/n*1000:.1f}ms/example)"
    )

    # ── select top-1 non-self result for each example ─────────────────────────
    retrieved_notes: List[str] = []
    n_self_skipped  = 0
    n_fallback      = 0

    for query_id, results in zip(ids, all_results):
        chosen = None
        for hit in results:
            if hit["id"] != query_id:
                chosen = hit["note"]   # full gold note — no truncation
                n_self_skipped += 1 if hit is not results[0] else 0
                break

        if chosen is None:
            # Both top-2 were self-matches (shouldn't happen with k=2 on
            # a corpus > 2 examples, but guard anyway)
            chosen = results[0]["note"] if results else ""
            n_fallback += 1
            print(
                f"[RAFT] Warning: could not find non-self match for id={query_id}; "
                f"using top-1 result."
            )

        retrieved_notes.append(chosen)

    print(
        f"[RAFT] Self-matches skipped: {n_self_skipped}  "
        f"Fallbacks: {n_fallback}"
    )

    # ── assemble and save output ───────────────────────────────────────────────
    out_df = pd.DataFrame({
        "id":             df["id"].values,
        "dialogue":       df["dialogue"].values,
        "note":           df["note"].values,
        "retrieved_note": retrieved_notes,
    })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    total_elapsed = time.perf_counter() - t_start
    col = 34
    print(
        f"\n{'─'*54}\n"
        f" RAFT Data Build Report\n"
        f"{'─'*54}\n"
        f"  {'Examples processed:':<{col}} {n}\n"
        f"  {'Self-matches skipped (used #2 hit):':<{col}} {n_self_skipped}\n"
        f"  {'Fallbacks (used #1 anyway):':<{col}} {n_fallback}\n"
        f"  {'Avg retrieved_note length (chars):':<{col}} "
        f"{sum(len(r) for r in retrieved_notes)/n:.0f}\n"
        f"  {'Total time:':<{col}} {total_elapsed:.1f}s\n"
        f"  {'Output saved to:':<{col}} {output_csv}\n"
        f"{'─'*54}"
    )

    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build RAFT training data by retrieving the top-1 non-self similar "
            "gold note for every training example."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train_csv",
        default=_TRAIN_CSV,
        help="Training CSV (columns: id, dialogue, note)",
    )
    p.add_argument(
        "--output_csv",
        default=_OUTPUT_CSV,
        help="Where to save the RAFT CSV (columns: id, dialogue, note, retrieved_note)",
    )
    p.add_argument(
        "--faiss_index",
        default=_FAISS_INDEX,
        help="FAISS index path (built from training dialogues; created if absent)",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=_CHUNK,
        help="Number of dialogues per retrieval batch (reduce if OOM)",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    build_raft_data(
        train_csv=args.train_csv,
        output_csv=args.output_csv,
        faiss_index=args.faiss_index,
        chunk=args.chunk,
    )
