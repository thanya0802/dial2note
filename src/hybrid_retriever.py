"""
Hybrid retriever for Dial2Note SOAP note generation.
Combines BM25 (rank_bm25) and FAISS (sentence-transformers) with
Reciprocal Rank Fusion to retrieve the most similar training examples.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── constants ──────────────────────────────────────────────────────────────────
_RRF_K = 60          # standard RRF constant
_CANDIDATE_POOL = 20  # candidates retrieved per method before fusion
_MAX_CHARS = 2000     # max chars for the returned dialogue field (metadata only)
# Notes are returned in full — no truncation — so the prompt builder can use
# the complete gold SOAP note as a reference without information loss.


# ── helpers ────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    """Whitespace + lowercase tokenisation for BM25."""
    return str(text).lower().split()


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ── main class ─────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Retrieves training examples via BM25 + FAISS Reciprocal Rank Fusion.

    Parameters
    ----------
    train_csv_path : str
        Path to CSV with columns: id, note, dialogue.
    faiss_index_path : str | None
        Path to save/load the FAISS index.  Defaults to
        "outputs/hybrid_faiss.index".
    embedding_model : str
        Sentence-Transformers model name for FAISS embeddings.
    """

    def __init__(
        self,
        train_csv_path: str,
        faiss_index_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.faiss_index_path = faiss_index_path or "outputs/hybrid_faiss.index"
        self.embedding_model_name = embedding_model

        # ── load data ─────────────────────────────────────────────────────────
        print(f"[HybridRetriever] Loading training data from: {train_csv_path}")
        df = pd.read_csv(train_csv_path)
        df["dialogue"] = df["dialogue"].apply(str)
        df["note"] = df["note"].apply(str)
        self.df = df.reset_index(drop=True)
        self.dialogues: List[str] = self.df["dialogue"].tolist()
        self.n = len(self.dialogues)
        print(f"[HybridRetriever] Loaded {self.n} training examples.")

        # ── BM25 ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        tokenised = [_tokenize(d) for d in self.dialogues]
        self.bm25 = BM25Okapi(tokenised)
        print(f"[HybridRetriever] BM25 index built in {time.perf_counter()-t0:.2f}s")

        # ── sentence-transformer + FAISS ──────────────────────────────────────
        print(f"[HybridRetriever] Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        if os.path.exists(self.faiss_index_path):
            t0 = time.perf_counter()
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            print(
                f"[HybridRetriever] FAISS index loaded from {self.faiss_index_path} "
                f"in {time.perf_counter()-t0:.2f}s  "
                f"(vectors: {self.faiss_index.ntotal})"
            )
        else:
            t0 = time.perf_counter()
            print("[HybridRetriever] Building FAISS index …")
            embeddings = self.encoder.encode(
                self.dialogues,
                batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # inner-product == cosine (normalised)
            self.faiss_index.add(np.array(embeddings, dtype=np.float32))
            os.makedirs(os.path.dirname(self.faiss_index_path) or ".", exist_ok=True)
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            print(
                f"[HybridRetriever] FAISS index built and saved to "
                f"{self.faiss_index_path} in {time.perf_counter()-t0:.2f}s"
            )

    # ── single retrieve ────────────────────────────────────────────────────────
    def retrieve(self, query_dialogue: str, k: int = 2) -> List[Dict]:
        """
        Retrieve top-k training examples for a single query dialogue.

        Returns a list of dicts with keys:
            id, dialogue, note, bm25_rank, faiss_rank, rrf_score
        """
        t0 = time.perf_counter()
        results = self._fuse(
            bm25_results=self._bm25_top(query_dialogue),
            faiss_results=self._faiss_top_batch([query_dialogue])[0],
            k=k,
        )
        print(
            f"[HybridRetriever] retrieve() completed in "
            f"{(time.perf_counter()-t0)*1000:.1f}ms"
        )
        return results

    # ── batch retrieve ─────────────────────────────────────────────────────────
    def retrieve_batch(
        self, query_dialogues: List[str], k: int = 2
    ) -> List[List[Dict]]:
        """
        Retrieve top-k training examples for a batch of query dialogues.

        FAISS encoding is batched; BM25 searches are done per query.
        """
        t0 = time.perf_counter()

        faiss_batch = self._faiss_top_batch(query_dialogues)

        all_results: List[List[Dict]] = []
        for query, faiss_res in zip(query_dialogues, faiss_batch):
            bm25_res = self._bm25_top(query)
            all_results.append(self._fuse(bm25_res, faiss_res, k=k))

        print(
            f"[HybridRetriever] retrieve_batch({len(query_dialogues)} queries) "
            f"completed in {(time.perf_counter()-t0)*1000:.1f}ms"
        )
        return all_results

    # ── internal helpers ───────────────────────────────────────────────────────
    def _bm25_top(self, query: str) -> List[tuple[int, float]]:
        """Return list of (corpus_idx, score) for top _CANDIDATE_POOL BM25 hits."""
        scores = self.bm25.get_scores(_tokenize(query))
        top_n = min(_CANDIDATE_POOL, self.n)
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _faiss_top_batch(
        self, queries: List[str]
    ) -> List[List[tuple[int, float]]]:
        """
        Encode all queries at once and return top _CANDIDATE_POOL FAISS hits
        per query as list of (corpus_idx, score).
        """
        embeddings = self.encoder.encode(
            queries,
            batch_size=max(1, len(queries)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        k = min(_CANDIDATE_POOL, self.n)
        scores_mat, indices_mat = self.faiss_index.search(
            np.array(embeddings, dtype=np.float32), k
        )
        results = []
        for scores_row, indices_row in zip(scores_mat, indices_mat):
            results.append(
                [(int(idx), float(sc)) for idx, sc in zip(indices_row, scores_row)]
            )
        return results

    def _fuse(
        self,
        bm25_results: List[tuple[int, float]],
        faiss_results: List[tuple[int, float]],
        k: int,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion over BM25 and FAISS candidate lists.

        RRF score for each candidate = Σ 1 / (rank + _RRF_K)
        where rank is 1-based position in each ranked list.
        """
        rrf_scores: Dict[int, float] = {}
        bm25_rank_map: Dict[int, int] = {}
        faiss_rank_map: Dict[int, int] = {}

        for rank, (idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + _RRF_K)
            bm25_rank_map[idx] = rank

        for rank, (idx, _) in enumerate(faiss_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + _RRF_K)
            faiss_rank_map[idx] = rank

        sorted_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        output: List[Dict] = []
        for idx, score in sorted_candidates[:k]:
            row = self.df.iloc[idx]
            output.append(
                {
                    "id": int(row["id"]),
                    "dialogue": _truncate(row["dialogue"]),  # truncated — used for display only
                    "note": str(row["note"]),                # full note — no truncation
                    "bm25_rank": bm25_rank_map.get(idx, -1),
                    "faiss_rank": faiss_rank_map.get(idx, -1),
                    "rrf_score": round(score, 6),
                }
            )
        return output


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_CSV = "data/processed/shared_task_train.csv"
    FAISS_PATH = "outputs/hybrid_faiss.index"

    print("=" * 70)
    print("Initialising HybridRetriever …")
    print("=" * 70)
    retriever = HybridRetriever(
        train_csv_path=TRAIN_CSV,
        faiss_index_path=FAISS_PATH,
    )

    # ── single retrieve ────────────────────────────────────────────────────────
    sample_query = (
        "Patient presents with chest pain for 3 days, "
        "worse with exertion. Denies shortness of breath."
    )
    print("\n" + "=" * 70)
    print(f"Single retrieve query:\n  {sample_query}")
    print("=" * 70)
    results = retriever.retrieve(sample_query, k=1)
    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"  id         : {res['id']}")
        print(f"  bm25_rank  : {res['bm25_rank']}")
        print(f"  faiss_rank : {res['faiss_rank']}")
        print(f"  rrf_score  : {res['rrf_score']}")
        print(f"  dialogue   : {res['dialogue'][:200]} …")
        print(f"  note       : {res['note'][:200]} …  (full length: {len(res['note'])} chars)")

    # ── batch retrieve ─────────────────────────────────────────────────────────
    batch_queries = [
        "Patient presents with chest pain for 3 days, worse with exertion. Denies shortness of breath.",
        "Elderly patient with confusion, fever, and productive cough for 5 days. History of COPD.",
        "Follow-up for type 2 diabetes management. Patient reports good compliance with metformin.",
    ]
    print("\n" + "=" * 70)
    print(f"Batch retrieve — {len(batch_queries)} queries  (k=1)")
    print("=" * 70)
    batch_results = retriever.retrieve_batch(batch_queries, k=1)
    for q_idx, (query, hits) in enumerate(zip(batch_queries, batch_results), 1):
        print(f"\n[Query {q_idx}] {query[:80]} …")
        for j, res in enumerate(hits, 1):
            print(
                f"  [{j}] id={res['id']}  bm25_rank={res['bm25_rank']}  "
                f"faiss_rank={res['faiss_rank']}  rrf={res['rrf_score']}"
            )
            print(f"       dialogue: {res['dialogue'][:200]} …")
            print(f"       note    : {res['note'][:200]} …  (full: {len(res['note'])} chars)")
