"""
NER Pipeline v3 — Main tagging script.
Processes train and/or eval CSVs, annotates dialogues with clinical entities,
and saves results with checkpointing support.

Usage:
    python -m ner_pipeline_v3.run_tagging --split both
    python -m ner_pipeline_v3.run_tagging --split train
    python -m ner_pipeline_v3.run_tagging --split eval
"""

import argparse
import json
import os
import time
from collections import Counter

import pandas as pd

from ner_pipeline_v3.config import (
    TRAIN_CSV,
    EVAL_CSV,
    TRAIN_OUTPUT,
    EVAL_OUTPUT,
    CHECKPOINT_EVERY,
)
from ner_pipeline_v3.extractor import ClinicalExtractorV3


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _partial_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_partial{ext}"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _safe_dialogue(value) -> str:
    """Convert a potentially NaN cell to a clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _load_with_resume(csv_path: str, output_path: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Load the source CSV and check for a partial checkpoint.

    Returns:
        df_source   — full source DataFrame
        df_done     — already-processed rows (may be empty)
        start_idx   — index to resume from
    """
    df_source = pd.read_csv(csv_path)
    partial_path = _partial_path(output_path)

    if os.path.exists(partial_path):
        df_done = pd.read_csv(partial_path)
        start_idx = len(df_done)
        print(f"  Resuming from checkpoint: {start_idx}/{len(df_source)} rows already done.")
    else:
        df_done = pd.DataFrame()
        start_idx = 0

    return df_source, df_done, start_idx


def _print_summary(results: list[dict], split_name: str) -> None:
    """Print per-split summary statistics to the console."""
    total = len(results)
    if total == 0:
        print(f"\n[{split_name}] No rows processed.")
        return

    all_entities = [e for row in results for e in row["entities"]]
    counts = [len(row["entities"]) for row in results]
    min_scores = [row["min_score"] for row in results if row["entities"]]
    max_scores = [row["max_score"] for row in results if row["entities"]]

    avg_count = sum(counts) / total
    avg_min = sum(min_scores) / len(min_scores) if min_scores else 0.0
    avg_max = sum(max_scores) / len(max_scores) if max_scores else 0.0

    label_dist = Counter(e["label"] for e in all_entities)

    print(f"\n{'─' * 60}")
    print(f"  Summary — {split_name} split")
    print(f"{'─' * 60}")
    print(f"  Total rows processed   : {total}")
    print(f"  Avg entities per row   : {avg_count:.2f}")
    print(f"  Avg min confidence     : {avg_min:.3f}")
    print(f"  Avg max confidence     : {avg_max:.3f}")
    print(f"\n  Label distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {count:>5}")
    print(f"{'─' * 60}\n")


# ─── Core processing ──────────────────────────────────────────────────────────

def process_split(
    extractor: ClinicalExtractorV3,
    csv_path: str,
    output_path: str,
    split_name: str,
) -> None:
    """Tag one CSV split (train or eval) and save annotated output."""

    print(f"\n{'═' * 60}")
    print(f"  Processing split: {split_name.upper()}")
    print(f"  Source : {csv_path}")
    print(f"  Output : {output_path}")
    print(f"{'═' * 60}")

    _ensure_dir(output_path)
    partial_path = _partial_path(output_path)

    df_source, df_done, start_idx = _load_with_resume(csv_path, output_path)
    total = len(df_source)

    if start_idx >= total:
        print("  All rows already processed. Loading checkpoint as final output.")
        df_done.to_csv(output_path, index=False)
        os.remove(partial_path)
        return

    # Rows already done — keep their entities_json as-is
    done_rows = df_done.to_dict("records") if not df_done.empty else []

    # Statistics accumulators for summary
    results_meta: list[dict] = []
    for row in done_rows:
        try:
            ents = json.loads(row.get("entities_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            ents = []
        scores = [e["score"] for e in ents]
        results_meta.append(
            {
                "entities": ents,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
            }
        )

    # Timing
    speed_window: list[float] = []  # per-row processing times

    new_rows: list[dict] = []

    for idx in range(start_idx, total):
        row = df_source.iloc[idx].to_dict()
        dialogue = _safe_dialogue(row.get("dialogue", ""))

        if not dialogue:
            row["entities_json"] = json.dumps([])
            new_rows.append(row)
            results_meta.append({"entities": [], "min_score": 0.0, "max_score": 0.0})
            print(f"  [{idx + 1}/{total}] (empty dialogue — skipped)")
            continue

        t0 = time.perf_counter()
        entities = extractor.extract(dialogue)
        elapsed = time.perf_counter() - t0

        speed_window.append(elapsed)
        # Keep rolling window of last 20 rows for ETA
        if len(speed_window) > 20:
            speed_window.pop(0)

        scores = [e["score"] for e in entities]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        row["entities_json"] = json.dumps(entities)
        new_rows.append(row)
        results_meta.append(
            {"entities": entities, "min_score": min_score, "max_score": max_score}
        )

        # ETA calculation
        avg_time = sum(speed_window) / len(speed_window)
        remaining = total - (idx + 1)
        eta_sec = avg_time * remaining
        eta_str = _format_eta(eta_sec)

        print(
            f"  [{idx + 1}/{total}] "
            f"Extracted {len(entities):>2} entities  "
            f"(scores: {min_score:.2f}–{max_score:.2f})  "
            f"| {elapsed:.2f}s  ETA: {eta_str}"
        )

        # Checkpoint
        rows_since_start = idx - start_idx + 1
        if rows_since_start % CHECKPOINT_EVERY == 0:
            df_partial = pd.DataFrame(done_rows + new_rows)
            df_partial.to_csv(partial_path, index=False)
            print(f"  [checkpoint] Saved {len(df_partial)} rows → {partial_path}")

    # Final save
    all_rows = done_rows + new_rows
    df_final = pd.DataFrame(all_rows)
    df_final.to_csv(output_path, index=False)
    print(f"\n  Final output saved → {output_path}  ({len(df_final)} rows)")

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)

    _print_summary(results_meta, split_name)


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NER Pipeline v3 — Tag clinical dialogues with SOAP entities."
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval", "both"],
        default="both",
        help="Which data split to process (default: both)",
    )
    args = parser.parse_args()

    print("\n  Loading GLiNER model … (this may take a moment)")
    extractor = ClinicalExtractorV3()
    print("  Model loaded.\n")

    splits = {
        "train": (TRAIN_CSV, TRAIN_OUTPUT),
        "eval": (EVAL_CSV, EVAL_OUTPUT),
    }

    to_run = ["train", "eval"] if args.split == "both" else [args.split]

    for split_name in to_run:
        csv_path, output_path = splits[split_name]
        process_split(extractor, csv_path, output_path, split_name)

    print("  All done.")


if __name__ == "__main__":
    main()
