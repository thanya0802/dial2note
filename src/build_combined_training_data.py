"""
Merge NER annotations + RAFT retrieved notes into one training CSV.

Usage:
    python -m src.build_combined_training_data

Optional eval merge:
    python -m src.build_combined_training_data \\
        --eval-ner path/to/eval_annotated.csv \\
        --eval-raft path/to/shared_task_eval_raft.csv \\
        --output-eval data/processed/shared_task_eval_combined.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.combined_prompt import parse_entities_from_json

DEFAULT_TRAIN_NER = Path("ner_pipeline_v3/outputs/train_annotated_v3_mistral.csv")
DEFAULT_TRAIN_RAFT = Path("data/processed/shared_task_train_raft.csv")
DEFAULT_OUT_TRAIN = Path("data/processed/shared_task_train_combined.csv")


def _require_cols(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] {name} CSV missing columns: {missing}. Expected: {cols}.")


def _merge_pair(
    ner_path: Path,
    raft_path: Path,
    output_path: Path,
    *,
    label: str,
) -> pd.DataFrame:
    if not ner_path.exists():
        sys.exit(f"[ERROR] {ner_path} not found.")
    if not raft_path.exists():
        sys.exit(f"[ERROR] {raft_path} not found.")

    ner_df = pd.read_csv(ner_path, engine="python", on_bad_lines="warn")
    raft_df = pd.read_csv(raft_path, engine="python", on_bad_lines="warn")

    _require_cols(ner_df, {"id", "dialogue", "note", "entities_json"}, str(ner_path))
    _require_cols(raft_df, {"id", "dialogue", "note", "retrieved_note"}, str(raft_path))

    merged = ner_df.merge(
        raft_df[["id", "retrieved_note"]],
        on="id",
        how="inner",
    )

    if len(merged) != len(ner_df) or len(merged) != len(raft_df):
        print(
            f"[WARN] {label}: row count mismatch after merge "
            f"(ner={len(ner_df)}, raft={len(raft_df)}, merged={len(merged)}).",
        )

    bad_entities = merged["entities_json"].isna() | (merged["entities_json"].astype(str).str.strip() == "")
    bad_ret = merged["retrieved_note"].isna() | (merged["retrieved_note"].astype(str).str.strip() == "")
    if bad_entities.any():
        sys.exit(f"[ERROR] {label}: {bad_entities.sum()} rows missing entities_json.")
    if bad_ret.any():
        sys.exit(f"[ERROR] {label}: {bad_ret.sum()} rows missing retrieved_note.")

    out = merged[["id", "dialogue", "note", "entities_json", "retrieved_note"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} rows → {output_path}")
    return out


def _stats(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    ent_counts = []
    for _, row in df.iterrows():
        ents = parse_entities_from_json(row["entities_json"])
        ent_counts.append(len(ents))
    avg_e = sum(ent_counts) / n if n else 0.0
    lens = df["retrieved_note"].astype(str).str.len()
    avg_r = float(lens.mean()) if n else 0.0

    print(f"\n--- Stats ({label}) ---")
    print(f"  Rows:                    {n}")
    print(f"  Avg entities / dialogue: {avg_e:.2f}")
    print(f"  Avg retrieved_note len:    {avg_r:.1f} chars")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge NER + RAFT CSVs for combined fine-tuning.")
    ap.add_argument("--train-ner", type=Path, default=DEFAULT_TRAIN_NER)
    ap.add_argument("--train-raft", type=Path, default=DEFAULT_TRAIN_RAFT)
    ap.add_argument("--output-train", type=Path, default=DEFAULT_OUT_TRAIN)
    ap.add_argument("--eval-ner", type=Path, default=None, help="Optional eval NER CSV (same columns as train NER).")
    ap.add_argument("--eval-raft", type=Path, default=None, help="Optional eval RAFT CSV with retrieved_note.")
    ap.add_argument("--output-eval", type=Path, default=Path("data/processed/shared_task_eval_combined.csv"))
    args = ap.parse_args()

    print("=" * 60)
    print("Build combined training data (NER + retrieved SOAP)")
    print("=" * 60)

    train_df = _merge_pair(args.train_ner, args.train_raft, args.output_train, label="train")
    _stats(train_df, "train")

    if args.eval_ner is not None and args.eval_raft is not None:
        eval_df = _merge_pair(args.eval_ner, args.eval_raft, args.output_eval, label="eval")
        _stats(eval_df, "eval")
    elif args.eval_ner is not None or args.eval_raft is not None:
        sys.exit("[ERROR] Provide both --eval-ner and --eval-raft, or neither.")

    print("\nDone.")


if __name__ == "__main__":
    main()
