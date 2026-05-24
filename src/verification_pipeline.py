"""
Dial2Note verification pipeline — verify → correct → save → evaluate.

Runs AFTER inference_pipeline.py has generated SOAP notes.
No GPU, no vLLM — entirely CPU-based rule corrections.

Usage
-----
python -m src.verification_pipeline [options]

Run with --help for full argument list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

from src.verifier_agent import ClinicalVerifier, VerificationResult


# ── helpers ──────────────────────────────────────────────────────────────────────────────

def _parse_entities(raw) -> dict:
    """
    Parse entities_json column value into a Python dict.

    Handles:
      - NaN / None          → empty dict
      - already a dict      → return as-is
      - JSON string         → parse with json.loads()
    """
    if raw is None:
        return {}
    try:
        import math
        if isinstance(raw, float) and math.isnan(raw):
            return {}
    except (TypeError, ValueError):
        pass
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_annotated_entities(
    annotated_csv: str,
    ids: List[int],
) -> Dict[int, dict]:
    """
    Load pre-computed NER from annotated_csv.

    Returns a dict mapping {id → parsed entities dict}.
    If the file does not exist, returns an empty dict (runs in reduced-capability mode).
    """
    if not os.path.isfile(annotated_csv):
        print(
            f"[Verify] WARNING: Annotated CSV not found at '{annotated_csv}'.\n"
            "         Proceeding with empty entities — negation, completeness, and\n"
            "         appropriateness dimensions will have limited signal."
        )
        return {}

    print(f"[Verify] Loading pre-computed NER from: {annotated_csv}")
    ann_df = pd.read_csv(annotated_csv)

    if "entities_json" not in ann_df.columns:
        print(
            "[Verify] WARNING: 'entities_json' column not found in annotated CSV.\n"
            "         Proceeding with empty entities."
        )
        return {}

    entity_map: Dict[int, dict] = {}
    for _, row in ann_df.iterrows():
        row_id = int(row["id"])
        entity_map[row_id] = _parse_entities(row.get("entities_json"))

    print(f"[Verify] Loaded entities for {len(entity_map)} rows.")
    return entity_map


# ── run_verification ─────────────────────────────────────────────────────────────────────

def run_verification(
    submission_csv: str,
    eval_csv: str,
    annotated_csv: str,
    output_csv: str,
    pass_threshold: float = 0.6,
) -> None:
    """
    Load generated notes, verify each one with the 5-dimension clinical verifier,
    apply rule-based corrections to failing notes, and save improved results.

    Parameters
    ----------
    submission_csv : str
        Path to generated notes CSV (columns: id, generated_note).
    eval_csv : str
        Path to eval data CSV (columns: id, note, dialogue).
    annotated_csv : str
        Path to pre-computed NER CSV (columns: id, entities_json).
        If not found, verifier runs with reduced capability.
    output_csv : str
        Destination path for the corrected notes CSV (same schema as submission_csv).
    pass_threshold : float
        Minimum overall_score for a note to be considered passing.
    """
    total_start = time.perf_counter()

    # ── load data ────────────────────────────────────────────────────────────────────────
    print(f"\n[Verify] Loading submission: {submission_csv}")
    sub_df = pd.read_csv(submission_csv)
    sub_df["generated_note"] = sub_df["generated_note"].apply(str)

    print(f"[Verify] Loading eval data:   {eval_csv}")
    eval_df = pd.read_csv(eval_csv)
    eval_df["dialogue"] = eval_df["dialogue"].apply(str)

    merged = pd.merge(
        sub_df,
        eval_df[["id", "dialogue", "note"]],
        on="id",
        how="inner",
    )
    merged["note"] = merged["note"].apply(str)
    n = len(merged)
    print(f"[Verify] {n} notes to verify (threshold={pass_threshold}).")

    ids: List[int]       = merged["id"].tolist()
    notes: List[str]     = merged["generated_note"].tolist()
    dialogues: List[str] = merged["dialogue"].tolist()

    # ── load NER entities ────────────────────────────────────────────────────────────────
    entity_map = _load_annotated_entities(annotated_csv, ids)

    # ── initialise verifier ──────────────────────────────────────────────────────────────
    verifier = ClinicalVerifier(pass_threshold=pass_threshold)

    # ── verify loop ──────────────────────────────────────────────────────────────────────
    print(f"\n[Verify] Starting verification …")
    results: List[VerificationResult] = []
    corrected_notes: List[str] = []

    dim_names = ["faithfulness", "negation", "completeness", "consistency", "appropriateness"]
    dim_sums: Dict[str, float] = {d: 0.0 for d in dim_names}
    total_corrections = 0

    for i, (row_id, note, dialogue) in enumerate(zip(ids, notes, dialogues)):
        entities = entity_map.get(row_id, {})
        result = verifier.verify(
            note=note,
            dialogue_entities=entities,
            dialogue_text=dialogue,
        )
        results.append(result)

        # use corrected note only when the original failed
        final_note = result.note_corrected if not result.passed else note
        corrected_notes.append(final_note)
        total_corrections += result.corrections_applied

        for dim in dim_names:
            dim_sums[dim] += result.dimensions[dim].score

        # progress every 100 notes
        if (i + 1) % 100 == 0 or (i + 1) == n:
            mean_so_far = sum(r.overall_score for r in results) / len(results)
            print(
                f"[Verify] Verified {i+1}/{n}  "
                f"(mean score so far: {mean_so_far:.3f})"
            )

    # ── aggregate stats ──────────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r.passed)
    failed_count = n - passed_count
    mean_overall = sum(r.overall_score for r in results) / n
    dim_means = {d: dim_sums[d] / n for d in dim_names}
    elapsed = time.perf_counter() - total_start

    # ── save output ──────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df = pd.DataFrame({"id": ids, "generated_note": corrected_notes})
    out_df.to_csv(output_csv, index=False)
    print(f"\n[Verify] Results saved to: {output_csv}")

    # ── verification report ──────────────────────────────────────────────────────────────
    col = 20
    print(
        f"\n{'─'*55}\n"
        f" Verification Report\n"
        f"{'─'*55}\n"
        f"  {'Total notes:':<{col}} {n}\n"
        f"  {'Passed:':<{col}} {passed_count} ({passed_count/n*100:.1f}%)\n"
        f"  {'Failed:':<{col}} {failed_count}\n"
        f"  {'Mean overall score:':<{col}} {mean_overall:.4f}\n"
        f"\n  Per-dimension mean scores:\n"
    )
    for dim in dim_names:
        print(f"    {dim+':':<20} {dim_means[dim]:.4f}")
    print(
        f"\n  {'Total corrections:':<{col}} {total_corrections}\n"
        f"  {'Time elapsed:':<{col}} {elapsed:.1f}s\n"
        f"{'─'*55}"
    )


# ── compare_before_after ─────────────────────────────────────────────────────────────────

def compare_before_after(
    original_csv: str,
    verified_csv: str,
    ground_truth_csv: str,
) -> Dict:
    """
    Evaluate both the original and verified submissions against ground truth,
    and print a side-by-side comparison table.

    Parameters
    ----------
    original_csv : str
        Path to pre-verification generated notes (columns: id, generated_note).
    verified_csv : str
        Path to post-verification corrected notes (columns: id, generated_note).
    ground_truth_csv : str
        Path to ground truth CSV (columns: id, note, dialogue).

    Returns
    -------
    dict with "before" and "after" score dicts, each containing
    bleu, rouge1, rouge2, rougeL, rougeLsum, meteor.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError:
        print("[Compare] ERROR: 'evaluate' package not installed. Run: pip install evaluate")
        sys.exit(1)

    def _score(pred_csv: str, label: str) -> Dict[str, float]:
        pred_df = pd.read_csv(pred_csv)
        gt_df   = pd.read_csv(ground_truth_csv)
        merged  = pd.merge(pred_df, gt_df[["id", "note"]], on="id", how="inner")
        preds   = merged["generated_note"].apply(str).tolist()
        refs    = merged["note"].apply(str).tolist()
        print(f"[Compare] Scoring '{label}' ({len(merged)} examples) …")

        bleu   = hf_evaluate.load("bleu").compute(
            predictions=preds, references=[[r] for r in refs]
        )
        rouge  = hf_evaluate.load("rouge").compute(
            predictions=preds, references=refs
        )
        meteor = hf_evaluate.load("meteor").compute(
            predictions=preds, references=refs
        )
        return {
            "bleu":      bleu["bleu"],
            "rouge1":    rouge["rouge1"],
            "rouge2":    rouge["rouge2"],
            "rougeL":    rouge["rougeL"],
            "rougeLsum": rouge["rougeLsum"],
            "meteor":    meteor["meteor"],
        }

    before = _score(original_csv, "before")
    after  = _score(verified_csv,  "after")

    # ── print comparison table ───────────────────────────────────────────────────────────
    metrics = ["bleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor"]
    col_m, col_v = 12, 9

    header = (
        f"\n{'─'*46}\n"
        f" {'Metric':<{col_m}} {'Before':>{col_v}} {'After':>{col_v}} {'Delta':>{col_v}}\n"
        f"{'─'*46}"
    )
    print(header)
    for m in metrics:
        b = before[m]
        a = after[m]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(
            f" {m:<{col_m}} {b:>{col_v}.4f} {a:>{col_v}.4f} "
            f"{sign}{delta:>{col_v-1}.4f}"
        )
    print(f"{'─'*46}\n")

    return {"before": before, "after": after}


# ── CLI entry point ──────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Dial2Note verification pipeline: "
            "load generated notes → verify → correct → save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--submission",
        default="outputs/submission_rag.csv",
        help="Generated notes CSV from inference_pipeline.py",
    )
    p.add_argument(
        "--eval_csv",
        default="data/processed/shared_task_eval.csv",
        help="Eval data CSV with dialogue and gold notes",
    )
    p.add_argument(
        "--annotated_csv",
        default="outputs/dev_annotated.csv",
        help="Pre-computed NER CSV (optional — verifier runs with reduced capability if missing)",
    )
    p.add_argument(
        "--output_csv",
        default="outputs/submission_verified.csv",
        help="Where to save corrected notes",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Minimum overall score to consider a note passing",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run BLEU/ROUGE/METEOR comparison between original and verified notes",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    run_verification(
        submission_csv=args.submission,
        eval_csv=args.eval_csv,
        annotated_csv=args.annotated_csv,
        output_csv=args.output_csv,
        pass_threshold=args.threshold,
    )

    if args.compare:
        compare_before_after(
            original_csv=args.submission,
            verified_csv=args.output_csv,
            ground_truth_csv=args.eval_csv,
        )
