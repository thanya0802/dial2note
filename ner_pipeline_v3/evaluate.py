"""
NER Pipeline v3 — Evaluation script.
Computes quality metrics on annotated CSV output.

Usage:
    python -m ner_pipeline_v3.evaluate --input ner_pipeline_v3/outputs/train_annotated_v3.csv
    python -m ner_pipeline_v3.evaluate --input ner_pipeline_v3/outputs/dev_annotated_v3.csv
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

BODY_PARTS = {
    "skin", "nails", "stomach", "abdomen", "chest", "head", "neck", "back",
    "arm", "leg", "foot", "hand", "eye", "ear", "nose", "throat", "heart",
    "lung", "liver", "kidney",
}

# Simple keyword lists for clinical coverage scoring (used to mine gold notes)
CLINICAL_KEYWORDS = {
    "symptoms": [
        "pain", "fatigue", "nausea", "vomiting", "dizziness", "headache",
        "cough", "fever", "shortness of breath", "dyspnea", "palpitations",
        "weight loss", "weight gain", "anorexia", "insomnia", "sweating",
        "chills", "weakness", "numbness", "tingling", "swelling", "bloating",
        "diarrhea", "constipation", "bleeding", "rash", "itching", "anxiety",
        "depression", "confusion", "syncope", "fainting", "tremor",
    ],
    "diagnoses": [
        "diabetes", "hypertension", "asthma", "copd", "anemia", "cancer",
        "hypothyroidism", "hyperthyroidism", "depression", "anxiety",
        "pneumonia", "infection", "obesity", "hyperlipidemia", "arthritis",
        "osteoporosis", "gerd", "ibs", "uti", "sinusitis", "migraine",
        "atrial fibrillation", "heart failure", "coronary artery disease",
    ],
    "medications": [
        "metformin", "lisinopril", "atorvastatin", "omeprazole", "levothyroxine",
        "amlodipine", "metoprolol", "aspirin", "ibuprofen", "acetaminophen",
        "albuterol", "sertraline", "fluoxetine", "amoxicillin", "prednisone",
        "insulin", "warfarin", "furosemide", "gabapentin", "hydrochlorothiazide",
    ],
    "tests": [
        "cbc", "bmp", "cmp", "hba1c", "tsh", "ecg", "ekg", "x-ray", "mri",
        "ct scan", "ultrasound", "urinalysis", "blood culture", "lipid panel",
        "thyroid", "glucose", "creatinine", "hemoglobin", "a1c", "echo",
        "echocardiogram", "colonoscopy", "mammogram", "pap smear",
    ],
}


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _parse_entities(entities_json: str) -> list[dict]:
    try:
        return json.loads(entities_json) if entities_json else []
    except (json.JSONDecodeError, TypeError):
        return []


def metric_usefulness(df: pd.DataFrame) -> dict:
    """
    Entity Usefulness Score: fraction of extracted entity texts that appear
    as substrings in the gold SOAP note (case-insensitive).
    Target: > 0.70
    """
    row_scores = []
    for _, row in df.iterrows():
        entities = _parse_entities(row.get("entities_json", "[]"))
        note = str(row.get("note", "")).lower()
        if not entities:
            continue
        useful = sum(1 for e in entities if e["text"].lower() in note)
        row_scores.append(useful / len(entities))

    if not row_scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "target": "> 0.70"}

    return {
        "mean": statistics.mean(row_scores),
        "median": statistics.median(row_scores),
        "std": statistics.stdev(row_scores) if len(row_scores) > 1 else 0.0,
        "target": "> 0.70",
    }


def metric_noise_ratio(df: pd.DataFrame) -> dict:
    """
    Noise Ratio: fraction of entities that are standalone body parts
    with <= 2 words. Target: < 0.10
    """
    row_ratios = []
    for _, row in df.iterrows():
        entities = _parse_entities(row.get("entities_json", "[]"))
        if not entities:
            continue
        noise = sum(
            1 for e in entities
            if len(e["text"].split()) <= 2 and e["text"].lower() in BODY_PARTS
        )
        row_ratios.append(noise / len(entities))

    return {
        "mean": statistics.mean(row_ratios) if row_ratios else 0.0,
        "target": "< 0.10",
    }


def metric_clinical_coverage(df: pd.DataFrame) -> dict:
    """
    Clinical Coverage Score: fraction of clinical keywords found in the gold
    SOAP note that were captured by at least one extracted entity.
    """
    all_keywords = [kw for kws in CLINICAL_KEYWORDS.values() for kw in kws]

    row_scores = []
    for _, row in df.iterrows():
        note = str(row.get("note", "")).lower()
        entities = _parse_entities(row.get("entities_json", "[]"))
        entity_texts = [e["text"].lower() for e in entities]

        # Keywords present in this gold note
        note_keywords = [kw for kw in all_keywords if kw in note]
        if not note_keywords:
            continue

        # How many of those keywords appear in at least one entity span
        captured = sum(
            1 for kw in note_keywords
            if any(kw in et or et in kw for et in entity_texts)
        )
        row_scores.append(captured / len(note_keywords))

    return {
        "mean": statistics.mean(row_scores) if row_scores else 0.0,
        "rows_evaluated": len(row_scores),
    }


def metric_label_distribution(df: pd.DataFrame) -> dict:
    """Count of entities per label type across the full dataset."""
    dist: Counter = Counter()
    for _, row in df.iterrows():
        entities = _parse_entities(row.get("entities_json", "[]"))
        for e in entities:
            dist[e["label"]] += 1
    return dict(dist)


def metric_span_lengths(df: pd.DataFrame) -> dict:
    """Word-count statistics for entity spans."""
    lengths = []
    for _, row in df.iterrows():
        entities = _parse_entities(row.get("entities_json", "[]"))
        for e in entities:
            lengths.append(len(e["text"].split()))

    if not lengths:
        return {}

    buckets = {
        "1_word": sum(1 for l in lengths if l == 1),
        "2_words": sum(1 for l in lengths if l == 2),
        "3_4_words": sum(1 for l in lengths if 3 <= l <= 4),
        "5_7_words": sum(1 for l in lengths if 5 <= l <= 7),
        "7plus_words": sum(1 for l in lengths if l > 7),
    }
    total = len(lengths)
    bucket_pct = {k: round(v / total * 100, 1) for k, v in buckets.items()}

    return {
        "mean": round(statistics.mean(lengths), 2),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "total_spans": total,
        "bucket_counts": buckets,
        "bucket_pct": bucket_pct,
    }


def metric_confidence_distribution(df: pd.DataFrame) -> dict:
    """Confidence score statistics. Should NOT be all-identical (v2 bug)."""
    scores = []
    for _, row in df.iterrows():
        entities = _parse_entities(row.get("entities_json", "[]"))
        for e in entities:
            scores.append(float(e.get("score", 0.0)))

    if not scores:
        return {}

    unique_vals = set(round(s, 4) for s in scores)
    buckets = {
        "lt_0.3": sum(1 for s in scores if s < 0.3),
        "0.3_0.5": sum(1 for s in scores if 0.3 <= s < 0.5),
        "0.5_0.7": sum(1 for s in scores if 0.5 <= s < 0.7),
        "0.7_0.9": sum(1 for s in scores if 0.7 <= s < 0.9),
        "gt_0.9": sum(1 for s in scores if s >= 0.9),
    }

    return {
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "unique_score_count": len(unique_vals),
        "all_identical_flag": len(unique_vals) == 1,
        "bucket_counts": buckets,
    }


# ─── Printing helpers ─────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def _print_metrics(metrics: dict) -> None:
    sep = "─" * 64

    print(f"\n{'═' * 64}")
    print("  NER Pipeline v3 — Evaluation Report")
    print(f"{'═' * 64}")

    # ── Usefulness ──
    u = metrics["entity_usefulness_score"]
    print(f"\n  1. Entity Usefulness Score  (target > 70%)")
    print(f"  {sep}")
    print(f"     Mean   : {u['mean']:.1%}  {_bar(u['mean'])}")
    print(f"     Median : {u['median']:.1%}")
    print(f"     Std    : {u['std']:.1%}")
    status = "✓ PASS" if u["mean"] >= 0.70 else "✗ FAIL"
    print(f"     Status : {status}")

    # ── Noise ──
    n = metrics["noise_ratio"]
    print(f"\n  2. Noise Ratio  (target < 10%)")
    print(f"  {sep}")
    print(f"     Mean   : {n['mean']:.1%}  {_bar(n['mean'])}")
    status = "✓ PASS" if n["mean"] < 0.10 else "✗ FAIL"
    print(f"     Status : {status}")

    # ── Coverage ──
    c = metrics["clinical_coverage_score"]
    print(f"\n  3. Clinical Coverage Score")
    print(f"  {sep}")
    print(f"     Mean   : {c['mean']:.1%}  {_bar(c['mean'])}")
    print(f"     Rows   : {c['rows_evaluated']}")

    # ── Label Distribution ──
    ld = metrics["label_distribution"]
    print(f"\n  4. Label Distribution")
    print(f"  {sep}")
    total_ents = sum(ld.values())
    for label, count in sorted(ld.items(), key=lambda x: -x[1]):
        pct = count / total_ents if total_ents else 0
        print(f"     {label:<30} {count:>5}  ({pct:.1%})")
    print(f"     {'TOTAL':<30} {total_ents:>5}")

    # ── Span Lengths ──
    sl = metrics["span_length_stats"]
    if sl:
        print(f"\n  5. Span Length Statistics")
        print(f"  {sep}")
        print(f"     Mean / Median : {sl['mean']} / {sl['median']} words")
        print(f"     Min / Max     : {sl['min']} / {sl['max']} words")
        print(f"     Total spans   : {sl['total_spans']}")
        print(f"     Distribution:")
        for bucket, pct in sl["bucket_pct"].items():
            label = bucket.replace("_", " ")
            print(f"       {label:<14} {pct:>5.1f}%  {_bar(pct / 100, 20)}")
        if sl["bucket_pct"].get("7plus_words", 0) > 0:
            print("     ⚠  Spans > 7 words found — review span cleaning.")

    # ── Confidence ──
    cd = metrics["confidence_distribution"]
    if cd:
        print(f"\n  6. Confidence Score Distribution")
        print(f"  {sep}")
        print(f"     Mean / Median : {cd['mean']} / {cd['median']}")
        print(f"     Min / Max     : {cd['min']} / {cd['max']}")
        print(f"     Unique values : {cd['unique_score_count']}")
        if cd["all_identical_flag"]:
            print("     ⚠  ALL SCORES IDENTICAL — model may be misconfigured!")
        print(f"     Buckets:")
        total_s = sum(cd["bucket_counts"].values())
        for bucket, count in cd["bucket_counts"].items():
            pct = count / total_s if total_s else 0
            print(f"       {bucket:<12} {count:>6}  ({pct:.1%})")

    print(f"\n{'═' * 64}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NER Pipeline v3 — Evaluate annotated output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to annotated CSV (must have columns: id, note, dialogue, entities_json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        raise SystemExit(1)

    print(f"\n  Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Rows loaded: {len(df)}")

    metrics = {
        "entity_usefulness_score": metric_usefulness(df),
        "noise_ratio": metric_noise_ratio(df),
        "clinical_coverage_score": metric_clinical_coverage(df),
        "label_distribution": metric_label_distribution(df),
        "span_length_stats": metric_span_lengths(df),
        "confidence_distribution": metric_confidence_distribution(df),
    }

    _print_metrics(metrics)

    # Save JSON
    output_dir = "ner_pipeline_v3/outputs"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "eval_metrics_v3.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {json_path}\n")


if __name__ == "__main__":
    main()
