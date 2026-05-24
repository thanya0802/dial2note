"""
Phase 7 – Evaluation & Ablation Metrics
=========================================

Surface-level metrics:
  A. ``compute_text_metrics``       – BLEU / ROUGE / METEOR
  B. ``compute_medcon_if_available`` – MedCon (graceful fallback)

Entity-level metrics:
  C. ``extract_entities_from_texts`` + ``compute_entity_metrics``
     – NER-based precision / recall / F1

Structural checks:
  D. ``compute_structure_metrics``   – SOAP header presence checks

LLM judge:
  E. ``build_llm_judge_prompt``      – prompt template (no API call)

Clinical deep metrics (Phase 7 extension):
  G. ``normalize_text``              – fair text comparison
  H. ``split_sections``              – robust SOAP section splitter
  I. ``compute_section_rouge``       – per-section ROUGE-L
  J. ``compute_hallucination_rate``  – entity grounding check
  K. ``compute_symptom_recall``      – clinical completeness
  L. ``compute_length_stats``        – output length stability

Orchestrator:
  F. ``run_full_evaluation``         – combines all of the above
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ====================================================================
# A.  Text Metrics (BLEU / ROUGE / METEOR)
# ====================================================================

def compute_text_metrics(
    preds: List[str],
    refs: List[str],
) -> Dict[str, Any]:
    """
    Compute corpus-level BLEU, ROUGE-{1,2,L,Lsum}, and METEOR.

    Each metric is loaded from Hugging Face ``evaluate``.  If a metric
    is unavailable it is silently skipped and reported as ``None``.
    """
    results: Dict[str, Any] = {}

    # ── BLEU ─────────────────────────────────────────────────────
    try:
        import evaluate
        bleu = evaluate.load("bleu")
        bleu_out = bleu.compute(
            predictions=preds,
            references=[[r] for r in refs],
        )
        results["bleu"] = bleu_out["bleu"]
    except Exception as exc:
        print(f"  [WARN] BLEU unavailable: {exc}")
        results["bleu"] = None

    # ── ROUGE ────────────────────────────────────────────────────
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        rouge_out = rouge.compute(predictions=preds, references=refs)
        results["rouge1"]    = rouge_out.get("rouge1")
        results["rouge2"]    = rouge_out.get("rouge2")
        results["rougeL"]    = rouge_out.get("rougeL")
        results["rougeLsum"] = rouge_out.get("rougeLsum")
    except Exception as exc:
        print(f"  [WARN] ROUGE unavailable: {exc}")
        results.update(rouge1=None, rouge2=None, rougeL=None, rougeLsum=None)

    # ── METEOR ───────────────────────────────────────────────────
    try:
        import evaluate
        meteor = evaluate.load("meteor")
        meteor_out = meteor.compute(predictions=preds, references=refs)
        results["meteor"] = meteor_out["meteor"]
    except Exception as exc:
        print(f"  [WARN] METEOR unavailable: {exc}")
        results["meteor"] = None

    return results


# ====================================================================
# B.  MedCon (optional)
# ====================================================================

def compute_medcon_if_available(
    preds: List[str],
    refs: List[str],
) -> Dict[str, Any]:
    """Try to compute MedCon; return ``None`` gracefully if missing."""
    try:
        import evaluate
        medcon = evaluate.load("medcon")
        score = medcon.compute(predictions=preds, references=refs)
        return {"medcon": score.get("medcon", score), "medcon_available": True}
    except Exception:
        return {"medcon": None, "medcon_available": False}


# ====================================================================
# C.  Entity-level Metrics
# ====================================================================

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def _normalize_entity_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", t).strip()


def extract_entities_from_texts(
    texts: List[str],
    ner_model,
    batch_size: int = 16,
) -> List[Set[Tuple[str, str]]]:
    """
    Run ``ClinicalNER.extract_entities`` on every text and return a
    list of normalised entity sets.

    Each entity is a ``(label, normalised_text)`` tuple.
    Only **non-negated** entities are kept.
    """
    all_entity_sets: List[Set[Tuple[str, str]]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_results = ner_model.extract_entities(batch)

        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        for ner_out in batch_results:
            ents: Set[Tuple[str, str]] = set()
            for e in ner_out.get("entities", []):
                if not e.get("negated", False):
                    norm = _normalize_entity_text(e["text"])
                    if norm:
                        ents.add((e["label"], norm))
            all_entity_sets.append(ents)

    return all_entity_sets


def compute_entity_metrics(
    pred_entity_sets: List[Set[Tuple[str, str]]],
    gold_entity_sets: List[Set[Tuple[str, str]]],
) -> Dict[str, Any]:
    """
    Micro-averaged entity precision / recall / F1, plus per-label
    breakdown.
    """
    assert len(pred_entity_sets) == len(gold_entity_sets)

    total_tp = total_fp = total_fn = 0
    label_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for pred_set, gold_set in zip(pred_entity_sets, gold_entity_sets):
        tp = pred_set & gold_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for label, _ in tp:
            label_counts[label]["tp"] += 1
        for label, _ in fp:
            label_counts[label]["fp"] += 1
        for label, _ in fn:
            label_counts[label]["fn"] += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)

    per_label: Dict[str, Dict[str, Any]] = {}
    for label, counts in sorted(label_counts.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-12)
        per_label[label] = {"p": round(p, 4), "r": round(r, 4),
                            "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        "entity_precision": round(precision, 4),
        "entity_recall":    round(recall, 4),
        "entity_f1":        round(f1, 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "per_label": per_label,
    }


# ====================================================================
# D.  SOAP Structure Checks
# ====================================================================

_SOAP_HEADERS = {
    "Subjective": re.compile(
        r"\*{0,2}\s*(?:1[\.\):]?\s*)?Subjective\s*:?\s*\*{0,2}", re.IGNORECASE,
    ),
    "Objective": re.compile(
        r"\*{0,2}\s*(?:2[\.\):]?\s*)?Objective\s*:?\s*\*{0,2}", re.IGNORECASE,
    ),
    "Assessment": re.compile(
        r"\*{0,2}\s*(?:3[\.\):]?\s*)?Assessment\s*:?\s*\*{0,2}", re.IGNORECASE,
    ),
    "Plan": re.compile(
        r"\*{0,2}\s*(?:4[\.\):]?\s*)?Plan\s*:?\s*\*{0,2}", re.IGNORECASE,
    ),
}


def compute_structure_metrics(preds: List[str]) -> Dict[str, Any]:
    """
    Check how many predicted notes contain the required SOAP headers.
    """
    n = len(preds)
    if n == 0:
        return {
            "percent_with_all_headers": 0.0,
            "percent_with_each_header": {h: 0.0 for h in _SOAP_HEADERS},
        }

    header_counts = {h: 0 for h in _SOAP_HEADERS}
    all_present_count = 0

    for note in preds:
        found = {h: bool(pat.search(note)) for h, pat in _SOAP_HEADERS.items()}
        for h, present in found.items():
            header_counts[h] += int(present)
        if all(found.values()):
            all_present_count += 1

    return {
        "percent_with_all_headers": round(100 * all_present_count / n, 2),
        "percent_with_each_header": {
            h: round(100 * cnt / n, 2)
            for h, cnt in header_counts.items()
        },
    }


# ====================================================================
# G.  Text Normalization
# ====================================================================

def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for fair comparison."""
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


# ====================================================================
# H.  SOAP Section Splitter (strict regex for gold/pred comparison)
# ====================================================================

_STRICT_SECTION_PAT = re.compile(
    r"\*\*1\.\s*Subjective:\*\*(.*?)"
    r"\*\*2\.\s*Objective:\*\*(.*?)"
    r"\*\*3\.\s*Assessment:\*\*(.*?)"
    r"\*\*4\.\s*Plan:\*\*(.*)",
    re.DOTALL,
)

_LOOSE_SECTION_PATS = [
    ("Subjective", re.compile(
        r"(?:\*{0,2}\s*(?:1[\.\):]?\s*)?Subjective\s*:?\s*\*{0,2})", re.IGNORECASE)),
    ("Objective", re.compile(
        r"(?:\*{0,2}\s*(?:2[\.\):]?\s*)?Objective\s*:?\s*\*{0,2})", re.IGNORECASE)),
    ("Assessment", re.compile(
        r"(?:\*{0,2}\s*(?:3[\.\):]?\s*)?Assessment\s*:?\s*\*{0,2})", re.IGNORECASE)),
    ("Plan", re.compile(
        r"(?:\*{0,2}\s*(?:4[\.\):]?\s*)?Plan\s*:?\s*\*{0,2})", re.IGNORECASE)),
]


def split_sections(note: str) -> Dict[str, str]:
    """
    Extract Subjective, Objective, Assessment, Plan sections.

    Tries strict ``**1. Subjective:**`` format first, then falls back
    to a loose header scan.  Missing sections return empty strings.
    """
    empty = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
    if not note:
        return empty

    m = _STRICT_SECTION_PAT.search(note)
    if m:
        return {
            "Subjective": m.group(1).strip(),
            "Objective":  m.group(2).strip(),
            "Assessment": m.group(3).strip(),
            "Plan":       m.group(4).strip(),
        }

    # Loose fallback — find header positions and slice between them
    found: List[tuple] = []
    for name, pat in _LOOSE_SECTION_PATS:
        hit = pat.search(note)
        if hit:
            found.append((name, hit.end()))
    if not found:
        return empty

    found.sort(key=lambda x: x[1])
    result = dict(empty)
    for i, (name, body_start) in enumerate(found):
        body_end = found[i + 1][1] if i + 1 < len(found) else len(note)
        # walk back to start of header line for the next section
        if i + 1 < len(found):
            pos = found[i + 1][1]
            while pos > 0 and note[pos - 1] != "\n":
                pos -= 1
            body_end = pos
        result[name] = note[body_start:body_end].strip()
    return result


# ====================================================================
# I.  Section-wise ROUGE
# ====================================================================

def compute_section_rouge(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """
    Compute average ROUGE-L per SOAP section across all examples.
    """
    try:
        import evaluate
        rouge = evaluate.load("rouge")
    except Exception:
        return {s: 0.0 for s in ("Subjective", "Objective", "Assessment", "Plan")}

    section_scores: Dict[str, List[float]] = {
        "Subjective": [], "Objective": [], "Assessment": [], "Plan": [],
    }

    for pred, ref in zip(preds, refs):
        pred_sec = split_sections(pred)
        ref_sec  = split_sections(ref)

        for sec in section_scores:
            p_text = normalize_text(pred_sec[sec])
            r_text = normalize_text(ref_sec[sec])
            if not p_text and not r_text:
                section_scores[sec].append(1.0)
            elif not p_text or not r_text:
                section_scores[sec].append(0.0)
            else:
                out = rouge.compute(predictions=[p_text], references=[r_text])
                section_scores[sec].append(out.get("rougeL", 0.0))

    return {
        sec: round(float(np.mean(vals)), 4) if vals else 0.0
        for sec, vals in section_scores.items()
    }


# ====================================================================
# J.  Entity-Based Hallucination Rate
# ====================================================================

def compute_hallucination_rate(
    preds: List[str],
    dialogues: List[str],
    ner_model,
) -> float:
    """
    Fraction of entities in the generated note that are NOT grounded
    in the source dialogue.  Lower is better.
    """
    total_generated = 0
    hallucinated = 0

    for pred, dialogue in zip(preds, dialogues):
        pred_ents = ner_model.extract_entities(pred)
        if isinstance(pred_ents, list):
            pred_ents = pred_ents[0]
        dlg_ents = ner_model.extract_entities(dialogue)
        if isinstance(dlg_ents, list):
            dlg_ents = dlg_ents[0]

        dlg_texts = {
            normalize_text(e["text"])
            for e in dlg_ents.get("entities", [])
        }

        for e in pred_ents.get("entities", []):
            total_generated += 1
            if normalize_text(e["text"]) not in dlg_texts:
                hallucinated += 1

    return round(hallucinated / max(total_generated, 1), 4)


# ====================================================================
# K.  Symptom Recall (Clinical Completeness)
# ====================================================================

# Labels that count as symptoms for recall (GLiNER + legacy)
_SYMPTOM_LABELS = {"SYMPTOM", "S_CC_COMPLAINT", "S_HPI_SYMPTOM"}


def compute_symptom_recall(
    preds: List[str],
    dialogues: List[str],
    ner_model,
) -> float:
    """
    Fraction of non-negated symptom entities (S_CC_COMPLAINT, S_HPI_SYMPTOM, SYMPTOM)
    from the dialogue that appear somewhere in the predicted note.  Higher is better.
    """
    total_symptoms = 0
    covered = 0

    for pred, dialogue in zip(preds, dialogues):
        dlg_ents = ner_model.extract_entities(dialogue)
        if isinstance(dlg_ents, list):
            dlg_ents = dlg_ents[0]

        pred_norm = normalize_text(pred)

        for e in dlg_ents.get("entities", []):
            if e["label"] in _SYMPTOM_LABELS and not e.get("negated", False):
                total_symptoms += 1
                if normalize_text(e["text"]) in pred_norm:
                    covered += 1

    return round(covered / max(total_symptoms, 1), 4)


# ====================================================================
# L.  Output Length Stability
# ====================================================================

def compute_length_stats(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """
    Compare word-count distributions between predicted and gold notes.
    """
    pred_lengths = [len(p.split()) for p in preds]
    ref_lengths  = [len(r.split()) for r in refs]

    n = len(pred_lengths)
    over_2x = sum(
        1 for p, r in zip(pred_lengths, ref_lengths)
        if r > 0 and p > 2 * r
    )

    return {
        "avg_pred_length": round(float(np.mean(pred_lengths)), 1),
        "avg_ref_length":  round(float(np.mean(ref_lengths)), 1),
        "over_2x_ratio":   round(over_2x / max(n, 1), 4),
    }


# ====================================================================
# E.  LLM Judge Prompt Builder
# ====================================================================

def build_llm_judge_prompt(
    dialogue: str,
    pred_note: str,
    gold_note: Optional[str] = None,
) -> str:
    """
    Build a prompt that asks a language model to score a generated
    SOAP note on six quality dimensions (1–5 each).

    No API call is made — the caller decides how to use this prompt.
    """
    gold_block = ""
    if gold_note:
        gold_block = (
            "\n--- REFERENCE (Gold) SOAP Note ---\n"
            f"{gold_note}\n"
        )

    return (
        "You are an expert clinical documentation reviewer.\n"
        "Evaluate the GENERATED SOAP note against the original dialogue"
        " and (if provided) the reference note.\n"
        "\nScore each dimension from 1 (worst) to 5 (best):\n"
        "  1. Hallucination  – Are there unsupported facts not in the dialogue?\n"
        "  2. Omission       – Is key clinical information missing?\n"
        "  3. SOAP Structure – Are all four headers present and content placed correctly?\n"
        "  4. Clinical Consistency – Are findings, assessment, and plan logically aligned?\n"
        "  5. Plan Realism   – Are medications, dosages, and referrals appropriate?\n"
        "  6. Overall Quality – Holistic score considering all dimensions.\n"
        "\n--- Original Dialogue ---\n"
        f"{dialogue}\n"
        f"{gold_block}"
        "\n--- GENERATED SOAP Note ---\n"
        f"{pred_note}\n"
        "\nReturn your evaluation as a JSON object with exactly these keys:\n"
        "{\n"
        '  "hallucination": <1-5>,\n'
        '  "omission": <1-5>,\n'
        '  "soap_structure": <1-5>,\n'
        '  "clinical_consistency": <1-5>,\n'
        '  "plan_realism": <1-5>,\n'
        '  "overall": <1-5>,\n'
        '  "notes": "<short justification>"\n'
        "}\n"
        "\nJSON:"
    )


# ====================================================================
# F.  Full Evaluation Orchestrator
# ====================================================================

def run_full_evaluation(
    preds: List[str],
    refs: List[str],
    ner_model=None,
    dialogues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run all automated metrics and return a combined report dict.

    Parameters
    ----------
    preds : list[str]    Generated SOAP notes.
    refs  : list[str]    Gold SOAP notes (same length / order).
    ner_model             Optional ``ClinicalNER`` instance for
                          entity-level metrics.  If ``None`` the
                          entity section is skipped.
    dialogues : list[str] | None
                          Original dialogues (same length / order).
                          Required for hallucination rate and symptom
                          recall.  If ``None`` those metrics are skipped.
    """
    # Normalize all inputs for fair comparison
    preds_norm = [normalize_text(p) for p in preds]
    refs_norm  = [normalize_text(r) for r in refs]

    report: Dict[str, Any] = {}

    # ── Text metrics (on normalised text) ────────────────────────
    print("\n  Computing text metrics (BLEU / ROUGE / METEOR) …")
    report["text_metrics"] = compute_text_metrics(preds_norm, refs_norm)

    # ── MedCon ───────────────────────────────────────────────────
    print("  Checking MedCon availability …")
    report["medcon"] = compute_medcon_if_available(preds_norm, refs_norm)

    # ── Entity metrics ───────────────────────────────────────────
    if ner_model is not None:
        print("  Extracting entities from predictions …")
        pred_ents = extract_entities_from_texts(preds, ner_model)
        print("  Extracting entities from gold notes …")
        gold_ents = extract_entities_from_texts(refs, ner_model)
        report["entity_metrics"] = compute_entity_metrics(pred_ents, gold_ents)
    else:
        report["entity_metrics"] = None

    # ── Structure metrics ────────────────────────────────────────
    print("  Computing structure metrics …")
    report["structure_metrics"] = compute_structure_metrics(preds)

    # ── Section-wise ROUGE (on original text to preserve headers) ─
    print("  Computing section-wise ROUGE-L …")
    report["section_rouge"] = compute_section_rouge(preds, refs)

    # ── Length stats ─────────────────────────────────────────────
    print("  Computing length statistics …")
    report["length_stats"] = compute_length_stats(preds, refs)

    # ── Clinical deep metrics (need NER + dialogues) ─────────────
    if ner_model is not None and dialogues is not None:
        print("  Computing hallucination rate …")
        report["hallucination_rate"] = compute_hallucination_rate(
            preds, dialogues, ner_model,
        )
        print("  Computing symptom recall …")
        report["symptom_recall"] = compute_symptom_recall(
            preds, dialogues, ner_model,
        )
    else:
        report["hallucination_rate"] = None
        report["symptom_recall"] = None

    return report
