"""
NER Pipeline v3 — Core extraction logic.
2-pass: GLiNER broad labels → rule-based SOAP splitting.
"""

import re
from gliner import GLiNER

from ner_pipeline_v3.config import (
    GLINER_MODEL,
    LABELS,
    DESCRIPTION_TO_LABEL,
    PRIMARY_THRESHOLD,
    FALLBACK_THRESHOLD,
    MAX_SPAN_WORDS,
    MIN_ENTITIES,
    MUST_HAVE_CC,
    SPECIALIST_PATTERNS,
    DOCTOR_PATTERN,
    DOSAGE_PATTERN,
    VITAL_PATTERN,
    DIAGNOSIS_KEYWORDS,
)

_FILLER_PREFIXES = [
    "i've also been having ",
    "i've also been ",
    "i've been having ",
    "i've been ",
    "i'm having ",
    "i also have ",
    "i have ",
    "patient reports ",
    "patient denies ",
    "patient has ",
    "the ",
    "my ",
]

_FILLER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _FILLER_PREFIXES) + r")",
    re.IGNORECASE,
)


class ClinicalExtractorV3:

    def __init__(self, model_name: str = GLINER_MODEL) -> None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model_name)
        self.model = self.model.to(device)
        print(f"  GLiNER loaded on: {device}")

    def extract(self, dialogue_text: str) -> list[dict]:
        if not dialogue_text or not dialogue_text.strip():
            return []

        # Pass 1: GLiNER with 4 broad labels
        entities = self._run_gliner(dialogue_text, PRIMARY_THRESHOLD)
        entities = self._filter_confidence(entities, dialogue_text)
        entities = self._clean_spans(entities)

        # Pass 2: Rule-based splitting into 10 SOAP labels
        entities = self._split_to_soap_labels(entities)

        entities = self._deduplicate(entities)
        return entities

    # ── Pass 1: GLiNER ────────────────────────────────────────

    def _run_gliner(self, text: str, threshold: float) -> list[dict]:
        import re as _re

        turns = _re.split(r'(?=\[(?:doctor|patient)\]:)', text, flags=_re.IGNORECASE)
        turns = [t.strip() for t in turns if t.strip()]

        if len(turns) <= 1:
            turns = [s.strip() for s in text.split("\n") if s.strip()]

        MAX_WORDS = 200
        chunks = []
        current_chunk = []
        current_len = 0

        for turn in turns:
            turn_len = len(turn.split())
            if current_len + turn_len > MAX_WORDS and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [turn]
                current_len = turn_len
            else:
                current_chunk.append(turn)
                current_len += turn_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        all_entities = []
        for chunk in chunks:
            if len(chunk.strip()) < 5:
                continue
            raw = self.model.predict_entities(chunk, LABELS, threshold=threshold)
            for ent in raw:
                all_entities.append(
                    {
                        "text": ent["text"],
                        "label": DESCRIPTION_TO_LABEL.get(ent["label"], ent["label"]),
                        "score": float(ent["score"]),
                    }
                )

        return all_entities

    # ── Pass 1b: Confidence filtering ─────────────────────────

    def _filter_confidence(self, entities: list[dict], text: str) -> list[dict]:
        filtered = [e for e in entities if e["score"] >= PRIMARY_THRESHOLD]

        if len(filtered) < MIN_ENTITIES:
            filtered = self._run_gliner(text, FALLBACK_THRESHOLD)
            filtered = [e for e in filtered if e["score"] >= FALLBACK_THRESHOLD]

        return filtered

    # ── Pass 1c: Span cleaning ────────────────────────────────

    def _clean_spans(self, entities: list[dict]) -> list[dict]:
        cleaned = []
        for ent in entities:
            text = ent["text"].strip()
            text = text.replace('\u2019', "'").replace('\u2018', "'")

            if len(text.split()) > MAX_SPAN_WORDS:
                continue

            prev = None
            while prev != text:
                prev = text
                text = _FILLER_RE.sub("", text).strip()

            if not text or len(text) <= 1:
                continue
            if len(text.split()) > MAX_SPAN_WORDS:
                continue

            cleaned.append({**ent, "text": text})
        return cleaned

    # ── Pass 2: Rule-based SOAP splitting ─────────────────────

    def _split_to_soap_labels(self, entities: list[dict]) -> list[dict]:
        """
        Split 4 broad GLiNER labels into 10 SOAP-specific labels
        using pattern matching and keyword lookup.
        """
        result = []

        for ent in entities:
            text = ent["text"]
            text_lower = text.lower().strip()
            broad_label = ent["label"]
            score = ent["score"]

            if broad_label == "subjective finding":
                # Check if it's a known diagnosis
                if any(kw in text_lower for kw in DIAGNOSIS_KEYWORDS):
                    fine_label = "diagnosis"
                else:
                    fine_label = "symptom"

            elif broad_label == "objective finding":
                if VITAL_PATTERN.search(text):
                    fine_label = "vital sign"
                elif any(kw in text_lower for kw in DIAGNOSIS_KEYWORDS):
                    fine_label = "diagnosis"
                else:
                    fine_label = "physical exam finding"

            elif broad_label == "treatment":
                if DOCTOR_PATTERN.search(text) or SPECIALIST_PATTERNS.search(text):
                    fine_label = "referral"
                elif DOSAGE_PATTERN.search(text):
                    fine_label = "drug detail"
                elif text_lower in ['follow-up', 'follow up'] or any(
                    kw in text_lower for kw in [
                        'follow-up', 'follow up', 'week', 'month',
                        'return', 'appointment', 'lifestyle', 'exercise',
                        'diet', 'yoga', 'meditation', 'meal plan',
                        'physical therapy', 'occupational therapy',
                        'monitor', 'check-up', 'recheck',
                    ]
                ):
                    fine_label = "follow-up instruction"
                else:
                    fine_label = "medication"

            elif broad_label == "test":
                fine_label = "test or lab order"

            else:
                fine_label = broad_label

            result.append({
                "text": text,
                "label": fine_label,
                "score": score,
            })

        # CC promotion: first/highest-scoring symptom becomes chief complaint
        if MUST_HAVE_CC:
            symptoms = [e for e in result if e["label"] == "symptom"]
            if symptoms:
                best = max(symptoms, key=lambda e: e["score"])
                for e in result:
                    if e["text"] == best["text"] and e["label"] == "symptom":
                        e["label"] = "chief complaint"
                        break

        return result

    # ── Pass 3: Deduplication ─────────────────────────────────

    def _deduplicate(self, entities: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for ent in entities:
            key = ent["text"].lower().strip()
            if key not in seen or ent["score"] > seen[key]["score"]:
                seen[key] = ent
        unique = list(seen.values())

        unique.sort(key=lambda e: len(e["text"]), reverse=True)
        keep: list[dict] = []
        for candidate in unique:
            cand_text = candidate["text"].lower()
            cand_label = candidate["label"]
            dominated = False
            for kept in keep:
                if kept["label"] == cand_label:
                    if cand_text in kept["text"].lower():
                        dominated = True
                        break
            if not dominated:
                keep.append(candidate)

        return keep
