"""
NER Enrichment – UMLS Normalization + medspaCy Negation
=========================================================

Runs immediately after GLiNER to add:
  - UMLS concept mapping (umls_term, cui) via QuickUMLS or scispaCy
  - medspaCy negation detection (overrides heuristic when available)

UMLS normalization priority:
  1. QuickUMLS (QUICKUMLS_PATH) – when you have a local database
  2. scispaCy – auto-downloads UMLS KB from S3, works without setup (if installed)
  3. None – raw text when neither is available
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ────────────────────────────────────────────────────────────────────
# Fallback: common lay-to-clinical term mappings (use_fallback=True only)
# ────────────────────────────────────────────────────────────────────
_LAY_TO_CLINICAL: Dict[str, tuple] = {
    "tummy hurts": ("Abdominal Pain", None),
    "tummy ache": ("Abdominal Pain", None),
    "stomach ache": ("Abdominal Pain", None),
    "belly ache": ("Abdominal Pain", None),
    "stomach pain": ("Abdominal Pain", None),
    "abdominal pain": ("Abdominal Pain", None),
    "burning sensation": ("Burning sensation", None),
    "burning when urinating": ("Dysuria", None),
    "burning when peeing": ("Dysuria", None),
    "painful urination": ("Dysuria", None),
    "throwing up": ("Vomiting", None),
    "vomiting": ("Vomiting", None),
    "nausea": ("Nausea", None),
    "shortness of breath": ("Dyspnea", None),
    "can't breathe": ("Dyspnea", None),
    "difficulty breathing": ("Dyspnea", None),
    "trouble breathing": ("Dyspnea", None),
    "chest pain": ("Chest Pain", None),
    "fever": ("Fever", None),
    "headache": ("Headache", None),
    "headaches": ("Headache", None),
    "frequent urination": ("Urinary frequency", None),
    "urinary frequency": ("Urinary frequency", None),
    "fatigue": ("Fatigue", None),
    "tired": ("Fatigue", None),
    "cough": ("Cough", None),
    "coughing": ("Cough", None),
    "sore throat": ("Pharyngitis", None),
    "runny nose": ("Rhinorrhea", None),
    "congestion": ("Nasal congestion", None),
    "back pain": ("Back pain", None),
    "joint pain": ("Arthralgia", None),
    "muscle pain": ("Myalgia", None),
    "diarrhea": ("Diarrhea", None),
    "constipation": ("Constipation", None),
    "insomnia": ("Insomnia", None),
    "trouble sleeping": ("Insomnia", None),
    "anxiety": ("Anxiety", None),
    "depression": ("Depression", None),
    "dizziness": ("Dizziness", None),
    "lightheadedness": ("Dizziness", None),
}


def _normalize_with_fallback(span_text: str) -> Dict[str, Optional[str]]:
    """Use built-in dictionary for common terms when QuickUMLS unavailable."""
    key = span_text.strip().lower()
    if not key:
        return {"term": span_text, "cui": None}
    # Exact match
    if key in _LAY_TO_CLINICAL:
        term, cui = _LAY_TO_CLINICAL[key]
        return {"term": term, "cui": cui}
    # Substring match (e.g. "burning sensation when urinating" -> Dysuria)
    for phrase, (term, cui) in _LAY_TO_CLINICAL.items():
        if phrase in key or key in phrase:
            return {"term": term, "cui": cui}
    return {"term": span_text, "cui": None}


# ────────────────────────────────────────────────────────────────────
# UMLS Normalizer: QuickUMLS (preferred) or scispaCy (no-setup alternative)
# ────────────────────────────────────────────────────────────────────
def _get_quickumls_matcher(path: Optional[str] = None, threshold: float = 0.7):
    """Lazy-load QuickUMLS. Returns None if path unset or import fails."""
    qpath = path or os.environ.get("QUICKUMLS_PATH", "")
    if not qpath or not os.path.isdir(qpath):
        return None
    try:
        from quickumls import QuickUMLS
        return QuickUMLS(qpath, threshold=threshold)
    except Exception:
        return None


_scispacy_nlp = None


def _get_scispacy_nlp():
    """
    Lazy-load scispaCy pipeline for UMLS linking.
    Uses whole-span-as-entity so we can link arbitrary GLiNER spans.
    Returns None if scispacy not installed or model unavailable.
    """
    global _scispacy_nlp
    if _scispacy_nlp is not None:
        return _scispacy_nlp
    try:
        import spacy
        from spacy.tokens import Span
    except ImportError:
        return None

    nlp = None
    for model_name in ("en_core_sci_sm", "en_core_sci_md", "en_core_sci_lg"):
        try:
            nlp = spacy.load(model_name)
            break
        except OSError:
            continue
    if nlp is None:
        return None

    # Add span-as-entity: treat whole doc as one entity for linking
    def span_as_entity(doc):
        if len(doc) > 0:
            doc.ents = (Span(doc, 0, len(doc)),)
        return doc

    nlp.add_pipe(span_as_entity, name="span_entity", first=True)

    try:
        nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": False})
    except Exception:
        return None

    _scispacy_nlp = nlp  # already declared global at top
    return nlp


def _normalize_with_scispacy(span_text: str) -> Dict[str, Optional[str]]:
    """Use scispaCy EntityLinker to map span to UMLS. Returns {term, cui}."""
    global _scispacy_nlp
    if _scispacy_nlp is None:
        _scispacy_nlp = _get_scispacy_nlp()
    if _scispacy_nlp is None:
        return {"term": span_text, "cui": None}

    try:
        doc = _scispacy_nlp(span_text.strip())
        if not doc.ents:
            return {"term": span_text, "cui": None}
        span = doc.ents[0]
        kb_ents = getattr(span._, "kb_ents", None) or []
        if not kb_ents:
            return {"term": span_text, "cui": None}
        # kb_ents: (cui, score); take best
        best_cui, _ = kb_ents[0]
        linker = _scispacy_nlp.get_pipe("scispacy_linker")
        entity = linker.kb.cui_to_entity.get(best_cui)
        term = entity.canonical_name if entity else span_text
        return {"term": term, "cui": best_cui}
    except Exception:
        return {"term": span_text, "cui": None}


class UMLSNormalizer:
    """
    Map raw entity spans to UMLS concepts (term + CUI).
    Priority: QuickUMLS > scispaCy > dictionary fallback (if enabled).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        threshold: float = 0.7,
        use_scispacy: bool = True,
        use_fallback: bool = False,
    ) -> None:
        self.path = path or os.environ.get("QUICKUMLS_PATH", "")
        self.matcher = _get_quickumls_matcher(self.path, threshold)
        self._quickumls_available = self.matcher is not None
        self.use_scispacy = use_scispacy
        self.use_fallback = use_fallback
        self._scispacy_available: Optional[bool] = None  # lazy-check

        if self._quickumls_available:
            print("[UMLSNormalizer] Using QuickUMLS at", self.path)
        elif self.path:
            print("[UMLSNormalizer] QuickUMLS path set but matcher failed. See UMLS_SETUP.md.")
        elif use_scispacy:
            # Lazy init scispacy on first normalize()
            print("[UMLSNormalizer] QUICKUMLS_PATH not set – will try scispaCy (pip install scispacy + en_core_sci_sm).")
        else:
            print("[UMLSNormalizer] No UMLS backend. Set QUICKUMLS_PATH or use_scispacy=True. See UMLS_SETUP.md.")

    def _check_scispacy(self) -> bool:
        if self._scispacy_available is not None:
            return self._scispacy_available
        result = _get_scispacy_nlp()
        self._scispacy_available = result is not None
        if self._scispacy_available:
            print("[UMLSNormalizer] Using scispaCy UMLS linker (auto-downloaded KB).")
        return self._scispacy_available

    @property
    def available(self) -> bool:
        """True if any backend can normalize."""
        return self._quickumls_available or self.use_fallback or (
            self.use_scispacy and (self._scispacy_available is True or self._check_scispacy())
        )

    def normalize(self, span_text: str) -> Dict[str, Optional[str]]:
        """Return {term, cui}. Uses QuickUMLS, then scispaCy, then fallback dict."""
        if not span_text.strip():
            return {"term": span_text, "cui": None}
        if self.matcher is not None:
            try:
                matches = self.matcher.match(span_text, best_match=True)
                if matches and len(matches) > 0 and len(matches[0]) > 0:
                    best = matches[0][0]
                    return {"term": best.get("term", span_text), "cui": best.get("cui")}
            except Exception:
                pass
        if self.use_scispacy and self._check_scispacy():
            return _normalize_with_scispacy(span_text)
        if self.use_fallback:
            return _normalize_with_fallback(span_text)
        return {"term": span_text, "cui": None}


# ────────────────────────────────────────────────────────────────────
# medspaCy Negation Detection
# ────────────────────────────────────────────────────────────────────
def _get_medspacy_nlp():
    """Lazy-load medspaCy. Returns None if not installed."""
    try:
        import medspacy
        return medspacy.load()
    except Exception:
        return None


_medspacy_nlp = None


def detect_negation_medspacy(dialogue_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use medspaCy ConText to detect negation on entity spans.
    If an entity's char range overlaps a negated scope, set negated=True.
    Modifies entities in-place. Returns the same list.
    """
    global _medspacy_nlp
    if _medspacy_nlp is None:
        _medspacy_nlp = _get_medspacy_nlp()
    if _medspacy_nlp is None:
        return entities

    try:
        doc = _medspacy_nlp(dialogue_text)
        # ConText adds scope to targets; check doc.ents for is_negated
        for ent in entities:
            start, end = ent.get("start", 0), ent.get("end", 0)
            for doc_ent in doc.ents:
                if not (doc_ent.end_char <= start or doc_ent.start_char >= end):
                    # Overlap
                    if hasattr(doc_ent, "_") and hasattr(doc_ent._, "is_negated") and doc_ent._.is_negated:
                        ent["negated"] = True
                        break
        # Also check custom char spans if ConText stores modifier scopes
        if hasattr(doc, "_") and hasattr(doc._, "context_graph"):
            cg = doc._.context_graph
            for mod in getattr(cg, "modifiers", []) or []:
                if getattr(mod, "category", "").lower() == "negation":
                    scope = getattr(mod, "scope_span", None) or getattr(mod, "span", None)
                    if scope:
                        ms, me = scope.start_char, scope.end_char
                        for ent in entities:
                            start, end = ent.get("start", 0), ent.get("end", 0)
                            if not (me <= start or ms >= end):
                                ent["negated"] = True
    except Exception:
        pass
    return entities
