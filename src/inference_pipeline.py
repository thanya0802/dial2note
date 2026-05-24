"""
Combined Entity-conditioned + RAFT inference (matches ``combined_prompt`` training).

Also exposes the standard ``SOAPInferencePipeline`` for imports.

CLI (example)::

    python -m src.inference_pipeline --combined_mode \\
        --retrieval_cache data/processed/shared_task_eval_raft.csv \\
        --annotated_csv outputs/dev_annotated_v3_mistral.csv \\
        --dev_csv data/processed/shared_task_eval.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.combined_prompt import build_combined_mistral_prompt
from src.inference import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    SOAPInferencePipeline,
    _is_empty,
)
from src.validation import MAX_CORRECTIONS, validate_note

__all__ = [
    "SOAPInferencePipeline",
    "CombinedSOAPInferencePipeline",
    "model_dir_from_lora_path",
    "load_retrieval_cache",
    "load_entities_by_id",
]


def model_dir_from_lora_path(lora_path: str) -> str:
    """If path ends with ``best``, use parent as Unsloth adapter root."""
    p = Path(lora_path)
    if p.name == "best" and p.is_dir():
        return str(p.parent)
    return str(p)


def load_retrieval_cache(path: Path | str) -> Dict[int, str]:
    """CSV with columns ``id``, ``retrieved_note`` (no truncation applied)."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] retrieval cache not found: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    if "id" not in df.columns or "retrieved_note" not in df.columns:
        sys.exit(f"[ERROR] {path} must have columns: id, retrieved_note")
    out: Dict[int, str] = {}
    for _, r in df.iterrows():
        out[int(r["id"])] = str(r["retrieved_note"])
    return out


def load_entities_by_id(path: Path | str) -> Dict[int, str]:
    """CSV with ``id`` and ``entities_json``."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] annotated CSV not found: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    if "id" not in df.columns or "entities_json" not in df.columns:
        sys.exit(f"[ERROR] {path} must have columns: id, entities_json")
    out: Dict[int, str] = {}
    for _, r in df.iterrows():
        out[int(r["id"])] = r["entities_json"] if not _is_empty(r.get("entities_json")) else "{}"
    return out


class CombinedSOAPInferencePipeline(SOAPInferencePipeline):
    """
    Inference with the same [INST] layout as ``build_combined_training_text``.
    Resolves ``retrieved_note`` and ``entities_json`` from row fields or preloaded CSV maps.
    """

    def __init__(
        self,
        model_dir: str = "models/mistral_finetune_combined",
        retrieval_cache_path: Optional[str] = None,
        annotated_csv_path: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            retriever=None,
            ner=None,
            demographics_extractor=None,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self._retrieval_by_id: Dict[int, str] = {}
        self._entities_by_id: Dict[int, str] = {}
        if retrieval_cache_path:
            self._retrieval_by_id = load_retrieval_cache(retrieval_cache_path)
        if annotated_csv_path:
            self._entities_by_id = load_entities_by_id(annotated_csv_path)

    def _resolve_entities(
        self,
        sample_id: Optional[int],
        entities_json: Optional[Any],
    ) -> Any:
        if entities_json is not None and not _is_empty(entities_json):
            return entities_json
        if sample_id is not None and sample_id in self._entities_by_id:
            return self._entities_by_id[sample_id]
        return "{}"

    def _resolve_retrieved_note(
        self,
        sample_id: Optional[int],
        retrieved_note: Optional[str],
    ) -> str:
        if retrieved_note is not None and str(retrieved_note).strip():
            return str(retrieved_note)
        if sample_id is not None and sample_id in self._retrieval_by_id:
            return self._retrieval_by_id[sample_id]
        return ""

    def generate_one(
        self,
        dialogue_raw: str,
        entities_json: Optional[str] = None,
        demographics_json: Optional[str] = None,
        dialogue_tagged: Optional[str] = None,
        k: int = 1,
        max_input_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        validate: bool = False,
        *,
        sample_id: Optional[int] = None,
        retrieved_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate one note using the combined prompt (training-aligned)."""
        max_in = max_input_length or self.max_input_length
        max_out = max_new_tokens or self.max_new_tokens

        ej = self._resolve_entities(sample_id, entities_json)
        rn = self._resolve_retrieved_note(sample_id, retrieved_note)
        if not rn.strip():
            raise ValueError(
                "combined inference requires a non-empty retrieved_note "
                "(pass retrieved_note= or preload --retrieval_cache with this id).",
            )

        try:
            if isinstance(ej, str):
                entities_dict = json.loads(ej) if ej.strip() else {}
            else:
                entities_dict = ej or {}
        except json.JSONDecodeError:
            entities_dict = {}
        demographics = {}
        if demographics_json is not None and not _is_empty(demographics_json):
            demographics = json.loads(demographics_json) if isinstance(demographics_json, str) else demographics_json

        prompt = build_combined_mistral_prompt(dialogue_raw, rn, ej)
        note_pred = self._generate_section_aware_combined(
            dialogue_raw=dialogue_raw,
            retrieved_note=rn,
            entities_json=ej,
            max_input_length=max_in,
            max_new_tokens=max_out,
        )

        validation_passed = False
        correction_attempts = 0
        if validate:
            for attempt in range(MAX_CORRECTIONS + 1):
                report = validate_note(
                    note_pred, entities_dict, demographics, dialogue_raw,
                )
                if report["all_valid"]:
                    validation_passed = True
                    break
                if attempt < MAX_CORRECTIONS:
                    correction_attempts += 1
                    hint = "\n\n[Some sections were incomplete. Generate complete sections.]"
                    note_pred = self._generate_section_aware_combined(
                        dialogue_raw=dialogue_raw,
                        retrieved_note=rn,
                        entities_json=ej,
                        max_input_length=max_in,
                        max_new_tokens=max_out,
                        extra_instruction=hint,
                    )
        else:
            validation_passed = True

        return {
            "prompt": prompt,
            "retrieved_ids": [],
            "note_pred": note_pred,
            "validation_passed": validation_passed,
            "correction_attempts": correction_attempts,
        }

    def _generate_section_aware_combined(
        self,
        dialogue_raw: str,
        retrieved_note: str,
        entities_json: Any,
        max_input_length: int,
        max_new_tokens: int,
        extra_instruction: Optional[str] = None,
    ) -> str:
        prev: Dict[str, str] = {}
        section_max = max(500, max_new_tokens // 4)

        for section in ("Subjective", "Objective", "Assessment", "Plan"):
            prompt = build_combined_mistral_prompt(
                dialogue_raw,
                retrieved_note,
                entities_json,
                section_name=section,
                previous_sections=prev,
                extra_instruction=extra_instruction,
            )
            section_text = self._generate_text(
                prompt, max_input_length, section_max,
            )
            prev[section] = section_text.strip()

        return "\n\n".join(
            f"**{i+1}. {name}:**\n{text}"
            for i, (name, text) in enumerate(prev.items())
        )

    def generate_batch(
        self,
        rows: List[Dict[str, Any]],
        k: int = 1,
        max_input_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        gen_batch_size: int = 2,
        validate: bool = False,
        use_rag: bool = True,
        use_tagged: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch generation with combined prompts (``use_rag`` / ``use_tagged`` ignored)."""
        from src.combined_prompt import build_combined_mistral_prompt
        max_in = max_input_length or self.max_input_length
        max_out = max_new_tokens or self.max_new_tokens
        n = len(rows)
        prompts: List[str] = []
        meta: List[Dict[str, Any]] = []

        t0 = time.time()
        for idx, row in enumerate(rows):
            dialogue_raw = row["dialogue"]
            sid = int(row["id"]) if row.get("id") is not None else None
            ej = self._resolve_entities(sid, row.get("entities_json"))
            rn = self._resolve_retrieved_note(sid, row.get("retrieved_note"))
            if not rn.strip():
                raise ValueError(
                    f"Row id={sid}: missing retrieved_note for combined inference.",
                )
            prompt = build_combined_mistral_prompt(dialogue_raw, rn, ej)
            prompts.append(prompt)
            meta.append({
                "id": sid,
                "dialogue_raw": dialogue_raw,
                "entities_json": ej,
                "retrieved_note": rn,
            })
            if (idx + 1) % 25 == 0 or idx == n - 1:
                print(f"  Prompts built: {idx + 1}/{n}  ({time.time() - t0:.1f}s)")

        print(f"\n  Generating {n} notes (combined, single-pass batch) …")
        # Build full prompts for single-pass generation
        full_prompts = []
        for m in meta:
            prompt = build_combined_mistral_prompt(
                dialogue=m["dialogue_raw"],
                retrieved_note=m["retrieved_note"],
                entities_json=m["entities_json"],
            )
            full_prompts.append(prompt)
        # Generate all notes single-pass
        all_preds = []
        for i, prompt in enumerate(full_prompts):
            text = self._generate_text(prompt, max_in, max_out)
            all_preds.append(text)
            if (i + 1) % 25 == 0 or i == n - 1:
                print(f"    Generated {i + 1}/{n}")

        val_passed = [False] * n
        corr_attempts = [0] * n

        if validate:
            for i in range(n):
                note = all_preds[i]
                m = meta[i]
                ej = m["entities_json"]
                try:
                    if isinstance(ej, str):
                        entities_dict = json.loads(ej) if ej.strip() else {}
                    else:
                        entities_dict = ej or {}
                except json.JSONDecodeError:
                    entities_dict = {}
                for attempt in range(MAX_CORRECTIONS + 1):
                    report = validate_note(
                        note, entities_dict, {}, m["dialogue_raw"],
                    )
                    if report["all_valid"]:
                        val_passed[i] = True
                        break
                    if attempt < MAX_CORRECTIONS:
                        corr_attempts[i] += 1
                        hint = "\n\n[Some sections were incomplete. Generate complete sections.]"
                        note = self._generate_section_aware_combined(
                            dialogue_raw=m["dialogue_raw"],
                            retrieved_note=m["retrieved_note"],
                            entities_json=m["entities_json"],
                            max_input_length=max_in,
                            max_new_tokens=max_out,
                            extra_instruction=hint,
                        )
                all_preds[i] = note
        else:
            val_passed = [True] * n

        output: List[Dict[str, Any]] = []
        for i in range(n):
            output.append({
                "id": meta[i]["id"],
                "note_pred": all_preds[i],
                "prompt": prompts[i],
                "retrieved_ids": [],
                "validation_passed": val_passed[i],
                "correction_attempts": corr_attempts[i],
            })
        return output


def _main_combined() -> None:
    ap = argparse.ArgumentParser(description="Combined SOAP inference (entity + RAFT prompt).")
    ap.add_argument("--combined_mode", action="store_true", help="Use combined prompt + model.")
    ap.add_argument("--annotated_csv", default="outputs/dev_annotated_v3_mistral.csv")
    ap.add_argument("--retrieval_cache", required=True, help="CSV: id, retrieved_note")
    ap.add_argument("--lora_path", default="models/mistral_finetune_combined/best")
    ap.add_argument("--dev_csv", default="data/processed/shared_task_eval.csv")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if not args.combined_mode:
        print("Pass --combined_mode to run combined inference.")
        sys.exit(0)

    dev_path = Path(args.dev_csv)
    if not dev_path.exists():
        sys.exit(f"[ERROR] {dev_path} not found.")

    dev_df = pd.read_csv(dev_path, engine="python", on_bad_lines="warn")
    if args.max_examples:
        dev_df = dev_df.head(args.max_examples)

    model_dir = model_dir_from_lora_path(args.lora_path)
    pipe = CombinedSOAPInferencePipeline(
        model_dir=model_dir,
        retrieval_cache_path=args.retrieval_cache,
        annotated_csv_path=args.annotated_csv,
    )

    rows = []
    for _, row in dev_df.iterrows():
        rows.append({
            "id": int(row["id"]),
            "dialogue": row["dialogue"],
            "entities_json": row.get("entities_json"),
            "retrieved_note": row.get("retrieved_note"),
        })

    t0 = time.time()
    results = pipe.generate_batch(rows, validate=args.validate)
    # Save results to CSV
    out_df = pd.DataFrame([{"id": r["id"], "generated_note": r["note_pred"]} for r in results])
    out_path = "outputs/submission_combined.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} notes to {out_path}")
    print(f"Done in {time.time() - t0:.1f}s  ({len(results)} examples)")


if __name__ == "__main__":
    _main_combined()
