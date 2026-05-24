"""
NER Pipeline v3 — Mistral-based entity extraction using vLLM.
Replaces GLiNER with base Mistral-7B-Instruct for accurate SOAP entity extraction.

Usage:
    python -m ner_pipeline_v3.mistral_extract --split both
    python -m ner_pipeline_v3.mistral_extract --split train
    python -m ner_pipeline_v3.mistral_extract --split eval
"""

import argparse
import json
import os
import time
from collections import Counter

import pandas as pd
from vllm import LLM, SamplingParams

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_CSV = "data/processed/shared_task_train.csv"
EVAL_CSV = "data/processed/shared_task_eval.csv"
TRAIN_OUTPUT = "ner_pipeline_v3/outputs/train_annotated_v3_mistral.csv"
EVAL_OUTPUT = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
CHECKPOINT_EVERY = 50
BATCH_SIZE = 32  # vLLM processes this many at once

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical NER system. Extract medical entities from the doctor-patient dialogue below.

Return ONLY a JSON array. No explanation, no markdown, no backticks. Just the raw JSON array.

Entity types (use EXACTLY these labels):
- "chief complaint": the primary reason for the visit (extract only 1-2)
- "symptom": physical symptoms reported by the patient (pain, fatigue, nausea, etc.)
- "diagnosis": medical diagnosis or disease named by the doctor
- "medication": drug name or supplement prescribed
- "drug detail": dosage, frequency, route (e.g., "500mg twice daily", "2 tablets oral")
- "test or lab order": diagnostic test or imaging ordered (CBC, MRI, ECG, etc.)
- "vital sign": measured value with number (BP 120/80, BMI 24.5, HR 72)
- "physical exam finding": doctor's observation from exam (tenderness, swelling, rash)
- "referral": specialist referral (cardiologist, Dr. Smith, dietitian)
- "follow-up instruction": follow-up plan, lifestyle advice, activity recommendation

Rules:
- Keep entity text SHORT (1-5 words max)
- Do NOT extract generic words like "doctor", "patient", "medicine"
- Do NOT extract body parts alone (extract "chest pain" not "chest")
- Extract 10-25 entities per dialogue
- Each entity needs: "text", "label", "score" (your confidence 0.0-1.0)

Example output:
[{"text": "chest pain", "label": "chief complaint", "score": 0.95}, {"text": "shortness of breath", "label": "symptom", "score": 0.9}]"""


def build_prompt(dialogue: str) -> str:
    """Build the Mistral instruct prompt."""
    return f"[INST] {SYSTEM_PROMPT}\n\nDialogue:\n{dialogue[:6000]} [/INST]"


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_entities(response: str) -> list[dict]:
    """Parse LLM response into entity list. Handles common formatting issues."""
    text = response.strip()

    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Try direct parse
    try:
        entities = json.loads(text)
        if isinstance(entities, list):
            return _validate_entities(entities)
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in response
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            entities = json.loads(text[start:end + 1])
            if isinstance(entities, list):
                return _validate_entities(entities)
        except json.JSONDecodeError:
            pass

    return []


def _validate_entities(entities: list) -> list[dict]:
    """Validate and clean entity list."""
    VALID_LABELS = {
        "chief complaint", "symptom", "diagnosis", "medication",
        "drug detail", "test or lab order", "vital sign",
        "physical exam finding", "referral", "follow-up instruction",
    }

    cleaned = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        if "text" not in e or "label" not in e:
            continue

        text = str(e["text"]).strip()
        label = str(e["label"]).strip().lower()
        score = float(e.get("score", 0.8))

        # Skip empty or too long
        if not text or len(text) <= 1:
            continue
        if len(text.split()) > 7:
            continue

        # Map close label matches
        if label == "follow-up" or label == "follow up":
            label = "follow-up instruction"
        if label == "test" or label == "lab order" or label == "lab":
            label = "test or lab order"
        if label == "exam finding" or label == "physical exam":
            label = "physical exam finding"
        if label == "dosage" or label == "drug dosage":
            label = "drug detail"

        if label not in VALID_LABELS:
            continue

        # Clamp score
        score = max(0.0, min(1.0, score))

        cleaned.append({"text": text, "label": label, "score": round(score, 4)})

    return cleaned


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(entities: list[dict]) -> list[dict]:
    """Exact + substring dedup."""
    seen: dict[str, dict] = {}
    for ent in entities:
        key = ent["text"].lower().strip()
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent
    unique = list(seen.values())

    unique.sort(key=lambda e: len(e["text"]), reverse=True)
    keep = []
    for candidate in unique:
        cand_text = candidate["text"].lower()
        cand_label = candidate["label"]
        dominated = False
        for kept in keep:
            if kept["label"] == cand_label and cand_text in kept["text"].lower():
                dominated = True
                break
        if not dominated:
            keep.append(candidate)

    return keep


# ─── Processing ───────────────────────────────────────────────────────────────

def _partial_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_partial{ext}"


def process_split(llm, sampling_params, csv_path, output_path, split_name):
    """Process a full split with batched vLLM inference and checkpointing."""
    print(f"\n{'=' * 60}")
    print(f"  Processing: {split_name.upper()}")
    print(f"  Source: {csv_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    partial = _partial_path(output_path)

    df = pd.read_csv(csv_path)
    total = len(df)

    # Resume from checkpoint
    start_idx = 0
    done_rows = []
    if os.path.exists(partial):
        df_done = pd.read_csv(partial)
        start_idx = len(df_done)
        done_rows = df_done.to_dict("records")
        print(f"  Resuming from checkpoint: {start_idx}/{total}")

    if start_idx >= total:
        pd.DataFrame(done_rows).to_csv(output_path, index=False)
        if os.path.exists(partial):
            os.remove(partial)
        print("  Already complete.")
        return

    new_rows = []
    t_start = time.time()

    # Process in batches
    for batch_start in range(start_idx, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_rows = []
        prompts = []

        for idx in range(batch_start, batch_end):
            row = df.iloc[idx].to_dict()
            dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""

            if not dialogue.strip():
                row["entities_json"] = json.dumps([])
                new_rows.append(row)
                batch_rows.append(None)
                continue

            prompts.append(build_prompt(dialogue))
            batch_rows.append(row)

        # vLLM batch inference
        if prompts:
            outputs = llm.generate(prompts, sampling_params)

            prompt_idx = 0
            for i, row in enumerate(batch_rows):
                if row is None:
                    continue
                if batch_rows[i] is not None and prompt_idx < len(outputs):
                    response = outputs[prompt_idx].outputs[0].text
                    entities = parse_entities(response)
                    entities = deduplicate(entities)
                    row["entities_json"] = json.dumps(entities)
                    new_rows.append(row)
                    prompt_idx += 1

        # Progress
        done = batch_end - start_idx
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - batch_end
        eta = remaining / rate if rate > 0 else 0
        eta_str = f"{eta / 60:.1f}m" if eta < 3600 else f"{eta / 3600:.1f}h"

        # Count entities in this batch
        batch_ent_counts = []
        for row in new_rows[-len(prompts):] if prompts else []:
            try:
                ents = json.loads(row.get("entities_json", "[]"))
                batch_ent_counts.append(len(ents))
            except:
                batch_ent_counts.append(0)

        avg_ents = sum(batch_ent_counts) / len(batch_ent_counts) if batch_ent_counts else 0

        print(
            f"  [{batch_end}/{total}] "
            f"Batch {len(prompts)} rows | "
            f"Avg {avg_ents:.1f} entities/row | "
            f"{elapsed:.0f}s elapsed | "
            f"ETA: {eta_str}"
        )

        # Checkpoint
        if (batch_end - start_idx) % (CHECKPOINT_EVERY * BATCH_SIZE // BATCH_SIZE) == 0:
            df_partial = pd.DataFrame(done_rows + new_rows)
            df_partial.to_csv(partial, index=False)
            print(f"  [checkpoint] Saved {len(df_partial)} rows")

    # Final save
    all_rows = done_rows + new_rows
    df_final = pd.DataFrame(all_rows)
    df_final.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df_final)} rows → {output_path}")

    if os.path.exists(partial):
        os.remove(partial)

    # Summary
    all_entities = []
    for row in all_rows:
        try:
            ents = json.loads(row.get("entities_json", "[]"))
            all_entities.extend(ents)
        except:
            pass

    label_dist = Counter(e["label"] for e in all_entities)
    print(f"\n  Summary — {split_name}")
    print(f"  {'─' * 50}")
    print(f"  Total entities: {len(all_entities)}")
    print(f"  Avg per row: {len(all_entities) / len(all_rows):.1f}")
    print(f"  Label distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {count:>5}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mistral-based NER extraction")
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    print("\n  Loading Mistral-7B-Instruct-v0.3 with vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="half",
    )
    print("  Model loaded.\n")

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        top_p=0.95,
    )

    splits = {
        "train": (TRAIN_CSV, TRAIN_OUTPUT),
        "eval": (EVAL_CSV, EVAL_OUTPUT),
    }

    to_run = ["train", "eval"] if args.split == "both" else [args.split]

    for split_name in to_run:
        csv_path, output_path = splits[split_name]
        process_split(llm, sampling_params, csv_path, output_path, split_name)

    print("\n  All done.")


if __name__ == "__main__":
    main()
