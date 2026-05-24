"""
Few-shot ablation: Example 1 has NER + dialogue, Example 2 has dialogue only.
Uses v1 fine-tuned model (models/mistral_finetune/best/) with vLLM.
Examples: training IDs 1 and 3747. Output: outputs/submission_fewshot_mixed.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Paths
TRAIN_CSV = Path("data/processed/shared_task_train.csv")
TRAIN_ANNOTATED = Path("outputs/train_annotated_v2.csv")
DEV_ANNOTATED = Path("outputs/dev_annotated_v2.csv")
EVAL_CSV = Path("data/processed/shared_task_eval.csv")
LORA_DIR = Path("models/mistral_finetune/best")
OUTPUT_CSV = Path("outputs/submission_fewshot_mixed.csv")

EXAMPLE_IDS = [1, 3747]

MAX_TOTAL = 8000
MAX_PROMPT = MAX_TOTAL - 1500


def format_entities_for_prompt(entities_list):
    groups = {}
    for ent in entities_list:
        groups.setdefault(ent["label"], []).append(ent["text"])
    sections = []
    if "chief complaint" in groups:
        sections.append(f'Chief Complaint: {", ".join(groups["chief complaint"])}')
    if "symptom" in groups:
        sections.append(f'Symptoms: {", ".join(groups["symptom"])}')
    if "diagnosis" in groups:
        sections.append(f'Diagnoses: {", ".join(groups["diagnosis"])}')
    med_parts = []
    if "medication" in groups:
        med_parts.extend(groups["medication"])
    for d in ["drug dosage", "drug frequency", "drug duration", "drug route"]:
        if d in groups:
            med_parts.extend([f'({d.replace("drug ", "")}:{t})' for t in groups[d]])
    if med_parts:
        sections.append(f"Medications: {', '.join(med_parts)}")
    if "test or lab order" in groups:
        sections.append(f'Tests/Labs: {", ".join(groups["test or lab order"])}')
    if "procedure" in groups:
        sections.append(f'Procedures: {", ".join(groups["procedure"])}')
    if "vital sign" in groups:
        sections.append(f'Vital Signs: {", ".join(groups["vital sign"])}')
    for h in ["past medical history", "family history", "social history", "allergy"]:
        if h in groups:
            sections.append(f'{h.title()}: {", ".join(groups[h])}')
    if "treatment plan" in groups:
        sections.append(f'Treatment Plan: {", ".join(groups["treatment plan"])}')
    if "follow-up instruction" in groups:
        sections.append(f'Follow-up: {", ".join(groups["follow-up instruction"])}')
    for c in ["body part or location", "severity", "temporal expression"]:
        if c in groups:
            sections.append(f'{c.title()}: {", ".join(groups[c])}')
    if not sections:
        return "No clinical entities extracted."
    return "\n".join(f"- {s}" for s in sections)


def load_examples():
    """Load examples (IDs 1, 3747) from train CSV and train_annotated_v2."""
    train_df = pd.read_csv(TRAIN_CSV, engine="python", on_bad_lines="warn")
    train_ner = pd.read_csv(TRAIN_ANNOTATED, engine="python", on_bad_lines="warn")
    train_ner_by_id = train_ner.set_index("id").to_dict("index")

    examples = []
    for eid in EXAMPLE_IDS:
        row = train_df[train_df["id"] == eid].iloc[0]
        ner_row = train_ner_by_id.get(eid, {})
        ej = ner_row.get("entities_json")
        entities = json.loads(ej) if isinstance(ej, str) and ej else []
        examples.append({
            "id": eid,
            "dialogue": str(row["dialogue"]),
            "note": str(row["note"]),
            "entities": entities,
        })
    return examples


def load_eval_data():
    """Load eval dialogues and NER from dev_annotated_v2."""
    dev_df = pd.read_csv(DEV_ANNOTATED, engine="python", on_bad_lines="warn")
    return dev_df


def build_prompt_mixed(ex1, ex2, eval_row):
    """Example 1 has NER + dialogue, Example 2 has dialogue only."""
    return f"""[INST] Generate a complete clinical SOAP note from the following doctor-patient dialogue. Use the extracted clinical entities to ensure all discussed symptoms, medications, diagnoses, and plans are included. Also include appropriate patient demographics and provider referral details as contextually relevant.

Here are examples:

=== Example 1 ===
Extracted Clinical Entities:
{format_entities_for_prompt(ex1["entities"])}

Dialogue:
{ex1["dialogue"]}

SOAP Note:
{ex1["note"]}

=== Example 2 ===
Dialogue:
{ex2["dialogue"]}

SOAP Note:
{ex2["note"]}

=== Your Case ===
Extracted Clinical Entities:
{format_entities_for_prompt(eval_row["entities"])}

Dialogue:
{eval_row["dialogue"]}

[/INST]
"""


def maybe_truncate_and_build(ex1, ex2, row, tok):
    """Build prompt, truncate eval dialogue if over MAX_PROMPT tokens."""
    eval_row = {"dialogue": str(row["dialogue"]), "entities": row["entities"]}
    prompt = build_prompt_mixed(ex1, ex2, eval_row)
    prompt_toks = len(tok.encode(prompt))
    if prompt_toks <= MAX_PROMPT:
        return prompt
    over = prompt_toks - MAX_PROMPT
    dial_tokens = tok.encode(str(row["dialogue"]))
    n_keep = max(0, len(dial_tokens) - over - 50)
    truncated = tok.decode(dial_tokens[:n_keep])
    eval_row["dialogue"] = truncated
    return build_prompt_mixed(ex1, ex2, eval_row)


def main():
    print("=" * 60)
    print("Few-shot Mixed – Example 1 NER+dialogue, Example 2 dialogue only")
    print("=" * 60)

    for p in [TRAIN_CSV, TRAIN_ANNOTATED, DEV_ANNOTATED, LORA_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    examples = load_examples()
    ex1, ex2 = examples[0], examples[1]
    print(f"Examples: IDs {ex1['id']}, {ex2['id']}")

    dev_df = load_eval_data()
    dev_df["entities"] = dev_df["entities_json"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x else []
    )

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    prompts = []
    meta = []
    for _, row in dev_df.iterrows():
        prompt = maybe_truncate_and_build(ex1, ex2, row, tok)
        prompts.append(prompt)
        meta.append({"id": row["id"]})

    print(f"Built {len(prompts)} prompts")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_path = str(LORA_DIR.resolve())
    lora_req = LoRARequest(lora_name="soap", lora_int_id=1, lora_path=lora_path)
    sampling = SamplingParams(temperature=0.3, top_p=1.0, max_tokens=1500)

    print(f"Loading vLLM + LoRA from {lora_path} …")
    t0 = time.time()
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_model_len=8192,
        disable_log_stats=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Generating …")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    elapsed = time.time() - t0
    notes = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
    print(f"  Generated {len(notes)} notes in {elapsed:.1f}s ({elapsed / len(notes):.3f}s each)")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame([{"id": m["id"], "generated_note": n} for m, n in zip(meta, notes)])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
