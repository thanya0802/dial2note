"""
Entity-Conditioned Fine-tuning for Dial2Note.
Trains Mistral-7B-Instruct-v0.3 with QLoRA using NER entities + dialogue → SOAP note.

The key difference from baseline: the training prompt includes extracted clinical entities
grouped by SOAP section, so the model learns to use entity signals for generation.

Usage:
    python -m ner_pipeline_v3.finetune_entity_conditioned

Requires:
    - ner_pipeline_v3/outputs/train_annotated_v3_mistral.csv (NER-tagged training data)
    - unsloth, trl, torch
"""

import json
import os
import torch
import pandas as pd
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"
OUTPUT_DIR = "models/mistral_entity_conditioned"

# Data
TRAIN_CSV = "ner_pipeline_v3/outputs/train_annotated_v3_mistral.csv"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"

# QLoRA hyperparameters (matching your proven setup)
MAX_SEQ_LENGTH = 8192
LORA_R = 16
LORA_ALPHA = 16
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LR_SCHEDULER = "linear"
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
EVAL_STEPS = 500


# ─── Entity Formatting ────────────────────────────────────────────────────────

# Map entity labels to SOAP sections
SOAP_MAP = {
    "Subjective": ["chief complaint", "symptom"],
    "Objective": ["vital sign", "physical exam finding"],
    "Assessment": ["diagnosis"],
    "Plan": [
        "medication", "drug detail", "test or lab order",
        "referral", "follow-up instruction",
    ],
}


def format_entities_for_prompt(entities_json: str) -> str:
    """
    Group entities by SOAP section and format as a structured string.
    Only includes sections that have at least one entity.
    """
    try:
        entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
    except (json.JSONDecodeError, TypeError):
        entities = []

    if not entities:
        return "No entities extracted."

    # Group by SOAP section
    grouped = {}
    for section, labels in SOAP_MAP.items():
        section_entities = [
            e["text"] for e in entities
            if e.get("label", "").lower() in labels
        ]
        if section_entities:
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for t in section_entities:
                t_lower = t.lower()
                if t_lower not in seen:
                    seen.add(t_lower)
                    unique.append(t)
            grouped[section] = unique

    if not grouped:
        return "No entities extracted."

    # Format as structured text
    parts = []
    for section, ents in grouped.items():
        parts.append(f"{section}: {', '.join(ents)}")

    return "\n".join(parts)


def build_training_prompt(dialogue: str, entities_json: str, note: str) -> str:
    """
    Build the full training example in Mistral-Instruct format.

    Format:
    [INST] instruction + entities + dialogue [/INST] note

    This MUST match the inference format exactly.
    """
    entity_text = format_entities_for_prompt(entities_json)

    prompt = (
        f"[INST] Convert the following clinical dialogue into a SOAP note. "
        f"Use the extracted clinical entities to ensure completeness. "
        f"Include sections: Subjective, Objective, Assessment, Plan.\n\n"
        f"Extracted Entities:\n{entity_text}\n\n"
        f"Dialogue:\n{dialogue}\n\n[/INST]\n{note}</s>"
    )

    return prompt


def build_inference_prompt(dialogue: str, entities_json: str) -> str:
    """
    Build the inference prompt (same format as training, without the note).
    Exported for use by the generation script.
    """
    entity_text = format_entities_for_prompt(entities_json)

    prompt = (
        f"[INST] Convert the following clinical dialogue into a SOAP note. "
        f"Use the extracted clinical entities to ensure completeness. "
        f"Include sections: Subjective, Objective, Assessment, Plan.\n\n"
        f"Extracted Entities:\n{entity_text}\n\n"
        f"Dialogue:\n{dialogue}\n\n[/INST]\n"
    )

    return prompt


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_datasets():
    """Load NER-annotated train and eval CSVs."""
    print(f"  Loading train: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"  Train rows: {len(df_train)}")

    print(f"  Loading eval: {EVAL_CSV}")
    df_eval = pd.read_csv(EVAL_CSV)
    print(f"  Eval rows: {len(df_eval)}")

    # Format training examples
    train_texts = []
    skipped = 0
    for _, row in df_train.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        note = str(row.get("note", "")) if pd.notna(row.get("note")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"

        if not dialogue.strip() or not note.strip():
            skipped += 1
            continue

        text = build_training_prompt(dialogue, entities_json, note)
        train_texts.append({"text": text})

    eval_texts = []
    for _, row in df_eval.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        note = str(row.get("note", "")) if pd.notna(row.get("note")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"

        if not dialogue.strip() or not note.strip():
            continue

        text = build_training_prompt(dialogue, entities_json, note)
        eval_texts.append({"text": text})

    print(f"  Train examples: {len(train_texts)} (skipped {skipped})")
    print(f"  Eval examples: {len(eval_texts)}")

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_texts)
    eval_dataset = Dataset.from_list(eval_texts)

    return train_dataset, eval_dataset


# ─── Fine-tuning ──────────────────────────────────────────────────────────────

def finetune():
    """Run entity-conditioned QLoRA fine-tuning with Unsloth."""
    print("\n" + "=" * 60)
    print("  Entity-Conditioned Fine-tuning")
    print("  Model: Mistral-7B-Instruct-v0.3 + QLoRA")
    print(f"  Train data: {TRAIN_CSV}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Load model with Unsloth ──
    print("\n  Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    print("  Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── Load data ──
    train_dataset, eval_dataset = load_datasets()

    # ── Training config ──
    print("\n  Configuring training...")
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # ── Train ──
    print("\n  Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch: {PER_DEVICE_BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR: {LEARNING_RATE}, Scheduler: {LR_SCHEDULER}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")

    trainer.train()

    # ── Save ──
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n  Model saved → {best_dir}")

    print("\n  Fine-tuning complete!")


if __name__ == "__main__":
    finetune()
