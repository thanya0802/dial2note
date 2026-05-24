"""
Phase 4 – Fine-Tuning SOAP-Note Generator
=========================================

Supports:
  - Mistral-7B-Instruct-v0.3 + Unsloth QLoRA (primary): Standalone, reads from
    data/processed/shared_task_train.csv and shared_task_eval.csv.
  - BioMistral 7B + QLoRA (legacy): Retrieval-augmented, requires NER + index.
  - Flan-T5-large (legacy): Seq2Seq, full fine-tune.

Run standalone (Mistral + Unsloth):
    python -m src.generator_finetune
    accelerate launch -m src.generator_finetune   # multi-GPU

Run via main.py (legacy retrieval-augmented):
    python main.py --mode finetune_generator
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

# ────────────────────────────────────────────────────────────────────
# Paths (aligned with main.py)
# ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DIR / "shared_task_train.csv"
EVAL_CSV = PROCESSED_DIR / "shared_task_eval.csv"
OUTPUT_DIR_MISTRAL = Path("models/v03_eos_fixed")


# ====================================================================
# A.  Load SOAP dataset (standalone Mistral flow)
# ====================================================================
def load_soap_dataset(
    train_path: Path | str = TRAIN_CSV,
    eval_path: Path | str = EVAL_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and eval CSVs with columns id, note, dialogue.

    Returns
    -------
    train_df, eval_df : pd.DataFrame
    """
    train_path = Path(train_path)
    eval_path = Path(eval_path)
    if not train_path.exists():
        sys.exit(f"[ERROR] {train_path} not found.")
    if not eval_path.exists():
        sys.exit(f"[ERROR] {eval_path} not found.")

    # engine='python' handles multi-line quoted fields (SOAP notes) correctly
    train_df = pd.read_csv(train_path, engine="python", on_bad_lines="warn")
    eval_df = pd.read_csv(eval_path, engine="python", on_bad_lines="warn")

    required = {"id", "note", "dialogue"}
    for name, df in [("train", train_df), ("eval", eval_df)]:
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"[ERROR] {name} CSV missing columns: {missing}. Expected: id, note, dialogue.")

    print(f"Loaded train: {len(train_df)} rows from {train_path}")
    print(f"Loaded eval:  {len(eval_df)} rows from {eval_path}")
    return train_df, eval_df


def format_dialogue_to_note(
    dialogue: str,
    note: str,
) -> str:
    """
    Format as: dialogue as input, SOAP note as target.
    Mistral-Instruct style: [INST] instruction [/INST] response
    """
    instruction = (
        "Convert the following clinical dialogue into a SOAP note. "
        "Include sections: Subjective, Objective, Assessment, Plan."
    )
    return f"{instruction}\n\nDialogue:\n{dialogue}\n\n[/INST]\n{note}</s>"


# ====================================================================
# B.  Mistral-7B + Unsloth QLoRA (standalone)
# ====================================================================
def finetune_mistral_unsloth(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: str | Path = OUTPUT_DIR_MISTRAL,
    *,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 8192,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    save_steps: int = 500,
    eval_steps: int = 500,
) -> None:
    """
    Fine-tune Mistral-7B-Instruct-v0.3 with Unsloth QLoRA.
    Reads from pre-split CSVs (id, note, dialogue). No NER/retrieval.

    Requires: pip install unsloth
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Mistral+Unsloth requires: pip install unsloth"
        ) from e

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("[WARN] CUDA not available. Training will be slow on CPU.")

    # HF_HOME from environment (user sets before run, e.g. on HPC)
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt/target strings
    def row_to_text(row: pd.Series) -> str:
        # Format: [INST] instruction + dialogue [/INST] note
        text = format_dialogue_to_note(
            str(row["dialogue"]).strip(),
            str(row["note"]).strip(),
        )
        # Wrap in [INST] for Mistral chat format
        return f"[INST] {text}"

    train_texts = [row_to_text(row) for _, row in train_df.iterrows()]
    eval_texts = [row_to_text(row) for _, row in eval_df.iterrows()]

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    print(f"\n--- Loading {model_name} (Unsloth 4-bit QLoRA) ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: float16 or bfloat16
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=5,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=50,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )

    print(f"\n--- Starting Mistral QLoRA fine-tuning ---")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Batch size: {per_device_train_batch_size} x grad_accum={gradient_accumulation_steps}")
    print(f"  LR: {learning_rate}, scheduler: {lr_scheduler_type}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Save/eval steps: {save_steps}")
    print(f"  Output: {output_dir}")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed / 60:.1f} min")

    # Save best model + tokenizer
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"  Best model saved → {best_dir}")


# ====================================================================
# C.  Pre-retrieve for training (legacy retrieval-augmented)
# ====================================================================
def precompute_retrieval_for_training(
    train_df: pd.DataFrame,
    retriever,
    k: int = 1,
) -> List[List[Dict[str, Any]]]:
    """Pre-retrieve top-k for every train row (excluding self)."""
    n = len(train_df)
    all_retrieved: List[List[Dict[str, Any]]] = []

    print(f"\n--- Pre-retrieving top-{k} for {n} train examples ---")
    t0 = time.time()

    for idx in range(n):
        row = train_df.iloc[idx]
        row_id = int(row["id"])
        dialogue = row["dialogue"]
        entities = json.loads(row["entities_json"])

        output = retriever.retrieve_similar(dialogue, entities, k=max(k + 2, 3))
        results = output["results"]

        filtered = [
            {"id": r["id"], "dialogue": r["dialogue"], "note": r["note"], "enriched_ner": r.get("enriched_ner", [])}
            for r in results
            if int(r["id"]) != row_id
        ][:k]

        all_retrieved.append(filtered)

        if (idx + 1) % 50 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{n}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done – {elapsed:.1f}s total  ({elapsed / n:.3f}s each)")
    return all_retrieved


# ====================================================================
# D.  Dataset construction (legacy)
# ====================================================================
def _row_to_example(
    row: pd.Series,
    retrieved_examples: List[Dict],
    use_enriched: bool = False,
) -> Dict[str, Any]:
    """Convert annotated row + retrieved examples into prompt/target."""
    from src.prompting import build_generator_prompt, build_generator_prompt_enriched

    demographics = json.loads(row["demographics_json"])
    if use_enriched and retrieved_examples:
        entities = json.loads(row["entities_json"])
        enriched_ner = entities.get("entities", [])
        prompt = build_generator_prompt_enriched(
            dialogue_raw=row["dialogue"],
            enriched_ner=enriched_ner,
            demographics=demographics,
            retrieved_examples=retrieved_examples,
        )
    else:
        prompt = build_generator_prompt(
            dialogue_raw=row["dialogue"],
            dialogue_tagged=row["dialogue_tagged"],
            demographics=demographics,
            retrieved_examples=retrieved_examples,
        )
    return {"id": int(row["id"]), "prompt": prompt, "target": row["note"]}


def build_train_examples(
    train_df: pd.DataFrame,
    retrieved_lists: List[List[Dict]],
    use_enriched: bool = False,
) -> List[Dict[str, Any]]:
    """Build prompt/target dicts using pre-computed retrieval."""
    assert len(train_df) == len(retrieved_lists)
    return [
        _row_to_example(train_df.iloc[idx], retrieved_lists[idx], use_enriched=use_enriched)
        for idx in range(len(train_df))
    ]


def build_dev_examples(
    dev_df: pd.DataFrame,
    retriever,
    k: int = 1,
    use_enriched: bool = False,
) -> List[Dict[str, Any]]:
    """Build dev examples with on-the-fly retrieval."""
    from src.prompting import build_generator_prompt, build_generator_prompt_enriched

    examples = []
    n = len(dev_df)
    print(f"\n--- Building {n} dev examples (on-the-fly retrieval) ---")
    t0 = time.time()

    for idx in range(n):
        row = dev_df.iloc[idx]
        dialogue = row["dialogue"]
        entities = json.loads(row["entities_json"])
        output = retriever.retrieve_similar(dialogue, entities, k=k)
        results = output["results"]
        retrieved = [
            {"id": r["id"], "dialogue": r["dialogue"], "note": r["note"], "enriched_ner": r.get("enriched_ner", [])}
            for r in results
        ]
        examples.append(_row_to_example(row, retrieved, use_enriched=use_enriched))

    print(f"  Done – {time.time() - t0:.1f}s total")
    return examples


# ====================================================================
# E.  Flan-T5 (legacy)
# ====================================================================
class _Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()}


def finetune_generator(
    train_examples: List[Dict],
    dev_examples: List[Dict],
    model_name: str = "google/flan-t5-large",
    output_dir: str = "outputs/generator_model",
    max_input_length: int = 1024,
    max_target_length: int = 768,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-5,
) -> None:
    """Fine-tune seq2seq model (legacy)."""
    has_cuda = torch.cuda.is_available()
    force_cpu = not has_cuda

    print(f"\n--- Loading {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def tokenize(examples: List[Dict]) -> dict:
        prompts = [ex["prompt"] for ex in examples]
        targets = [ex["target"] for ex in examples]
        model_inputs = tokenizer(prompts, max_length=max_input_length, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
        label_ids = labels["input_ids"]
        label_ids = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq] for seq in label_ids]
        model_inputs["labels"] = label_ids
        return model_inputs

    train_enc = tokenize(train_examples)
    dev_enc = tokenize(dev_examples)
    train_dataset = _Seq2SeqDataset(train_enc)
    dev_dataset = _Seq2SeqDataset(dev_enc)

    grad_accum = max(1, 4 // batch_size)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        use_cpu=force_cpu,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=False)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n  Best model saved → {best_dir}")


# ====================================================================
# F.  BioMistral 7B + QLoRA (legacy)
# ====================================================================
def finetune_biomistral_qlora(
    train_examples: List[Dict],
    dev_examples: List[Dict],
    model_name: str = "BioMistral/BioMistral-7B",
    output_dir: str = "outputs/generator_model",
    max_seq_length: int = 2048,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
) -> None:
    """Fine-tune BioMistral 7B with QLoRA (legacy, retrieval-augmented)."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError("BioMistral+QLoRA requires: pip install bitsandbytes peft trl") from e

    has_cuda = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"\n--- Loading {model_name} (4-bit QLoRA) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    def format_example(ex: Dict) -> str:
        return f"### Input:\n{ex['prompt']}\n\n### Output:\n{ex['target']}"

    train_texts = [format_example(ex) for ex in train_examples]
    dev_texts = [format_example(ex) for ex in dev_examples]

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": train_texts})
    dev_ds = Dataset.from_dict({"text": dev_texts})

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )

    trainer.train()
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n  Best model saved → {best_dir}")


# ====================================================================
# __main__ – Standalone Mistral + Unsloth
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dial2Note – Mistral-7B QLoRA Fine-Tuning (Unsloth)")
    print("=" * 60)

    # HF_HOME from env (set by HPC scripts or user)
    hf = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if hf:
        print(f"HF_HOME: {hf}")

    train_df, eval_df = load_soap_dataset()
    finetune_mistral_unsloth(train_df, eval_df)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
