"""
Rebuild entity-conditioned model: NER entities + dialogue -> note
No retrieved notes, no RAG.
"""
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("models/mistral_entity_conditioned")

def build_ec_training_text(dialogue: str, note: str, entities_json) -> str:
    from src.combined_prompt import format_entities_grouped_by_soap
    
    inst = ("Convert the following clinical dialogue into a SOAP note. "
            "Include sections: Subjective, Objective, Assessment, Plan.")
    
    ents_block = format_entities_grouped_by_soap(entities_json)
    
    parts = [f"[INST] {inst}", ""]
    if ents_block.strip():
        parts.append("Extracted clinical entities:")
        parts.append(ents_block)
        parts.append("")
    parts.append(f"Dialogue:\n{dialogue}")
    parts.append("\n[/INST]")
    
    prompt = "\n".join(parts)
    return f"{prompt}\n{note.strip()}</s>"


def main():
    import sys, os
    os.environ.setdefault("HF_HOME", "/scratch/mysoresanthosh.th/hf_cache")
    
    print("=" * 60)
    print("Entity-Conditioned Fine-Tuning (NER only, no RAG)")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv("ner_pipeline_v3/outputs/train_annotated_v3_mistral.csv")
    print(f"Train: {len(train_df)} rows")
    
    # Build training texts
    texts = []
    for _, row in train_df.iterrows():
        t = build_ec_training_text(
            str(row["dialogue"]),
            str(row["note"]),
            row.get("entities_json", "{}"),
        )
        texts.append(t)
    print(f"Training texts built: {len(texts)}")
    print(f"Sample (first 300 chars): {texts[0][:300]}")
    
    # Load model
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/mistral-7b-instruct-v0.1-bnb-4bit",
        max_seq_length=8192,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    
    # Dataset
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})
    
    # Training
    from trl import SFTTrainer, SFTConfig
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_seq_length=8192,
        dataset_text_field="text",
        packing=False,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        group_by_length=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    best_dir = OUTPUT_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\nModel saved to {best_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
