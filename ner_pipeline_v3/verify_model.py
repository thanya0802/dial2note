"""
Verify: run the EXACT same pipeline on dev set to confirm BLEU ~0.54.
Same model, same temp, same everything as run_test.py.
"""
import json, os
import pandas as pd
from pathlib import Path
from ner_pipeline_v3.finetune_entity_conditioned import format_entities_for_prompt

GEN_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(EVAL_CSV)
    print(f"Dev set: {len(df)} rows")

    lora_path = str(Path(LORA_DIR).resolve())
    print(f"Model: {GEN_BASE_MODEL}")
    print(f"LoRA: {lora_path}")

    llm = LLM(
        model=GEN_BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        dtype="half",
        enforce_eager=True,
    )

    lora_request = LoRARequest(
        lora_name="entity_conditioned",
        lora_int_id=1,
        lora_path=lora_path,
    )

    # Build prompts — SAME format as run_test.py
    prompts = []
    valid_indices = []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"
        if not dialogue.strip():
            continue
        entity_text = format_entities_for_prompt(entities_json)
        prompt = (
            f"[INST] Convert the following clinical dialogue into a SOAP note. "
            f"Use the extracted clinical entities to ensure completeness. "
            f"Include sections: Subjective, Objective, Assessment, Plan.\n\n"
            f"Extracted Entities:\n{entity_text}\n\n"
            f"Dialogue:\n{dialogue}\n\n[/INST]\n"
        )
        prompts.append(prompt)
        valid_indices.append(idx)

    print(f"Prompts: {len(prompts)}")
    print(f"Temperature: 0.1")

    sp = SamplingParams(temperature=0.1, top_p=1.0, max_tokens=4000)
    outputs = llm.generate(prompts, sp, lora_request=lora_request)

    results = []
    for i, vi in enumerate(valid_indices):
        results.append({
            "id": df.iloc[vi].get("id", vi),
            "generated_note": outputs[i].outputs[0].text.strip(),
        })

    df_sub = pd.DataFrame(results)
    path = f"{OUTPUT_DIR}/verify_dev_submission.csv"
    df_sub.to_csv(path, index=False)
    avg_len = df_sub["generated_note"].str.len().mean()
    print(f"\nSaved → {path}")
    print(f"Avg length: {avg_len:.0f} chars")

    # Run evaluation
    print("\nRunning evaluation...")
    os.system(f"python evaluate_dialogue_to_note.py --submission {path} --ground_truth data/processed/shared_task_eval.csv")

if __name__ == "__main__":
    main()
