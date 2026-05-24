"""
Full test pipeline: NER extraction + SOAP note generation.
Processes the 368 test dialogues and produces submission CSV.
"""
import json, os, time
import pandas as pd
from pathlib import Path

# ─── Config ───
TEST_CSV = "data/processed/test_set_participant_version.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
NER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
GEN_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"

NER_PROMPT = """You are a clinical NER system. Extract medical entities from the doctor-patient dialogue below.

Return ONLY a JSON array. No explanation, no markdown, no backticks. Just the raw JSON array.

Entity types (use EXACTLY these labels):
- "chief complaint": the primary reason for the visit (extract only 1-2)
- "symptom": physical symptoms reported by the patient
- "diagnosis": medical diagnosis or disease named by the doctor
- "medication": drug name or supplement prescribed
- "drug detail": dosage, frequency, route
- "test or lab order": diagnostic test or imaging ordered
- "vital sign": measured value with number (BP 120/80, BMI 24.5)
- "physical exam finding": doctor's observation from exam
- "referral": specialist referral
- "follow-up instruction": follow-up plan, lifestyle advice

Rules:
- Keep entity text SHORT (1-5 words max)
- Do NOT extract generic words like "doctor", "patient"
- Do NOT extract body parts alone
- Extract 10-25 entities per dialogue
- Each entity needs: "text", "label", "score" (confidence 0.0-1.0)"""

from ner_pipeline_v3.finetune_entity_conditioned import format_entities_for_prompt

def parse_entities(response):
    text = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        entities = json.loads(text)
        if isinstance(entities, list):
            return entities
    except:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return []

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} rows")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ═══ STEP 1: NER Extraction ═══
    print("\n" + "="*50)
    print("  STEP 1: NER Extraction (Mistral v0.3)")
    print("="*50)

    llm_ner = LLM(
        model=NER_MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="half",
        enforce_eager=True,
    )

    ner_prompts = []
    for _, row in df.iterrows():
        dialogue = str(row.get("dialogue", ""))
        ner_prompts.append(f"[INST] {NER_PROMPT}\n\nDialogue:\n{dialogue[:6000]} [/INST]")

    sp_ner = SamplingParams(temperature=0.1, max_tokens=1024, top_p=0.95)
    print(f"  Extracting entities from {len(ner_prompts)} dialogues...")

    ner_outputs = llm_ner.generate(ner_prompts, sp_ner)
    entities_list = []
    for out in ner_outputs:
        ents = parse_entities(out.outputs[0].text)
        entities_list.append(json.dumps(ents))

    df["entities_json"] = entities_list
    avg_ents = sum(len(json.loads(e)) for e in entities_list) / len(entities_list)
    print(f"  Avg entities per dialogue: {avg_ents:.1f}")

    # Save NER output
    df.to_csv(f"{OUTPUT_DIR}/test_annotated.csv", index=False)
    print(f"  Saved NER → {OUTPUT_DIR}/test_annotated.csv")

    # Free NER model memory
    del llm_ner
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ═══ STEP 2: SOAP Note Generation ═══
    print("\n" + "="*50)
    print("  STEP 2: SOAP Generation (Entity-conditioned v0.1)")
    print("="*50)

    lora_path = str(Path(LORA_DIR).resolve())
    llm_gen = LLM(
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

    # Build generation prompts
    gen_prompts = []
    for _, row in df.iterrows():
        dialogue = str(row.get("dialogue", ""))
        entities_json = row.get("entities_json", "[]")
        entity_text = format_entities_for_prompt(entities_json)
        prompt = (
            f"[INST] Convert the following clinical dialogue into a SOAP note. "
            f"Use the extracted clinical entities to ensure completeness. "
            f"Include sections: Subjective, Objective, Assessment, Plan.\n\n"
            f"Extracted Entities:\n{entity_text}\n\n"
            f"Dialogue:\n{dialogue}\n\n[/INST]\n"
        )
        gen_prompts.append(prompt)

    sp_gen = SamplingParams(temperature=0.1, top_p=1.0, max_tokens=4000)
    print(f"  Generating {len(gen_prompts)} SOAP notes...")

    gen_outputs = llm_gen.generate(gen_prompts, sp_gen, lora_request=lora_request)

    results = []
    for i, out in enumerate(gen_outputs):
        results.append({
            "id": df.iloc[i].get("id", i+1),
            "generated_note": out.outputs[0].text.strip(),
        })

    df_sub = pd.DataFrame(results)
    sub_path = f"{OUTPUT_DIR}/test_submission.csv"
    df_sub.to_csv(sub_path, index=False)

    avg_len = df_sub["generated_note"].str.len().mean()
    print(f"\n  Saved → {sub_path}")
    print(f"  Rows: {len(df_sub)}")
    print(f"  Avg note length: {avg_len:.0f} chars")
    print("\n  DONE! Ready to submit.")

if __name__ == "__main__":
    main()
