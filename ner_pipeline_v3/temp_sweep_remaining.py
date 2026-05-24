"""Remaining temps + ablation. Model loads once."""
import json, os, time
import pandas as pd
from pathlib import Path
from ner_pipeline_v3.finetune_entity_conditioned import build_inference_prompt

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
BATCH_SIZE = 16

# Only remaining temps + ablation
TEMPERATURES = [0.05, 0.2, 0.4, 0.6]

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(EVAL_CSV)
    prompts, no_ent_prompts, valid_indices = [], [], []

    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"
        if not dialogue.strip():
            continue
        prompts.append(build_inference_prompt(dialogue, entities_json))
        no_ent_prompts.append(build_inference_prompt(dialogue, "[]"))
        valid_indices.append(idx)

    print(f"Prompts: {len(prompts)}")

    lora_path = str(Path(LORA_DIR).resolve())
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64,
              max_model_len=8192, gpu_memory_utilization=0.85,
              dtype="half", enforce_eager=True)

    lora_request = LoRARequest(lora_name="ec", lora_int_id=1, lora_path=lora_path)

    for temp in TEMPERATURES:
        print(f"\n{'='*50}\n  Temp: {temp} WITH entities\n{'='*50}")
        sp = SamplingParams(temperature=max(temp, 0.01), top_p=1.0, max_tokens=4000)
        outs = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            for o in llm.generate(batch, sp, lora_request=lora_request):
                outs.append(o.outputs[0].text.strip())
            print(f"  [{min(i+BATCH_SIZE,len(prompts))}/{len(prompts)}]")

        results = [{"id": df.iloc[vi].get("id",vi), "generated_note": outs[i]} for i,vi in enumerate(valid_indices)]
        path = f"{OUTPUT_DIR}/sub_t{temp}.csv"
        pd.DataFrame(results).to_csv(path, index=False)
        os.system(f"python evaluate_dialogue_to_note.py --submission {path} --ground_truth data/processed/shared_task_eval.csv")

    # Ablation: no entities
    print(f"\n{'='*50}\n  ABLATION: NO entities, temp=0.3\n{'='*50}")
    sp = SamplingParams(temperature=0.3, top_p=1.0, max_tokens=4000)
    outs = []
    for i in range(0, len(no_ent_prompts), BATCH_SIZE):
        batch = no_ent_prompts[i:i+BATCH_SIZE]
        for o in llm.generate(batch, sp, lora_request=lora_request):
            outs.append(o.outputs[0].text.strip())
        print(f"  [{min(i+BATCH_SIZE,len(no_ent_prompts))}/{len(no_ent_prompts)}]")

    results = [{"id": df.iloc[vi].get("id",vi), "generated_note": outs[i]} for i,vi in enumerate(valid_indices)]
    path = f"{OUTPUT_DIR}/sub_no_entities.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    os.system(f"python evaluate_dialogue_to_note.py --submission {path} --ground_truth data/processed/shared_task_eval.csv")

    print("\nALL DONE!")

if __name__ == "__main__":
    main()
