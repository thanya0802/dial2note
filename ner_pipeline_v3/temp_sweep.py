"""Temperature sweep — load model once, generate at multiple temps."""

import json
import os
import time
import pandas as pd
from pathlib import Path
from ner_pipeline_v3.finetune_entity_conditioned import build_inference_prompt

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
BATCH_SIZE = 16  # smaller batch to save memory

TEMPERATURES = [0.1, 0.2, 0.25, 0.35, 0.4, 0.5, 0.6]


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(EVAL_CSV)
    print(f"Loaded {len(df)} rows")

    # Build prompts once
    prompts = []
    valid_indices = []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"
        if not dialogue.strip():
            continue
        prompts.append(build_inference_prompt(dialogue, entities_json))
        valid_indices.append(idx)

    print(f"Prompts: {len(prompts)}")

    # Also build no-entity prompts for ablation
    no_ent_prompts = []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        if not dialogue.strip():
            continue
        no_ent_prompts.append(build_inference_prompt(dialogue, "[]"))

    # Load model ONCE
    print(f"\nLoading {BASE_MODEL} + LoRA...")
    lora_path = str(Path(LORA_DIR).resolve())

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        dtype="half",
        enforce_eager=True,  # avoid CUDA graph OOM
    )

    lora_request = LoRARequest(
        lora_name="entity_conditioned",
        lora_int_id=1,
        lora_path=lora_path,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run temp sweep WITH entities
    for temp in TEMPERATURES:
        print(f"\n{'='*50}")
        print(f"  Temperature: {temp} (WITH entities)")
        print(f"{'='*50}")

        sampling_params = SamplingParams(
            temperature=temp, top_p=1.0, max_tokens=4000,
        )

        all_outputs = []
        t0 = time.time()
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            outputs = llm.generate(batch, sampling_params, lora_request=lora_request)
            for out in outputs:
                all_outputs.append(out.outputs[0].text.strip())
            done = min(i + BATCH_SIZE, len(prompts))
            print(f"  [{done}/{len(prompts)}]")

        # Save submission
        results = []
        for i, idx in enumerate(valid_indices):
            results.append({"id": df.iloc[idx].get("id", idx), "generated_note": all_outputs[i]})

        sub_path = f"{OUTPUT_DIR}/sub_t{temp}.csv"
        pd.DataFrame(results).to_csv(sub_path, index=False)

        avg_len = sum(len(o) for o in all_outputs) / len(all_outputs)
        print(f"  Saved → {sub_path} | Avg len: {avg_len:.0f} | Time: {time.time()-t0:.0f}s")

        # Evaluate
        os.system(f"python evaluate_dialogue_to_note.py --submission {sub_path} --ground_truth data/processed/shared_task_eval.csv")

    # Run ablation: NO entities at temp 0.3
    print(f"\n{'='*50}")
    print(f"  ABLATION: NO entities, temp=0.3")
    print(f"{'='*50}")

    sampling_params = SamplingParams(
        temperature=0.3, top_p=1.0, max_tokens=4000,
    )

    all_outputs = []
    t0 = time.time()
    for i in range(0, len(no_ent_prompts), BATCH_SIZE):
        batch = no_ent_prompts[i:i + BATCH_SIZE]
        outputs = llm.generate(batch, sampling_params, lora_request=lora_request)
        for out in outputs:
            all_outputs.append(out.outputs[0].text.strip())
        done = min(i + BATCH_SIZE, len(no_ent_prompts))
        print(f"  [{done}/{len(no_ent_prompts)}]")

    results = []
    for i, idx in enumerate(valid_indices):
        results.append({"id": df.iloc[idx].get("id", idx), "generated_note": all_outputs[i]})

    sub_path = f"{OUTPUT_DIR}/sub_no_entities.csv"
    pd.DataFrame(results).to_csv(sub_path, index=False)
    print(f"  Saved → {sub_path} | Time: {time.time()-t0:.0f}s")

    os.system(f"python evaluate_dialogue_to_note.py --submission {sub_path} --ground_truth data/processed/shared_task_eval.csv")

    print("\n  ALL DONE!")


if __name__ == "__main__":
    main()
