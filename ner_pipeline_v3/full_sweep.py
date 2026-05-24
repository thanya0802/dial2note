"""Full temp sweep with ALL metrics for every temperature."""
import os, time, json
import pandas as pd
from pathlib import Path
from ner_pipeline_v3.finetune_entity_conditioned import build_inference_prompt

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
BATCH_SIZE = 16
TEMPERATURES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(EVAL_CSV)
    prompts, valid_indices = [], []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"
        if not dialogue.strip():
            continue
        prompts.append(build_inference_prompt(dialogue, entities_json))
        valid_indices.append(idx)

    print(f"Prompts: {len(prompts)}")

    lora_path = str(Path(LORA_DIR).resolve())
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64,
              max_model_len=8192, gpu_memory_utilization=0.85,
              dtype="half", enforce_eager=True)
    lora_request = LoRARequest(lora_name="ec", lora_int_id=1, lora_path=lora_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for temp in TEMPERATURES:
        print(f"\n{'='*60}")
        print(f"  Temperature: {temp}")
        print(f"{'='*60}")
        sp = SamplingParams(temperature=max(temp, 0.01), top_p=1.0, max_tokens=4000)
        outs = []
        t0 = time.time()
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            for o in llm.generate(batch, sp, lora_request=lora_request):
                outs.append(o.outputs[0].text.strip())
            print(f"  [{min(i+BATCH_SIZE,len(prompts))}/{len(prompts)}]")

        results = [{"id": df.iloc[vi].get("id",vi), "generated_note": outs[i]}
                   for i,vi in enumerate(valid_indices)]
        path = f"{OUTPUT_DIR}/full_t{temp}.csv"
        pd.DataFrame(results).to_csv(path, index=False)
        avg_len = sum(len(o) for o in outs) / len(outs)
        print(f"  Saved → {path} | Avg len: {avg_len:.0f} | Time: {time.time()-t0:.0f}s")
        os.system(f"python evaluate_dialogue_to_note.py --submission {path} --ground_truth data/processed/shared_task_eval.csv")

    print("\nALL DONE!")

if __name__ == "__main__":
    main()
