"""
Llama 3.3 70B inference for SOAP note generation.
Uses existing fine-tuned Llama 70B model (dialogue-only, no NER).
"""
import os, time, json
import pandas as pd
from pathlib import Path

BASE_MODEL = "unsloth/llama-3.3-70b-instruct-bnb-4bit"
LORA_DIR = "models/llama33_70b_eos/best"
EVAL_CSV = "data/processed/shared_task_eval.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
BATCH_SIZE = 8  # smaller for 70B

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(EVAL_CSV)
    print(f"Loaded {len(df)} rows")

    # Build prompts — same format as Llama training (dialogue only)
    prompts = []
    valid_indices = []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        if not dialogue.strip():
            continue
        prompt = (
            f"[INST] Convert the following clinical dialogue into a SOAP note. "
            f"Include sections: Subjective, Objective, Assessment, Plan.\n\n"
            f"Dialogue:\n{dialogue}\n\n[/INST]\n"
        )
        prompts.append(prompt)
        valid_indices.append(idx)

    print(f"Prompts: {len(prompts)}")

    lora_path = str(Path(LORA_DIR).resolve())
    print(f"\nLoading {BASE_MODEL} + LoRA from {lora_path}...")

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="half",
        enforce_eager=True,
        tensor_parallel_size=1,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
    )

    lora_request = LoRARequest(
        lora_name="llama70b_soap",
        lora_int_id=1,
        lora_path=lora_path,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for temp in [0.1, 0.3]:
        print(f"\n{'='*50}")
        print(f"  Llama 70B — Temperature: {temp}")
        print(f"{'='*50}")

        sp = SamplingParams(temperature=max(temp, 0.01), top_p=1.0, max_tokens=4000)
        all_outputs = []
        t0 = time.time()

        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            outputs = llm.generate(batch, sp, lora_request=lora_request)
            for out in outputs:
                all_outputs.append(out.outputs[0].text.strip())
            done = min(i + BATCH_SIZE, len(prompts))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - done) / rate if rate > 0 else 0
            print(f"  [{done}/{len(prompts)}] {elapsed:.0f}s elapsed, ETA: {eta:.0f}s")

        results = [{"id": df.iloc[vi].get("id", vi), "generated_note": all_outputs[i]}
                   for i, vi in enumerate(valid_indices)]

        path = f"{OUTPUT_DIR}/llama70b_t{temp}.csv"
        pd.DataFrame(results).to_csv(path, index=False)
        avg_len = sum(len(o) for o in all_outputs) / len(all_outputs)
        print(f"  Saved → {path} | Avg len: {avg_len:.0f}")

        os.system(f"python evaluate_dialogue_to_note.py --submission {path} --ground_truth data/processed/shared_task_eval.csv")

    print("\nDone!")

if __name__ == "__main__":
    main()
