"""
Entity-Conditioned Inference — Generate SOAP notes using vLLM.
Uses the same prompt format as finetune_entity_conditioned.py.

Usage:
    python -m ner_pipeline_v3.generate_entity_conditioned
    python -m ner_pipeline_v3.generate_entity_conditioned --temperature 0.4
"""

import argparse
import json
import os
import time
import pandas as pd
from pathlib import Path

# Reuse entity formatting from training
from ner_pipeline_v3.finetune_entity_conditioned import build_inference_prompt

# ─── Config ───
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LORA_DIR = "models/mistral_entity_conditioned/best"
EVAL_CSV = "ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv"
OUTPUT_DIR = "ner_pipeline_v3/outputs"
BATCH_SIZE = 32


def generate(temperature=0.3, max_tokens=4000):
    print("\n" + "=" * 60)
    print("  Entity-Conditioned SOAP Note Generation")
    print(f"  Model: {BASE_MODEL} + LoRA from {LORA_DIR}")
    print(f"  Temperature: {temperature}, Max tokens: {max_tokens}")
    print("=" * 60)

    # Load data
    print(f"\n  Loading eval data: {EVAL_CSV}")
    df = pd.read_csv(EVAL_CSV)
    print(f"  Rows: {len(df)}")

    # Build prompts
    print("  Building prompts...")
    prompts = []
    valid_indices = []
    for idx, row in df.iterrows():
        dialogue = str(row.get("dialogue", "")) if pd.notna(row.get("dialogue")) else ""
        entities_json = str(row.get("entities_json", "[]")) if pd.notna(row.get("entities_json")) else "[]"

        if not dialogue.strip():
            continue

        prompt = build_inference_prompt(dialogue, entities_json)
        prompts.append(prompt)
        valid_indices.append(idx)

    print(f"  Valid prompts: {len(prompts)}")

    # Load vLLM with LoRA
    print("\n  Loading vLLM...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_path = str(Path(LORA_DIR).resolve())
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA adapter: {lora_path}")

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="half",
        enforce_eager=True,
    )

    lora_request = LoRARequest(
        lora_name="entity_conditioned",
        lora_int_id=1,
        lora_path=lora_path,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # Generate in batches
    print(f"\n  Generating {len(prompts)} notes (batch_size={BATCH_SIZE})...")
    all_outputs = []
    t_start = time.time()

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        outputs = llm.generate(batch, sampling_params, lora_request=lora_request)

        for out in outputs:
            text = out.outputs[0].text.strip()
            all_outputs.append(text)

        elapsed = time.time() - t_start
        done = min(i + BATCH_SIZE, len(prompts))
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(prompts) - done
        eta = remaining / rate if rate > 0 else 0
        print(f"  [{done}/{len(prompts)}] {elapsed:.0f}s elapsed, ETA: {eta:.0f}s")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build output dataframe
    results = []
    output_idx = 0
    for idx, row in df.iterrows():
        if idx in valid_indices:
            results.append({
                "id": row.get("id", idx),
                "dialogue": row.get("dialogue", ""),
                "note": row.get("note", ""),
                "generated_note": all_outputs[output_idx],
                "entities_json": row.get("entities_json", "[]"),
            })
            output_idx += 1

    df_out = pd.DataFrame(results)

    out_path = f"{OUTPUT_DIR}/generated_entity_conditioned_t{temperature}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df_out)} notes → {out_path}")

    # Quick stats
    avg_gen_len = df_out["generated_note"].str.len().mean()
    avg_gold_len = df_out["note"].str.len().mean()
    print(f"  Avg generated length: {avg_gen_len:.0f} chars")
    print(f"  Avg gold length: {avg_gold_len:.0f} chars")

    return out_path


def evaluate(csv_path):
    """Run official evaluation."""
    print(f"\n  Running evaluation on {csv_path}...")

    df = pd.read_csv(csv_path)

    # Save in format expected by evaluate_dialogue_to_note.py
    pred_path = f"{OUTPUT_DIR}/submission_entity_conditioned.csv"
    df_sub = df[["id", "generated_note"]].rename(columns={"generated_note": "note"})
    df_sub.to_csv(pred_path, index=False)

    gold_path = "data/processed/shared_task_eval.csv"

    print(f"  Predictions: {pred_path}")
    print(f"  Gold: {gold_path}")
    print(f"  Running evaluate_dialogue_to_note.py...")

    os.system(
        f"python evaluate_dialogue_to_note.py "
        f"--pred_path {pred_path} "
        f"--ref_path {gold_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    out_path = generate(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if not args.skip_eval:
        evaluate(out_path)


if __name__ == "__main__":
    main()
