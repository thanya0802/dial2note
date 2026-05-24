import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

df = pd.read_csv("data/processed/shared_task_eval.csv")
instruction = ("Convert the following clinical dialogue into a SOAP note. "
               "Include sections: Subjective, Objective, Assessment, Plan.")

prompts = []
for _, row in df.iterrows():
    d = str(row["dialogue"]) if not pd.isna(row["dialogue"]) else ""
    prompts.append(f"[INST] {instruction}\n\nDialogue:\n{d}\n\n[/INST]\n")

# Test v0.1 first
print("=== Loading v0.1 ===")
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    enable_lora=True, max_lora_rank=64,
    max_model_len=8192, disable_log_stats=True,
)

for temp in [0.3, 0.4, 0.6, 0.7]:
    print(f"\n=== v0.1 EOS temp={temp} ===")
    params = SamplingParams(temperature=temp, top_p=1.0, max_tokens=4000)
    lora = LoRARequest("mistral_ft", 1, "models/v01_eos_fixed_model")
    outputs = llm.generate(prompts, params, lora_request=lora)
    rows = [{"id": df.iloc[i]["id"], "generated_note": outputs[i].outputs[0].text} for i in range(len(outputs))]
    tag = f"v01_eos_temp{str(temp).replace('.','')}"
    pd.DataFrame(rows).to_csv(f"outputs/submission_{tag}.csv", index=False)
    avg_len = sum(len(r["generated_note"]) for r in rows) / len(rows)
    print(f"  Saved (avg {avg_len:.0f} chars)")

del llm

# Test v0.3
print("\n=== Loading v0.3 ===")
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    enable_lora=True, max_lora_rank=64,
    max_model_len=8192, disable_log_stats=True,
)

for temp in [0.3, 0.4, 0.6, 0.7]:
    print(f"\n=== v0.3 EOS temp={temp} ===")
    params = SamplingParams(temperature=temp, top_p=1.0, max_tokens=4000)
    lora = LoRARequest("mistral_ft", 1, "models/v03_eos_fixed/best")
    outputs = llm.generate(prompts, params, lora_request=lora)
    rows = [{"id": df.iloc[i]["id"], "generated_note": outputs[i].outputs[0].text} for i in range(len(outputs))]
    tag = f"v03_eos_temp{str(temp).replace('.','')}"
    pd.DataFrame(rows).to_csv(f"outputs/submission_{tag}.csv", index=False)
    avg_len = sum(len(r["generated_note"]) for r in rows) / len(rows)
    print(f"  Saved (avg {avg_len:.0f} chars)")

print("\n=== SWEEP DONE ===")
