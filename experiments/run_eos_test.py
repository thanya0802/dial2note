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

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    enable_lora=True, max_lora_rank=64,
    max_model_len=8192, disable_log_stats=True,
)

params = SamplingParams(temperature=0.5, top_p=1.0, max_tokens=4000)
lora = LoRARequest("mistral_ft", 1, "models/v01_eos_fixed_model")

print("Generating: v0.1 EOS-fixed, paper settings, NO truncation")
outputs = llm.generate(prompts, params, lora_request=lora)

rows = []
for i, out in enumerate(outputs):
    rows.append({"id": df.iloc[i]["id"], "generated_note": out.outputs[0].text})

pd.DataFrame(rows).to_csv("outputs/submission_v01_eos.csv", index=False)
print(f"Saved ({len(rows)} rows)")
