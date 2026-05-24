# NER Pipeline v3 — Dial2Note Clinical Entity Extraction

Zero-shot clinical NER using [GLiNER](https://github.com/urchade/GLiNER) to extract SOAP-aligned entities from doctor-patient dialogues. Entities extracted here are used downstream to guide a fine-tuned LLM in generating SOAP notes.

This pipeline is **standalone** — it does not import anything from the rest of the project.

---

## What it does

- Loads `urchade/gliner_large_bio-v0.1` once and tags every dialogue
- Extracts entities under **10 SOAP-essential labels** (no body parts, no demographics, no temporal noise)
- Applies 3 post-processing stages: confidence filtering → span cleaning → deduplication
- Supports checkpointing: crashes resume from the last saved row
- Evaluates output with 6 quality metrics; target Entity Usefulness Score > 70%

---

## Install

```bash
pip install -r ner_pipeline_v3/requirements.txt
```

---

## Run tagging

Process both train and eval splits (default):

```bash
python -m ner_pipeline_v3.run_tagging --split both
```

Train only:

```bash
python -m ner_pipeline_v3.run_tagging --split train
```

Eval only:

```bash
python -m ner_pipeline_v3.run_tagging --split eval
```

Outputs are saved to:
- `ner_pipeline_v3/outputs/train_annotated_v3.csv`
- `ner_pipeline_v3/outputs/dev_annotated_v3.csv`

Each output CSV has the original columns (`id`, `note`, `dialogue`) plus `entities_json` — a JSON-serialised list of entity dicts:

```json
[
  {"text": "weight loss", "label": "chief complaint", "score": 0.89},
  {"text": "palpitations", "label": "symptom", "score": 0.85}
]
```

---

## Evaluate

```bash
python -m ner_pipeline_v3.evaluate --input ner_pipeline_v3/outputs/train_annotated_v3.csv
python -m ner_pipeline_v3.evaluate --input ner_pipeline_v3/outputs/dev_annotated_v3.csv
```

Metrics are printed to console and saved to `ner_pipeline_v3/outputs/eval_metrics_v3.json`.

---

## Label schema

| Label | Description |
|---|---|
| `chief complaint` | Primary reason for the visit, stated as a concise clinical term |
| `symptom` | Physical symptom or sensation reported by the patient (pain, fatigue, palpitations, …) |
| `diagnosis` | Medical diagnosis or clinical condition identified/suspected by the doctor |
| `medication` | Specific drug name, supplement, or prescribed medication |
| `test or lab order` | Diagnostic test, imaging study, or screening ordered |
| `physical exam finding` | Observation from the physical exam (tenderness, swelling, rash, …) |
| `vital sign` | Measured physiological value (BP, HR, BMI, weight, …) |
| `drug detail` | Dosage, frequency, route, or duration of a medication |
| `referral` | Specialist or department the patient is being referred to |
| `follow-up instruction` | Follow-up appointments, lifestyle changes, or activity recommendations |

---

## Configuration

All thresholds, paths, and labels live in `config.py`. Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `PRIMARY_THRESHOLD` | 0.4 | Min confidence to keep an entity |
| `FALLBACK_THRESHOLD` | 0.25 | Used when fewer than `MIN_ENTITIES` extracted |
| `MAX_SPAN_WORDS` | 7 | Reject spans longer than this |
| `MIN_ENTITIES` | 5 | Trigger fallback if below this count |
| `MUST_HAVE_CC` | True | Promote top symptom to chief complaint if none found |
| `CHECKPOINT_EVERY` | 50 | Save partial CSV every N rows |
