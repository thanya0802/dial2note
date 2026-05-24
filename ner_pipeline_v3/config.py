"""
NER Pipeline v3 Configuration.
SOAP-aligned entity extraction using GLiNER.
"""

# ─── Model ───
GLINER_MODEL = "urchade/gliner_large_bio-v0.1"

# ─── Thresholds ───
PRIMARY_THRESHOLD = 0.4        # Main confidence cutoff for GLiNER
FALLBACK_THRESHOLD = 0.25      # Used if < MIN_ENTITIES extracted at primary threshold
MAX_SPAN_WORDS = 7             # Reject entity spans longer than this
MIN_ENTITIES = 5               # Minimum entities before fallback triggers
MUST_HAVE_CC = True            # Require at least 1 chief complaint

# ─── Data Paths ───
TRAIN_CSV = "data/processed/shared_task_train.csv"   # Columns: id, note, dialogue
EVAL_CSV = "data/processed/shared_task_eval.csv"      # Columns: id, note, dialogue

# ─── Output Paths ───
TRAIN_OUTPUT = "ner_pipeline_v3/outputs/train_annotated_v3.csv"
EVAL_OUTPUT = "ner_pipeline_v3/outputs/dev_annotated_v3.csv"

# ─── Checkpointing ───
CHECKPOINT_EVERY = 50          # Save progress every N rows

# ─── The 10 SOAP-Essential Entity Labels ───
# Each key is the label name, value is the descriptive prompt for GLiNER.
# GLiNER uses these descriptions to match text spans in zero-shot mode.
# These are the ONLY labels — no body parts, no temporal expressions, no demographics.

LABEL_PROMPTS = {
    "chief complaint": (
        "the main medical problem or primary reason for the patient visit, "
        "such as chest pain, back pain, visual disturbances, difficulty breathing, or skin rash"
    ),
    "symptom": (
        "a symptom or medical complaint such as pain, fatigue, nausea, dizziness, palpitations, "
        "blurred vision, shortness of breath, headache, swelling, numbness, weight loss, "
        "insomnia, cough, fever, rash, bleeding, vomiting, weakness, or any physical problem the patient reports"
    ),
    "diagnosis": (
        "a medical diagnosis or disease such as diabetes, hypertension, anemia, PCOS, "
        "asthma, depression, anorexia, arthritis, GERD, pneumonia, or any clinical condition named by the doctor"
    ),
    "medication": (
        "a specific drug name, supplement, or prescribed medication mentioned in the "
        "conversation"
    ),
    "test or lab order": (
        "a diagnostic test, laboratory test, imaging study, or screening ordered by the "
        "doctor"
    ),
    "physical exam finding": (
        "an observation or finding from the physical examination performed by the doctor, "
        "such as tenderness, swelling, rash, or abnormal sounds"
    ),
    "vital sign": (
        "a specific vital sign measurement with a numeric value, such as blood pressure 120/80, "
        "heart rate 72, BMI 24.5, temperature 98.6, weight 150 pounds, or oxygen saturation 98 percent"
    ),
    "drug detail": (
        "dosage amount, frequency, route of administration, or duration of a prescribed "
        "medication"
    ),
    "referral": (
        "a medical specialist the patient is being referred to, such as cardiologist, "
        "dermatologist, endocrinologist, neurologist, or psychiatrist"
    ),
    "follow-up instruction": (
        "a specific instruction for the patient including follow-up appointments, "
        "lifestyle changes, dietary plans, or activity recommendations"
    ),
}

# Labels list for GLiNER (flat list of descriptive prompts)
LABELS = list(LABEL_PROMPTS.values())

# Reverse map: description -> label name (to convert GLiNER output back to our labels)
DESCRIPTION_TO_LABEL = {v: k for k, v in LABEL_PROMPTS.items()}
