# Dial2Note: Automated SOAP Note Generation from Medical Dialogues

**Authors:** Thanya Mysore Santhosh, Deahan Yu  
**Affiliation:** Northeastern University, Khoury College of Computer Sciences  
**Contact:** {mysoresanthosh.th, d.yu}@northeastern.edu

## Overview

Dial2Note is a modular pipeline for converting doctor–patient dialogues into structured SOAP (Subjective, Objective, Assessment, Plan) notes. Developed as part of the **SMM4H 2026 Task 4** shared task, the system achieved **1st place** on the official leaderboard.

The pipeline comprises four stages:
1. **Clinical Entity Extraction** — Zero-shot LLM-based NER using Mistral-7B-Instruct-v0.3
2. **Hybrid Retrieval** — BM25 + FAISS with Reciprocal Rank Fusion
3. **SOAP Note Generation** — QLoRA fine-tuned Mistral-7B-Instruct-v0.1
4. **Post-Generation Verification** — Rule-based quality checks across 5 dimensions

## Key Results

| Configuration | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---|---|---|---|---|
| Published Baseline (MedSynth, Mistral-v0.3) | 0.5346 | 0.7441 | 0.5150 | 0.5885 | 0.6589 |
| **Fine-tuned Baseline (Ours)** | **0.5394** | **0.7469** | **0.5254** | **0.5992** | **0.6621** |
| **Entity-Conditioned (temp 0.1)** | **0.5464** | 0.7454 | 0.5219 | 0.5951 | 0.6645 |
| **EC Ensemble (best-of-5)** | **0.5465** | **0.7483** | 0.5255 | 0.6001 | **0.6700** |

### Key Findings
- **EOS token fix** is the single largest improvement (+0.22 BLEU)
- **Entity-conditioned generation** outperforms dialogue-only baseline across all temperatures
- **RAG degrades performance** when training-inference prompts are misaligned (−0.16 to −0.29 BLEU)
- **RAFT** fixes RAG degradation but does not surpass baseline
- **Verifier** achieves 97.9% pass rate but has negligible automatic metric impact

## Dataset

This work uses the [MedSynth](https://huggingface.co/datasets/Ahmad0067/MedSynth) dataset (10,035 dialogue-note pairs, 2,001 ICD-10 codes).

**Do not redistribute the dataset.** Download from the official HuggingFace link above.

Split used: 8,529 train / 1,506 eval / 368 test (test set provided separately by SMM4H 2026 organizers).

## Hardware Requirements

- **Minimum:** 1× NVIDIA A100 80GB (or equivalent)
- **Recommended:** NVIDIA H200 80GB for faster training
- Fine-tuning: ~6 hours per epoch on A100
- Inference: ~30 minutes for 1,506 dialogues via vLLM

## Environment Setup

```bash
# Clone repository
git clone https://github.com/thanya-northeastern/dial2note.git
cd dial2note

# Create conda environment
conda create -n dial2note python=3.10 -y
conda activate dial2note

# Install dependencies
pip install -r requirements.txt
```

### Software Versions Used
- Unsloth 2026.2.1
- PyTorch 2.10.0+cu128
- CUDA 12.8
- vLLM (with PagedAttention)
- TRL 0.7.x

## Repository Structure

```
dial2note/
├── src/
│   ├── inference_pipeline.py      # Main inference (all ablation configs)
│   ├── generator_finetune.py      # QLoRA fine-tuning
│   ├── hybrid_retriever.py        # BM25 + FAISS + RRF retriever
│   ├── verifier.py                # Rule-based verification
│   └── ensemble_pipeline.py       # Ensemble scoring and selection
├── ner_pipeline_v3/
│   └── mistral_extract.py         # Mistral-v0.3 zero-shot NER
├── experiments/
│   ├── run_baseline.py            # Baseline replication
│   ├── run_eos_test.py            # EOS token fix experiment
│   ├── run_temp_sweep.py          # Temperature sweep
│   ├── run_fewshot_ner.py         # Few-shot NER examples
│   ├── run_fewshot_mixed.py       # Few-shot mixed examples
│   ├── run_fewshot_dialogue.py    # Few-shot dialogue examples
│   └── eval_all.py                # Evaluation script
├── scripts/hpc/                   # SLURM job scripts
├── data/                          # Data processing (see data/README.md)
├── results/                       # Evaluation outputs
├── requirements.txt
├── HPC_SETUP.md
└── README.md
```

## Reproducing Results

### 1. Fine-tune Baseline Model
```bash
python src/generator_finetune.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --epochs 1 \
  --output_dir models/baseline
```

### 2. Run NER Extraction
```bash
python ner_pipeline_v3/mistral_extract.py \
  --input data/processed/shared_task_train.csv \
  --output ner_pipeline_v3/outputs/train_annotated.csv
```

### 3. Fine-tune Entity-Conditioned Model
```bash
python src/generator_finetune.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --entity_conditioned \
  --epochs 1 \
  --output_dir models/entity_conditioned
```

### 4. Generate and Evaluate
```bash
python src/inference_pipeline.py \
  --lora_path models/entity_conditioned/best \
  --temperature 0.1 \
  --output_csv outputs/submission_ec.csv

python experiments/eval_all.py \
  --pred outputs/submission_ec.csv \
  --gold data/processed/shared_task_eval.csv
```

### 5. Run Ensemble
```bash
python src/ensemble_pipeline.py \
  --candidate_csvs outputs/submission_t005.csv,outputs/submission_t01.csv,outputs/submission_t02.csv,outputs/submission_t03.csv,outputs/submission_t05.csv \
  --annotated_csv ner_pipeline_v3/outputs/dev_annotated_v3_mistral.csv \
  --dev_csv data/processed/shared_task_eval.csv \
  --output_csv outputs/submission_ensemble.csv
```

## Citation

```bibtex
@inproceedings{mysoresanthosh2026dial2note,
  title={Dial2Note: Automated SOAP Note Generation from Medical Dialogues},
  author={Mysore Santhosh, Thanya and Yu, Deahan},
  booktitle={Proceedings of the 11th Social Media Mining for Health Research and Applications Workshop (SMM4H)},
  year={2026}
}
```

## Acknowledgments

We thank the MedSynth dataset creators and the SMM4H 2026 shared task organizers. Experiments were conducted on the Northeastern University Explorer HPC cluster.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Note: The MedSynth dataset is subject to its own licensing terms. Please refer to the [official dataset page](https://huggingface.co/datasets/Ahmad0067/MedSynth) for details.
