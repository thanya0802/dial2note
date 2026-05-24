# Data

## MedSynth Dataset

This project uses the MedSynth dataset for training and evaluation.

- **Paper:** Rezaie Mianroodi et al. (2025). "MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs." arXiv:2508.01401
- **Dataset:** https://huggingface.co/datasets/Ahmad0067/MedSynth
- **Code:** https://github.com/ahmadrezarm/MedSynth

### Splits Used

| Split | Examples | Source |
|---|---|---|
| Train | 8,529 | MedSynth (85%) |
| Eval | 1,506 | MedSynth (15%) |
| Test | 368 | Provided separately by SMM4H 2026 Task 4 organizers |

### Setup

1. Download MedSynth from the HuggingFace link above
2. Place the processed CSV files in `data/processed/`:
   - `shared_task_train.csv`
   - `shared_task_eval.csv`
   - `shared_task_test.csv` (from SMM4H organizers)

### Gold Note Statistics (Training Set)

- Mean length: 3,006 characters
- Median length: 2,978 characters
- 95th percentile: 3,786 characters
- Average tokens: 621
- Average sentences: 23

**Note:** The MedSynth dataset is subject to its own licensing terms. Do not redistribute.
