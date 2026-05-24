# Dial2Note – HPC Cluster Setup Guide

This guide helps you run the Dial2Note pipeline on an HPC cluster (SLURM-based).

---

## Discovery Cluster (NEU) – Pre-configured

The scripts in `scripts/hpc/` are configured for **Discovery** at Northeastern:

| Setting | Value |
|---------|-------|
| Modules | `anaconda3/2024.06`, `cuda/12.8.0` |
| HF cache | `/scratch/$USER/hf_cache` |
| Fine-tune | `partition=multigpu`, `gpu:h200:1`, 128G, 8hr |
| Generate/Index/NER | `partition=gpu`, `gpu:a100:1`, 48G |
| Evaluate | `partition=short`, CPU, 8G |

**Execution order:**
```bash
cd ~/dial2note
sbatch scripts/hpc/run_ner.sh          # ~4 hr
sbatch scripts/hpc/run_build_index.sh  # ~4 hr
sbatch scripts/hpc/run_finetune.sh     # ~8 hr (H200)
sbatch scripts/hpc/run_generate.sh      # ~6 hr
sbatch scripts/hpc/run_evaluate.sh     # ~30 min
# Or agentic pipeline:
sbatch scripts/hpc/run_agentic.sh
```

**One-time setup (interactive session):**
```bash
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=32G --time=02:00:00 --pty bash
module load anaconda3/2024.06
module load cuda/12.8.0
bash scripts/hpc/setup_env.sh
```

---

## 1. Transfer the Project

```bash
# From your Mac, sync to Discovery
scp -r /Users/thanya/Desktop/dial2note mysoresanthosh.th@xfer.discovery.neu.edu:~/dial2note/
```

Or use `rsync`, `sftp`, or your institution's file transfer tool.

---

## 2. Environment Setup on the Cluster

### Option A: Use setup_env.sh (Discovery)

```bash
# Get interactive GPU session first
srun --partition=gpu-interactive --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash

module load anaconda3/2024.06
module load cuda/12.8.0
cd ~/dial2note
bash scripts/hpc/setup_env.sh
```

### Option B: Manual conda setup

```bash
module load anaconda3/2024.06
module load cuda/12.8.0
conda create -n dial2note python=3.10 -y
conda activate dial2note

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 3. Prepare Data

Place the pre-split MedSynth shared task files in `data/processed/`:

- `data/processed/shared_task_train.csv` – training set
- `data/processed/shared_task_eval.csv` – evaluation set

CSV format: columns `id`, `dialogue`, `note`.

```bash
mkdir -p data/processed
# Copy or symlink your files:
# cp /path/to/train.csv data/processed/shared_task_train.csv
# cp /path/to/eval.csv data/processed/shared_task_eval.csv
```

---

## 4. Create Logs Directory

```bash
mkdir -p logs
```

---

## 5. Adjust SLURM Scripts for Your Cluster

The scripts are pre-configured for Discovery. For other clusters, edit:

| Setting | Discovery default | What to change |
|---------|--------------------|----------------|
| `#SBATCH --partition=` | `multigpu` / `gpu` / `short` | Your partition names |
| `#SBATCH --gres=` | `gpu:h200:1` / `gpu:a100:1` | Your GPU type |
| `module load` | `anaconda3/2024.06`, `cuda/12.8.0` | Your module names |
| `HF_HOME` | `/scratch/$USER/hf_cache` | Your scratch path |

**Find your cluster's partitions:**
```bash
sinfo -o "%P %G %N %l" | grep gpu
module avail cuda
module avail anaconda
```

---

## 6. Run the Pipeline

### Option A: Run phases one by one (recommended for first run)

```bash
cd ~/dial2note

# Phase 1: NER (CPU, ~30 min)
sbatch scripts/hpc/run_ner.sh

# Phase 3: Build index (CPU, ~10 min)
sbatch scripts/hpc/run_build_index.sh

# Phase 4: Fine-tune (GPU, ~2–12 hours depending on data size)
sbatch scripts/hpc/run_finetune.sh

# Phase 5: Generate (GPU, ~6 hr)
sbatch scripts/hpc/run_generate.sh

# Phase 7: Evaluate (CPU, ~30 min)
sbatch scripts/hpc/run_evaluate.sh

# Agentic pipeline (optional)
sbatch scripts/hpc/run_agentic.sh
```

### Option B: Full pipeline in one job

```bash
sbatch scripts/hpc/run_full_pipeline.sh
```

### Option C: Interactive run (for debugging)

```bash
# Request an interactive GPU session (Discovery)
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=32G --time=02:00:00 --pty bash

cd ~/dial2note
module load anaconda3/2024.06
module load cuda/12.8.0
source activate dial2note
python main.py --mode ner
# etc.
```

---

## 7. Monitor Jobs

```bash
squeue -u $USER          # Your running jobs
sacct -u $USER           # Recent job history
tail -f logs/ner_12345.out   # Follow output (replace 12345 with job ID)
```

---

## 8. Common Cluster-Specific Tweaks

### If your cluster uses different module names

Edit each `scripts/hpc/*.sh` and uncomment/modify:

```bash
module load python/3.10    # or python3, anaconda3, etc.
module load cuda/11.8       # or cuda/12.1, etc.
```

### If you use conda and need to activate in SLURM (Discovery)

The scripts use:
```bash
module load anaconda3/2024.06
source activate dial2note
```

### If you hit "CUDA out of memory" during fine-tuning

Edit `scripts/hpc/run_finetune.sh`:
- Increase `--mem=32G` to `--mem=64G`
- The code already uses gradient checkpointing and CPU fallback; on HPC with a real GPU, it should use CUDA and run faster than on Mac.

### If you use PBS instead of SLURM

Replace `#SBATCH` directives with PBS equivalents, e.g.:

```bash
#PBS -N dial2note-ner
#PBS -l walltime=2:00:00
#PBS -l mem=16gb
#PBS -l nodes=1:ppn=4
#PBS -o logs/ner.out
#PBS -e logs/ner.err
```

---

## 9. Scaling to ~8500 Examples

| Phase | Approx. time (GPU) | Notes |
|-------|--------------------|-------|
| Phase 1 (NER) | 30–90 min | GLiNER batched; GPU helps |
| Phase 3 (Build index) | 5–15 min | 8500 embeddings |
| Phase 4 (Fine-tune) | 4–12 hours | Depends on epochs, batch size |
| Phase 5 (Generate dev) | 10–30 min | ~1275 dev examples |
| Phase 7 (Evaluate) | 5–15 min | NER + metrics |

**Memory:** 16–32 GB RAM is usually enough. Fine-tuning may need more if you increase batch size.

---

## 10. Output Locations

| Phase | Output |
|-------|--------|
| NER | `outputs/train_annotated.csv`, `outputs/dev_annotated.csv` |
| Build index | `outputs/faiss.index`, `outputs/retriever_meta.pkl` |
| Fine-tune | `models/mistral_finetune/best/` |
| Generate | `outputs/dev_predictions.json`, `outputs/dev_predictions_validated.json` |
| Agentic | `outputs/dev_predictions_agentic.json` |
| Evaluate | `outputs/eval_report.json`, `outputs/eval_report.txt` |

---

## 11. Quick Reference

```bash
# Check GPU availability
nvidia-smi

# Test that PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Run retrieve test (quick sanity check)
python main.py --mode retrieve_test
```
