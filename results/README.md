# Results

Evaluation outputs from all ablation experiments are stored here.

## Main Results (Validation Set, 1,506 dialogues)

| Configuration | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---|---|---|---|---|
| Original Agentic Pipeline | 0.2344 | 0.4539 | 0.2641 | 0.2863 | 0.5111 |
| v0.1, 4-epoch, no EOS | 0.3195 | 0.5757 | 0.3955 | 0.4395 | 0.6148 |
| v0.3, 1-epoch, no EOS | 0.3851 | 0.6507 | 0.4428 | 0.4996 | 0.6297 |
| Fine-tuned Baseline (v0.1, EOS, 1-ep) | 0.5394 | 0.7469 | 0.5254 | 0.5992 | 0.6621 |
| v0.3, 1-epoch, EOS | 0.5370 | 0.7460 | 0.5255 | 0.5992 | 0.6589 |
| RAG (truncated) | 0.3799 | 0.5757 | 0.3050 | 0.3848 | 0.5418 |
| RAG (full note, before) | 0.2517 | 0.4871 | 0.2011 | 0.2910 | 0.4833 |
| RAG (full note, after) | 0.2954 | 0.5024 | 0.2020 | 0.2986 | 0.4777 |
| RAG (full pair) | 0.3009 | 0.5020 | 0.2018 | 0.2988 | 0.4779 |
| RAFT 1-epoch + RAG | 0.5361 | 0.7402 | 0.5159 | 0.5878 | 0.6594 |
| RAFT 1-epoch, no RAG | 0.5379 | 0.7370 | 0.5115 | 0.5824 | 0.6599 |
| RAFT 2-epoch + RAG | 0.5358 | 0.7405 | 0.5159 | 0.5904 | 0.6583 |
| RAFT 2-epoch, no RAG | 0.5371 | 0.7355 | 0.5073 | 0.5799 | 0.6596 |
| Entity-Conditioned (temp 0.1) | 0.5464 | 0.7454 | 0.5219 | 0.5951 | 0.6645 |
| Combined NER+RAFT | 0.5423 | 0.7454 | 0.5245 | 0.5983 | 0.6633 |
| EC Ensemble (best-of-5) | 0.5465 | 0.7483 | 0.5255 | 0.6001 | 0.6700 |
| Few-shot (NER examples) | 0.2932 | — | — | — | — |
| Few-shot (mixed examples) | 0.2969 | — | — | — | — |
| Few-shot (dialogue examples) | 0.2995 | — | — | — | — |

## Official Test Set (SMM4H 2026 Task 4)

| Submission | BLEU | Average Score |
|---|---|---|
| Full Pipeline | 0.4427 | 0.54 |
| NER-Conditioned | 0.4415 | — |

**Rank: 1st place** among 7 participating teams.

## Evaluation Metrics

All metrics computed using the HuggingFace `evaluate` library:
- BLEU (primary ranking metric)
- ROUGE-1, ROUGE-2, ROUGE-L
- METEOR
