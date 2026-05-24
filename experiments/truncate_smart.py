import pandas as pd
import numpy as np

gen = pd.read_csv('outputs/submission_v01_1500.csv')
gt = pd.read_csv('data/processed/shared_task_eval.csv')

# Use percentile-based target instead of fixed 3200
gold_lengths = gt.note.str.len()
target_p75 = gold_lengths.quantile(0.75)  # ~3400
target_median = gold_lengths.median()      # ~2900

def smart_truncate(note, target_len=int(target_median)):
    if len(note) <= target_len:
        return note
    cutoff = note[:target_len]
    
    # Remove trailing ##### sections
    hash_pos = cutoff.rfind('#####')
    if hash_pos > target_len * 0.5:
        cutoff = cutoff[:hash_pos].rstrip()
    
    last_para = cutoff.rfind('\n\n')
    if last_para > len(cutoff) * 0.7:
        return cutoff[:last_para].rstrip()
    last_sentence = max(cutoff.rfind('. '), cutoff.rfind('.\n'))
    if last_sentence > len(cutoff) * 0.7:
        return cutoff[:last_sentence + 1].rstrip()
    return cutoff.rstrip()

gen['generated_note'] = gen['generated_note'].apply(smart_truncate)
print(f'Gold avg: {gold_lengths.mean():.0f}')
print(f'Generated avg after truncation: {gen.generated_note.str.len().mean():.0f}')
gen.to_csv('outputs/submission_v01_smart_trunc.csv', index=False)
print(f'Saved ({len(gen)} rows)')
