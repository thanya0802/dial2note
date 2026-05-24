"""
Shared Task Evaluation Script - Dialogue to Note
This script evaluates participant submissions for dialogue-to-note generation.

Usage:
    python evaluate_dialogue_to_note.py --submission <path_to_submission.csv> --output <output_path.csv>

Submission format:
    CSV file with columns: id, generated_note
    - id: matches the test set IDs
    - generated_note: the model's generated clinical note from dialogue
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from bench_utils.automatic_metrics import MetricsComputer


def load_ground_truth(gt_path):
    """Load the ground truth evaluation set."""
    try:
        gt_df = pd.read_csv(gt_path)
        required_cols = ['id', 'note', 'dialogue']
        if not all(col in gt_df.columns for col in required_cols):
            raise ValueError(f"Ground truth must contain columns: {required_cols}")
        return gt_df
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        sys.exit(1)


def load_submission(submission_path):
    """Load and validate participant submission."""
    try:
        sub_df = pd.read_csv(submission_path)
        required_cols = ['id', 'generated_note']
        if not all(col in sub_df.columns for col in required_cols):
            raise ValueError(f"Submission must contain columns: {required_cols}")
        return sub_df
    except Exception as e:
        print(f"Error loading submission: {e}")
        sys.exit(1)


def evaluate_submission(submission_df, ground_truth_df):
    """
    Evaluate submission against ground truth.
    
    Args:
        submission_df: DataFrame with columns [id, generated_note]
        ground_truth_df: DataFrame with columns [id, note, dialogue]
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Merge on ID to align predictions with ground truth
    merged = ground_truth_df.merge(submission_df, on='id', how='inner')
    
    if len(merged) != len(ground_truth_df):
        missing_ids = set(ground_truth_df['id']) - set(merged['id'])
        print(f"WARNING: {len(missing_ids)} IDs missing from submission: {missing_ids}")
    
    if len(merged) == 0:
        print("ERROR: No matching IDs found between submission and ground truth")
        sys.exit(1)
    
    # Prepare lists for metrics computation
    predictions = merged['generated_note'].fillna('').astype(str).tolist()
    references = merged['note'].fillna('').astype(str).tolist()
    
    # Compute metrics
    print(f"\nEvaluating {len(predictions)} samples...")
    metrics_computer = MetricsComputer(predictions, references)
    
    results = {}
    
    print("Computing BLEU...")
    results['bleu'] = metrics_computer.compute_BLEU()
    
    print("Computing ROUGE...")
    rouge_scores = metrics_computer.compute_ROUGE()
    results['rouge1'] = rouge_scores['rouge1']
    results['rouge2'] = rouge_scores['rouge2']
    results['rougeL'] = rouge_scores['rougeL']
    results['rougeLsum'] = rouge_scores['rougeLsum']
    
    print("Computing METEOR...")
    results['meteor'] = metrics_computer.compute_METEOR()
    
    # Add metadata
    results['num_samples'] = len(merged)
    results['coverage'] = len(merged) / len(ground_truth_df) * 100
    
    return results


def save_results(results, output_path, submission_name):
    """Save evaluation results to CSV."""
    # Create a DataFrame from results
    results_df = pd.DataFrame([results])
    results_df.insert(0, 'submission', submission_name)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def print_results(results):
    """Print formatted results to console."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples Evaluated: {results['num_samples']}")
    print(f"Coverage: {results['coverage']:.2f}%")
    print("-"*60)
    print(f"BLEU:           {results['bleu']:.4f}")
    print(f"ROUGE-1:        {results['rouge1']:.4f}")
    print(f"ROUGE-2:        {results['rouge2']:.4f}")
    print(f"ROUGE-L:        {results['rougeL']:.4f}")
    print(f"ROUGE-Lsum:     {results['rougeLsum']:.4f}")
    print(f"METEOR:         {results['meteor']:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate dialogue-to-note generation submissions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python evaluate_dialogue_to_note.py --submission team_submission.csv --output results.csv
    
Submission CSV format:
    id,generated_note
    1,"Patient presents with..."
    2,"Chief complaint..."
        """
    )
    parser.add_argument(
        '--submission',
        type=str,
        required=True,
        help='Path to submission CSV file'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='dataset/shared_task_eval.csv',
        help='Path to ground truth CSV file (default: dataset/shared_task_eval.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results CSV (default: results/<submission_name>_eval.csv)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Ground truth loaded: {len(ground_truth)} samples")
    
    print(f"\nLoading submission from: {args.submission}")
    submission = load_submission(args.submission)
    print(f"Submission loaded: {len(submission)} samples")
    
    # Evaluate
    results = evaluate_submission(submission, ground_truth)
    
    # Print results
    print_results(results)
    
    # Save results
    submission_name = Path(args.submission).stem
    if args.output is None:
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{submission_name}_eval.csv"
    else:
        output_path = Path(args.output)
    
    save_results(results, output_path, submission_name)
    
    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
