"""
Analyze validation predictions to determine if the model is learning meaningful structure
or simply centering predictions around 50.

This script loads saved validation prediction JSON files and generates visualizations
comparing predicted vs ground truth distributions.

Usage:
    python analyze_validation_predictions.py --pred_dir ./ckpt/llama3_8B/lora_mc/predictions
    python analyze_validation_predictions.py --pred_dir ./ckpt/llama3_8B/lora_mc/predictions --output_dir ./analysis_figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_predictions(filepath: str) -> Dict:
    """Load predictions from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_means_and_modes(data: Dict) -> Tuple[List[float], List[float], List[int], List[int]]:
    """
    Extract predicted and ground truth means and modes from prediction data.

    Returns:
        pred_means, gt_means, pred_modes, gt_modes
    """
    pred_means = []
    gt_means = []
    pred_modes = []
    gt_modes = []

    for example in data['examples']:
        for pred, gt in zip(example['predictions'], example['ground_truth']):
            pred_means.append(pred['predicted_mean'])
            gt_means.append(gt['ground_truth_mean'])
            pred_modes.append(pred['predicted_mode'])
            gt_modes.append(gt['ground_truth_mode'])

    return pred_means, gt_means, pred_modes, gt_modes


def compute_direction_metrics(pred_means: List[float], gt_means: List[float],
                               threshold: float = 50.0) -> Dict:
    """
    Compute metrics related to directional accuracy.

    Args:
        pred_means: Predicted means
        gt_means: Ground truth means
        threshold: Center threshold (default 50)

    Returns:
        Dictionary with direction metrics
    """
    pred_means = np.array(pred_means)
    gt_means = np.array(gt_means)

    # Direction relative to threshold
    pred_above = pred_means > threshold
    gt_above = gt_means > threshold

    # Direction accuracy: does model predict same side of 50 as ground truth?
    direction_correct = pred_above == gt_above
    direction_accuracy = np.mean(direction_correct)

    # For cases where GT is clearly not at 50 (e.g., |GT - 50| > 10)
    clear_cases = np.abs(gt_means - threshold) > 10
    if np.sum(clear_cases) > 0:
        clear_direction_accuracy = np.mean(direction_correct[clear_cases])
    else:
        clear_direction_accuracy = np.nan

    # Correlation between predicted and ground truth means
    correlation = np.corrcoef(pred_means, gt_means)[0, 1]

    # Mean absolute error
    mae = np.mean(np.abs(pred_means - gt_means))

    # Bias towards 50: average distance of predictions from 50 vs GT from 50
    pred_spread = np.mean(np.abs(pred_means - threshold))
    gt_spread = np.mean(np.abs(gt_means - threshold))
    spread_ratio = pred_spread / gt_spread if gt_spread > 0 else np.nan

    # Mean prediction vs mean GT
    mean_pred = np.mean(pred_means)
    mean_gt = np.mean(gt_means)

    return {
        'direction_accuracy': direction_accuracy,
        'clear_direction_accuracy': clear_direction_accuracy,
        'correlation': correlation,
        'mae': mae,
        'pred_spread': pred_spread,
        'gt_spread': gt_spread,
        'spread_ratio': spread_ratio,  # <1 means predictions are more centered than GT
        'mean_pred': mean_pred,
        'mean_gt': mean_gt,
        'n_samples': len(pred_means),
    }


def plot_prediction_vs_ground_truth(pred_means: List[float], gt_means: List[float],
                                     title: str, output_path: str):
    """
    Create scatter plot of predicted vs ground truth means.
    """
    pred_means = np.array(pred_means)
    gt_means = np.array(gt_means)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Scatter plot
    ax.scatter(gt_means, pred_means, alpha=0.5, s=20, c='blue', label='Predictions')

    # Perfect prediction line (y = x)
    ax.plot([0, 100], [0, 100], 'g--', linewidth=2, label='Perfect (y=x)')

    # Centering line (y = 50)
    ax.axhline(y=50, color='r', linestyle=':', linewidth=2, label='Centered (y=50)')
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Compute metrics for annotation
    correlation = np.corrcoef(pred_means, gt_means)[0, 1]
    mae = np.mean(np.abs(pred_means - gt_means))

    ax.set_xlabel('Ground Truth Mean', fontsize=12)
    ax.set_ylabel('Predicted Mean', fontsize=12)
    ax.set_title(f'{title}\nCorr: {correlation:.3f}, MAE: {mae:.2f}', fontsize=14)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_direction_analysis(pred_means: List[float], gt_means: List[float],
                            title: str, output_path: str):
    """
    Create visualization focusing on directional accuracy.
    """
    pred_means = np.array(pred_means)
    gt_means = np.array(gt_means)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Scatter with quadrants colored by correctness
    ax = axes[0, 0]
    pred_above = pred_means > 50
    gt_above = gt_means > 50
    correct = pred_above == gt_above

    ax.scatter(gt_means[correct], pred_means[correct], alpha=0.5, s=20,
               c='green', label=f'Correct direction ({np.sum(correct)})')
    ax.scatter(gt_means[~correct], pred_means[~correct], alpha=0.5, s=20,
               c='red', label=f'Wrong direction ({np.sum(~correct)})')
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1)
    ax.plot([0, 100], [0, 100], 'b:', linewidth=1, alpha=0.5)
    ax.set_xlabel('Ground Truth Mean')
    ax.set_ylabel('Predicted Mean')
    ax.set_title(f'Direction Accuracy: {np.mean(correct)*100:.1f}%')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')

    # 2. Distribution of predictions vs ground truth
    ax = axes[0, 1]
    bins = np.linspace(0, 100, 21)
    ax.hist(gt_means, bins=bins, alpha=0.5, label='Ground Truth', color='blue')
    ax.hist(pred_means, bins=bins, alpha=0.5, label='Predicted', color='orange')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Center (50)')
    ax.axvline(x=np.mean(gt_means), color='blue', linestyle='-', linewidth=2,
               label=f'GT mean: {np.mean(gt_means):.1f}')
    ax.axvline(x=np.mean(pred_means), color='orange', linestyle='-', linewidth=2,
               label=f'Pred mean: {np.mean(pred_means):.1f}')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Means')
    ax.legend(fontsize=9)

    # 3. Spread analysis: |value - 50| for predictions vs GT
    ax = axes[1, 0]
    pred_spread = np.abs(pred_means - 50)
    gt_spread = np.abs(gt_means - 50)

    ax.scatter(gt_spread, pred_spread, alpha=0.5, s=20, c='purple')
    ax.plot([0, 50], [0, 50], 'g--', linewidth=2, label='Equal spread')
    ax.axhline(y=np.mean(pred_spread), color='orange', linestyle='-',
               label=f'Pred avg spread: {np.mean(pred_spread):.1f}')
    ax.axhline(y=np.mean(gt_spread), color='blue', linestyle='-',
               label=f'GT avg spread: {np.mean(gt_spread):.1f}')
    ax.set_xlabel('|Ground Truth - 50|')
    ax.set_ylabel('|Predicted - 50|')
    ax.set_title(f'Spread from Center\nRatio: {np.mean(pred_spread)/np.mean(gt_spread):.2f}')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-2, 52)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')

    # 4. Residuals vs GT (to see if there's systematic bias)
    ax = axes[1, 1]
    residuals = pred_means - gt_means
    ax.scatter(gt_means, residuals, alpha=0.5, s=20, c='teal')
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='No bias')

    # Fit a line to see trend
    z = np.polyfit(gt_means, residuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 100, 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2,
            label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')

    ax.set_xlabel('Ground Truth Mean')
    ax.set_ylabel('Prediction - Ground Truth')
    ax.set_title('Residuals vs Ground Truth\n(Negative slope = regression to 50)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-5, 105)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(results: Dict[str, Dict], output_path: str):
    """
    Create comparison plot across different checkpoints/epochs.
    """
    if len(results) < 2:
        print("Need at least 2 files for comparison plot")
        return

    # Sort by epoch
    sorted_names = sorted(results.keys(),
                          key=lambda x: results[x]['metrics'].get('epoch', 0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = []
    correlations = []
    direction_accs = []
    maes = []
    spread_ratios = []
    mean_preds = []
    mean_gts = []

    for name in sorted_names:
        metrics = results[name]['metrics']
        epoch = results[name].get('epoch', 0)
        epochs.append(epoch)
        correlations.append(metrics['correlation'])
        direction_accs.append(metrics['direction_accuracy'])
        maes.append(metrics['mae'])
        spread_ratios.append(metrics['spread_ratio'])
        mean_preds.append(metrics['mean_pred'])
        mean_gts.append(metrics['mean_gt'])

    # Plot metrics over epochs
    ax = axes[0, 0]
    ax.plot(epochs, correlations, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title('Pred-GT Correlation')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, direction_accs, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Direction Accuracy (above/below 50)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, maes, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, spread_ratios, 'mo-', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='green', linestyle='--', label='Equal spread')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spread Ratio (pred/GT)')
    ax.set_title('Spread Ratio\n(<1 = more centered)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, mean_preds, 'o-', linewidth=2, markersize=8, label='Pred mean')
    ax.plot(epochs, mean_gts, 's--', linewidth=2, markersize=8, label='GT mean')
    ax.axhline(y=50, color='red', linestyle=':', label='Center (50)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Value')
    ax.set_title('Average Prediction vs GT Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 2]
    ax.axis('off')

    # Create summary text
    summary_text = "Summary Statistics:\n\n"
    for name in sorted_names:
        metrics = results[name]['metrics']
        epoch = results[name].get('epoch', 0)
        tag = results[name].get('tag', '')
        summary_text += f"Epoch {epoch} ({tag}):\n"
        summary_text += f"  Correlation: {metrics['correlation']:.3f}\n"
        summary_text += f"  Direction Acc: {metrics['direction_accuracy']*100:.1f}%\n"
        summary_text += f"  MAE: {metrics['mae']:.2f}\n"
        summary_text += f"  Spread Ratio: {metrics['spread_ratio']:.3f}\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Model Performance Across Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_learning_vs_centering(pred_means: List[float], gt_means: List[float]) -> str:
    """
    Analyze whether model is learning structure or just centering.
    Returns a text summary.
    """
    metrics = compute_direction_metrics(pred_means, gt_means)

    summary = []
    summary.append("=" * 60)
    summary.append("ANALYSIS: Learning vs Centering")
    summary.append("=" * 60)

    # Correlation analysis
    corr = metrics['correlation']
    if corr > 0.7:
        summary.append(f"[GOOD] High correlation ({corr:.3f}): Model captures GT structure")
    elif corr > 0.4:
        summary.append(f"[MODERATE] Medium correlation ({corr:.3f}): Some structure learned")
    elif corr > 0.1:
        summary.append(f"[WEAK] Low correlation ({corr:.3f}): Weak structure learning")
    else:
        summary.append(f"[POOR] Very low correlation ({corr:.3f}): No structure learned")

    # Direction accuracy
    dir_acc = metrics['direction_accuracy']
    if dir_acc > 0.7:
        summary.append(f"[GOOD] High direction accuracy ({dir_acc*100:.1f}%): Correct side of 50")
    elif dir_acc > 0.55:
        summary.append(f"[MODERATE] Direction accuracy ({dir_acc*100:.1f}%): Better than random")
    else:
        summary.append(f"[POOR] Low direction accuracy ({dir_acc*100:.1f}%): Near random")

    # Spread ratio analysis
    spread_ratio = metrics['spread_ratio']
    if spread_ratio > 0.8:
        summary.append(f"[GOOD] Spread ratio ({spread_ratio:.3f}): Not overly centered")
    elif spread_ratio > 0.5:
        summary.append(f"[MODERATE] Spread ratio ({spread_ratio:.3f}): Some centering tendency")
    else:
        summary.append(f"[POOR] Low spread ratio ({spread_ratio:.3f}): Heavy centering to 50")

    # Overall assessment
    summary.append("")
    if corr > 0.5 and dir_acc > 0.6 and spread_ratio > 0.6:
        summary.append("OVERALL: Model appears to be learning meaningful structure")
    elif corr < 0.2 and spread_ratio < 0.5:
        summary.append("OVERALL: Model is primarily centering predictions around 50")
    else:
        summary.append("OVERALL: Mixed results - some learning with centering tendency")

    summary.append("=" * 60)

    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze validation predictions for learning vs centering behavior'
    )
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save figures (default: pred_dir/analysis)')
    parser.add_argument('--files', type=str, nargs='*', default=None,
                        help='Specific files to analyze (default: all JSON files)')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Error: Directory not found: {pred_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else pred_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find prediction files
    if args.files:
        pred_files = [pred_dir / f for f in args.files]
    else:
        pred_files = list(pred_dir.glob('val_predictions_*.json'))

    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return

    print(f"Found {len(pred_files)} prediction file(s)")

    # Analyze each file
    all_results = {}

    for filepath in sorted(pred_files):
        print(f"\nAnalyzing: {filepath.name}")

        try:
            data = load_predictions(str(filepath))
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue

        pred_means, gt_means, pred_modes, gt_modes = extract_means_and_modes(data)

        if len(pred_means) == 0:
            print("  No predictions found")
            continue

        metrics = compute_direction_metrics(pred_means, gt_means)

        # Store results
        file_key = filepath.stem
        all_results[file_key] = {
            'metrics': metrics,
            'pred_means': pred_means,
            'gt_means': gt_means,
            'epoch': data.get('epoch', 0),
            'tag': data.get('tag', ''),
        }

        # Print analysis
        print(analyze_learning_vs_centering(pred_means, gt_means))

        # Generate individual plots
        plot_name = filepath.stem

        # Scatter plot
        plot_prediction_vs_ground_truth(
            pred_means, gt_means,
            title=f'{plot_name}',
            output_path=str(output_dir / f'{plot_name}_scatter.png')
        )

        # Direction analysis
        plot_direction_analysis(
            pred_means, gt_means,
            title=f'{plot_name}',
            output_path=str(output_dir / f'{plot_name}_direction.png')
        )

    # Generate comparison plot if multiple files
    if len(all_results) >= 2:
        plot_comparison(
            all_results,
            output_path=str(output_dir / 'comparison_across_epochs.png')
        )

    # Save metrics summary to JSON
    metrics_summary = {
        name: {
            'epoch': res['epoch'],
            'tag': res['tag'],
            'correlation': res['metrics']['correlation'],
            'direction_accuracy': res['metrics']['direction_accuracy'],
            'mae': res['metrics']['mae'],
            'spread_ratio': res['metrics']['spread_ratio'],
            'mean_pred': res['metrics']['mean_pred'],
            'mean_gt': res['metrics']['mean_gt'],
            'n_samples': res['metrics']['n_samples'],
        }
        for name, res in all_results.items()
    }

    summary_path = output_dir / 'metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved metrics summary to: {summary_path}")

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
