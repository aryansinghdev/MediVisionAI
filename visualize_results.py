"""Utility script to visualize training results and metrics."""
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()


def plot_metrics_comparison(metrics, output_path):
    """Plot comparison of precision, recall, and F1 scores per class."""
    class_names = list(metrics['per_class'].keys())
    precision = [metrics['per_class'][c]['precision'] for c in class_names]
    recall = [metrics['per_class'][c]['recall'] for c in class_names]
    f1_score = [metrics['per_class'][c]['f1_score'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to: {output_path}")
    plt.close()


def plot_overall_metrics(metrics, output_path):
    """Plot overall metrics as a bar chart."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, values, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Score')
    ax.set_title('Overall Model Performance')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Overall metrics saved to: {output_path}")
    plt.close()


def create_summary_report(metrics, output_path):
    """Create a text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MediVisionAI - Model Performance Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-"*60 + "\n")
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f} ({class_metrics['precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {class_metrics['recall']:.4f} ({class_metrics['recall']*100:.2f}%)\n")
            f.write(f"  F1 Score:  {class_metrics['f1_score']:.4f} ({class_metrics['f1_score']*100:.2f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Summary report saved to: {output_path}")


def visualize_results(metrics_path, output_dir='outputs'):
    """Generate all visualizations from metrics file."""
    # Load metrics
    print(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    class_names = list(metrics['per_class'].keys())
    confusion_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        confusion_matrix, 
        class_names,
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        metrics,
        os.path.join(output_dir, 'metrics_comparison.png')
    )
    
    # Plot overall metrics
    plot_overall_metrics(
        metrics,
        os.path.join(output_dir, 'overall_metrics.png')
    )
    
    # Create summary report
    create_summary_report(
        metrics,
        os.path.join(output_dir, 'performance_summary.txt')
    )
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MediVisionAI results')
    parser.add_argument('--metrics_path', type=str, 
                        default='outputs/test_metrics.json',
                        help='Path to metrics JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_path):
        print(f"Error: Metrics file not found: {args.metrics_path}")
        print("Please run evaluation first: python3 evaluate.py")
        exit(1)
    
    visualize_results(args.metrics_path, args.output_dir)
