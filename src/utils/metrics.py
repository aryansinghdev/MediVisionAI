"""Evaluation metrics utilities."""
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import config


def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
    
    Returns:
        Dictionary containing various metrics
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return metrics


def save_metrics(metrics, output_path):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing metrics
        output_path: Path to save the metrics file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a copy for saving (some values need conversion)
    metrics_to_save = metrics.copy()
    
    # Remove classification report from JSON (it's a string)
    classification_report = metrics_to_save.pop('classification_report', None)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # Save classification report separately
    if classification_report:
        report_path = output_path.replace('.json', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report)
    
    print(f"Metrics saved to: {output_path}")
