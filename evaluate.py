"""Evaluation script for MediVisionAI."""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import config
from src.models.model import build_model
from src.dataset import prepare_data_loaders
from src.utils.metrics import calculate_metrics, save_metrics


def evaluate(args):
    """
    Evaluate the trained model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Prepare data loaders
    print("Preparing data loaders...")
    _, _, test_loader = prepare_data_loaders(
        metadata_path=config.METADATA_CSV,
        image_dir=args.image_dir,
        batch_size=config.BATCH_SIZE
    )
    
    # Build model
    print(f"Building model: {config.MODEL_NAME}")
    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        num_metadata_features=len(config.METADATA_FEATURES)
    )
    
    # Load model weights
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, metadata, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            metadata = metadata.to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_labels, all_predictions)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1 Score: {class_metrics['f1_score']:.4f}")
    
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(metrics['classification_report'])
    
    # Save metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, 'test_metrics.json')
    save_metrics(metrics, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MediVisionAI model')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, default='data/synthetic',
                        help='Directory containing images')
    
    args = parser.parse_args()
    evaluate(args)
