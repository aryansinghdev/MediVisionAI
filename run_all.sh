#!/bin/bash

# Complete pipeline script for MediVisionAI
echo "==================================================="
echo "MediVisionAI - Complete Pipeline Execution"
echo "==================================================="

# Exit on error
set -e

# Step 1: Setup environment
echo ""
echo "Step 1: Setting up environment..."
echo "---------------------------------------------------"
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Step 2: Check for Kaggle data or generate synthetic data
echo ""
echo "Step 2: Preparing data..."
echo "---------------------------------------------------"

DATA_SOURCE="synthetic"

# Check if Kaggle credentials exist
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Kaggle credentials found. Checking for datasets..."
    # You can add specific Kaggle dataset download commands here
    # Example: kaggle datasets download -d <dataset-name>
    # For now, we'll use synthetic data
    echo "Note: Using synthetic data. To use Kaggle data, add download command to this script."
fi

# Generate synthetic data
echo "Generating synthetic data..."
python3 -c "
import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils.data_generator import generate_synthetic_data
generate_synthetic_data(num_samples=1000, seed=42)
print('Synthetic data generation completed!')
"

IMAGE_DIR="data/synthetic"

# Step 3: Train the model
echo ""
echo "Step 3: Training model..."
echo "---------------------------------------------------"
python3 train.py --image_dir "$IMAGE_DIR" --epochs 200 --batch_size 64

# Step 4: Evaluate the model
echo ""
echo "Step 4: Evaluating model..."
echo "---------------------------------------------------"
python3 evaluate.py --image_dir "$IMAGE_DIR" --model_path checkpoints/best_model.pth

# Step 5: Generate Grad-CAM visualization
echo ""
echo "Step 5: Generating Grad-CAM visualization..."
echo "---------------------------------------------------"
# Use the first test image for Grad-CAM
FIRST_IMAGE=$(ls "$IMAGE_DIR"/*.jpg | head -1)
python3 gradcam.py --image_path "$FIRST_IMAGE" --model_path checkpoints/best_model.pth --age 50 --gender M

# Step 6: Create output archive
echo ""
echo "Step 6: Creating output archive..."
echo "---------------------------------------------------"

# Create zip file with all outputs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ZIP="MediVisionAI_outputs_${TIMESTAMP}.zip"

echo "Archiving outputs to $OUTPUT_ZIP..."
zip -r "$OUTPUT_ZIP" \
    checkpoints/best_model.pth \
    checkpoints/final_model.pth \
    outputs/test_metrics.json \
    outputs/test_metrics_report.txt \
    outputs/gradcam.jpg \
    config.py \
    README.md

echo ""
echo "==================================================="
echo "Pipeline execution completed successfully!"
echo "==================================================="
echo ""
echo "Generated files:"
echo "  - Best model: checkpoints/best_model.pth"
echo "  - Final model: checkpoints/final_model.pth"
echo "  - Test metrics: outputs/test_metrics.json"
echo "  - Classification report: outputs/test_metrics_report.txt"
echo "  - Grad-CAM visualization: outputs/gradcam.jpg"
echo "  - Output archive: $OUTPUT_ZIP"
echo ""
echo "==================================================="
