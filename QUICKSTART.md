# Quick Start Guide

This guide will help you get MediVisionAI up and running in minutes.

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- Optional: CUDA-capable GPU

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/aryansinghdev/MediVisionAI.git
cd MediVisionAI

# Setup environment
bash setup.sh

# Activate environment
source venv/bin/activate
```

## Quick Test (5 minutes)

Run a quick test with 5 epochs on synthetic data:

```bash
# Generate synthetic data (100 samples)
python3 -c "from src.utils.data_generator import generate_synthetic_data; generate_synthetic_data(num_samples=100, seed=42)"

# Train for 5 epochs
python3 train.py --image_dir data/synthetic --epochs 5 --batch_size 32

# Evaluate
python3 evaluate.py --image_dir data/synthetic --model_path checkpoints/best_model.pth

# Generate Grad-CAM
python3 gradcam.py --image_path data/synthetic/image_0000.jpg --model_path checkpoints/best_model.pth
```

## Full Pipeline (2-4 hours for 200 epochs)

Run the complete pipeline:

```bash
bash run_all.sh
```

This will:
- Setup environment
- Generate/download data
- Train for 200 epochs
- Evaluate on test set
- Generate Grad-CAM visualizations
- Create output archive

## Expected Outputs

After running the pipeline, you'll find:

```
checkpoints/
├── best_model.pth          # Best model (highest validation accuracy)
├── final_model.pth         # Final model after all epochs
└── checkpoint_epoch_*.pth  # Periodic checkpoints

outputs/
├── test_metrics.json       # Evaluation metrics (JSON)
├── test_metrics_report.txt # Classification report
└── gradcam.jpg            # Grad-CAM visualization

MediVisionAI_outputs_*.zip  # Complete archive
```

## Verify Installation

Run the system test:

```bash
python3 test_system.py
```

All 5 tests should pass ✓

## Using with Real Data

1. Download chest X-ray dataset from Kaggle:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/kaggle/
   ```

2. Create/update metadata.csv with columns:
   - image_id, age, gender, label

3. Update config.py to point to your data directory

4. Run training:
   ```bash
   python3 train.py --image_dir data/kaggle/train
   ```

## Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size
python3 train.py --batch_size 16
```

**Slow training:**
```bash
# Use fewer epochs for testing
python3 train.py --epochs 10
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Modify `config.py` to customize parameters
- Try different model architectures (ResNet50)
- Experiment with hyperparameters

## Support

- Issues: [GitHub Issues](https://github.com/aryansinghdev/MediVisionAI/issues)
- Documentation: [README.md](README.md)
