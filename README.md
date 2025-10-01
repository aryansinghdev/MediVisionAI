# MediVisionAI

**E23BCAU0179_MediVisionAI**: Explainable Deep Learning for Early Disease Detection using Chest X-rays

## Overview

MediVisionAI is an advanced deep learning system for automated disease detection from chest X-ray images combined with patient metadata (age, gender). The system uses state-of-the-art convolutional neural networks with explainability features through Grad-CAM visualization.

### Key Features

- **Multi-modal Learning**: Combines chest X-ray images with patient metadata (age, gender)
- **State-of-the-art Architecture**: EfficientNet-B0 (with optional ResNet50 support)
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
- **Robust Training**: AdamW optimizer with Cross-Entropy loss
- **Explainability**: Grad-CAM visualizations for model interpretability
- **Flexible Data Handling**: Supports both Kaggle datasets and synthetic data generation
- **Comprehensive Evaluation**: Detailed metrics and classification reports

### Supported Disease Classes

1. **Normal**: Healthy chest X-rays
2. **Pneumonia**: Bacterial or viral pneumonia
3. **COVID-19**: COVID-19 related pneumonia

## Project Structure

```
MediVisionAI/
├── config.py                 # Configuration parameters
├── requirements.txt          # Python dependencies
├── setup.sh                  # Environment setup script
├── run_all.sh               # Complete pipeline execution script
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── gradcam.py               # Grad-CAM visualization script
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py         # Model architecture
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # Image preprocessing (CLAHE)
│   │   ├── data_generator.py # Synthetic data generation
│   │   └── metrics.py        # Evaluation metrics
│   └── dataset.py           # Dataset class and data loaders
├── data/                    # Data directory
│   ├── synthetic/          # Synthetic images
│   └── metadata.csv        # Metadata file
├── checkpoints/            # Model checkpoints
├── outputs/                # Results and visualizations
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aryansinghdev/MediVisionAI.git
   cd MediVisionAI
   ```

2. **Run setup script**:
   ```bash
   bash setup.sh
   ```

3. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

## Usage

### Quick Start - Complete Pipeline

Run the entire pipeline (data generation, training, evaluation, and visualization):

```bash
bash run_all.sh
```

This script will:
1. Setup the environment
2. Generate synthetic data (or use Kaggle data if available)
3. Train the model for 200 epochs with batch size 64
4. Evaluate the model on test set
5. Generate Grad-CAM visualizations
6. Save all outputs and create an archive

### Individual Components

#### 1. Generate Synthetic Data

```bash
python3 -c "from src.utils.data_generator import generate_synthetic_data; generate_synthetic_data(num_samples=1000, seed=42)"
```

#### 2. Train the Model

```bash
python3 train.py --image_dir data/synthetic --epochs 200 --batch_size 64
```

Options:
- `--image_dir`: Directory containing images (default: data/synthetic)
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)

#### 3. Evaluate the Model

```bash
python3 evaluate.py --model_path checkpoints/best_model.pth --image_dir data/synthetic
```

Options:
- `--model_path`: Path to model checkpoint (default: checkpoints/best_model.pth)
- `--image_dir`: Directory containing images (default: data/synthetic)

#### 4. Generate Grad-CAM Visualization

```bash
python3 gradcam.py --image_path data/synthetic/image_0000.jpg --model_path checkpoints/best_model.pth --age 50 --gender M
```

Options:
- `--image_path`: Path to input image (required)
- `--model_path`: Path to model checkpoint (default: checkpoints/best_model.pth)
- `--age`: Patient age (default: 50)
- `--gender`: Patient gender, M or F (default: M)

## Configuration

Key configuration parameters in `config.py`:

### Model Configuration
- `MODEL_NAME`: 'efficientnet_b0' or 'resnet50'
- `NUM_CLASSES`: 3 (Normal, Pneumonia, COVID-19)
- `IMAGE_SIZE`: 224x224 pixels

### Training Configuration
- `BATCH_SIZE`: 64
- `NUM_EPOCHS`: 200
- `LEARNING_RATE`: 0.001
- `WEIGHT_DECAY`: 0.01
- `OPTIMIZER`: 'adamw'
- `LOSS_FN`: 'cross_entropy'

### Data Configuration
- `RANDOM_SEED`: 42 (for reproducibility)
- `USE_CLAHE`: True (enable CLAHE preprocessing)
- `TRAIN_RATIO`: 0.7
- `VAL_RATIO`: 0.15
- `TEST_RATIO`: 0.15

### CLAHE Configuration
- `CLAHE_CLIP_LIMIT`: 2.0
- `CLAHE_TILE_GRID_SIZE`: (8, 8)

## Outputs

The system generates the following outputs:

1. **Model Checkpoints** (`checkpoints/`):
   - `best_model.pth`: Best model based on validation accuracy
   - `final_model.pth`: Final model after all epochs
   - `checkpoint_epoch_*.pth`: Periodic checkpoints every 10 epochs

2. **Evaluation Metrics** (`outputs/`):
   - `test_metrics.json`: Detailed metrics in JSON format
   - `test_metrics_report.txt`: Classification report
   - Includes accuracy, precision, recall, F1-score, confusion matrix

3. **Visualizations** (`outputs/`):
   - `gradcam.jpg`: Grad-CAM heatmap visualization

4. **Archive**:
   - `MediVisionAI_outputs_*.zip`: Complete archive of all outputs

## Using Kaggle Datasets

To use real chest X-ray datasets from Kaggle:

1. **Setup Kaggle API**:
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   # Copy your kaggle.json to ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download Dataset** (example):
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/kaggle/
   ```

3. **Update Configuration**:
   Modify `config.py` to point to your Kaggle data directory.

4. **Update Metadata**:
   Generate or provide a metadata.csv file with columns:
   - `image_id`: Image filename
   - `age`: Patient age
   - `gender`: Patient gender (M/F)
   - `label`: Disease label (Normal/Pneumonia/COVID-19)

## Model Architecture

### MediVisionModel

The model consists of three main components:

1. **Backbone Network**: 
   - EfficientNet-B0 (default) or ResNet50
   - Pretrained on ImageNet
   - Extracts visual features from chest X-rays

2. **Metadata Processing**:
   - Fully connected layers for age and gender
   - Learns relevant metadata patterns

3. **Fusion Layer**:
   - Combines image and metadata features
   - Final classification layers with dropout

### Training Details

- **Optimizer**: AdamW with weight decay 0.01
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: 0.001 with Cosine Annealing
- **Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter
- **Preprocessing**: CLAHE enhancement
- **Normalization**: ImageNet statistics

## Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations:

- Highlights important regions in the X-ray image
- Shows which areas influenced the model's decision
- Helps radiologists understand and trust AI predictions
- Useful for detecting potential biases or errors

## Performance Metrics

The system reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise predictions
- **Classification Report**: Comprehensive per-class metrics

## Dependencies

Key libraries:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- numpy >= 1.24.0

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in `config.py`
   - Use smaller model (ensure using EfficientNet-B0)

2. **Slow training**:
   - Ensure GPU is available: `torch.cuda.is_available()`
   - Reduce image size if needed

3. **Poor performance**:
   - Increase training epochs
   - Use real dataset instead of synthetic
   - Enable data augmentation
   - Adjust learning rate

## Citation

If you use this project, please cite:

```
@software{medivisionai2024,
  author = {E23BCAU0179},
  title = {MediVisionAI: Explainable Deep Learning for Early Disease Detection},
  year = {2024},
  url = {https://github.com/aryansinghdev/MediVisionAI}
}
```

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch and torchvision teams
- Kaggle for providing medical imaging datasets
- Research community for Grad-CAM methodology

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This system is designed for research and educational purposes. For clinical use, proper validation and regulatory approval are required.