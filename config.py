"""
Configuration file for MediVisionAI project.
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Data configuration
RANDOM_SEED = 42
IMAGE_SIZE = 224
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')

# Model configuration
MODEL_NAME = 'efficientnet_b0'  # Options: 'efficientnet_b0', 'resnet50'
NUM_CLASSES = 3  # Normal, Pneumonia, COVID-19

# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

# Optimizer configuration
OPTIMIZER = 'adamw'

# Loss function
LOSS_FN = 'cross_entropy'

# Data augmentation
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpointing
SAVE_EVERY = 10  # Save checkpoint every N epochs
SAVE_BEST = True

# Grad-CAM configuration
GRADCAM_LAYER = 'features'  # For EfficientNet-B0

# Metadata fields
METADATA_FEATURES = ['age', 'gender']  # Additional features
AGE_RANGE = (0, 100)
GENDERS = ['M', 'F']

# Synthetic data generation (if Kaggle data not available)
NUM_SYNTHETIC_SAMPLES = 1000
SYNTHETIC_IMAGE_SIZE = 224
