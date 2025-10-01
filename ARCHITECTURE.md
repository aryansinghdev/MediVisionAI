# MediVisionAI Architecture

## Overview

MediVisionAI uses a multi-modal deep learning architecture that combines image features from chest X-rays with patient metadata (age, gender) for disease classification.

## System Architecture

```
Input Layer
    ├── Image Input (224x224x3)
    └── Metadata Input (age, gender)
         ↓
Preprocessing
    ├── CLAHE Enhancement
    ├── Normalization
    └── Data Augmentation (training only)
         ↓
Feature Extraction
    ├── CNN Backbone (EfficientNet-B0 / ResNet50)
    │   └── ImageNet Pretrained Weights
    └── Metadata Processing
        └── Fully Connected Layers
         ↓
Feature Fusion
    └── Concatenation Layer
         ↓
Classification Head
    ├── Fully Connected Layer (256 units)
    ├── ReLU + Dropout (0.5)
    └── Output Layer (3 classes)
         ↓
Output
    └── Softmax Probabilities
        ├── Normal
        ├── Pneumonia
        └── COVID-19
```

## Model Components

### 1. Image Processing Pipeline

#### CLAHE Preprocessing
- **Purpose**: Enhance contrast in chest X-rays
- **Parameters**:
  - Clip Limit: 2.0
  - Tile Grid Size: 8x8
- **Benefits**: 
  - Improves visibility of subtle features
  - Standardizes image quality across dataset

#### Data Augmentation (Training Only)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±10°)
- Color Jitter (brightness=0.2, contrast=0.2)
- Normalization with ImageNet statistics

### 2. CNN Backbone

#### EfficientNet-B0 (Default)
```python
Architecture:
- Input: 224x224x3
- MBConv blocks with squeeze-and-excitation
- Total parameters: ~4.3M (with metadata fusion)
- Output features: 1280-dimensional vector

Advantages:
- Excellent accuracy-to-parameters ratio
- Fast inference
- Pretrained on ImageNet
```

#### ResNet50 (Optional)
```python
Architecture:
- Input: 224x224x3
- Residual blocks with skip connections
- Total parameters: ~23M
- Output features: 2048-dimensional vector

Advantages:
- Proven architecture for medical imaging
- Deep feature extraction
- Strong transfer learning capabilities
```

### 3. Metadata Processing

```python
Metadata Features:
- Age: Normalized to [0, 1] (age / 100)
- Gender: Binary encoded (M=0, F=1)

Architecture:
Input (2) → Linear(32) → ReLU → Dropout(0.3) → Output(32)

Purpose:
- Capture age-related disease patterns
- Account for gender-specific differences
- Provide additional context for diagnosis
```

### 4. Feature Fusion

```python
Fusion Strategy: Late Fusion (Concatenation)

Image Features (1280 for EfficientNet) 
    +                                    → Combined Features
Metadata Features (32)

Advantages:
- Preserves modality-specific information
- Simple and effective
- Allows independent feature learning
```

### 5. Classification Head

```python
Architecture:
Combined Features → Linear(256) → ReLU → Dropout(0.5) → Linear(3)

Output Classes:
- Class 0: Normal
- Class 1: Pneumonia
- Class 2: COVID-19

Loss Function: Cross-Entropy Loss
```

## Training Strategy

### Optimizer: AdamW

```python
Parameters:
- Learning Rate: 0.001
- Weight Decay: 0.01 (L2 regularization)
- Betas: (0.9, 0.999)

Benefits:
- Decoupled weight decay
- Better generalization
- Adaptive learning rates
```

### Learning Rate Scheduler

```python
Strategy: Cosine Annealing
- T_max: Number of epochs (200)
- Decreases learning rate following cosine curve
- Helps fine-tune in later epochs
```

### Training Configuration

```python
Hyperparameters:
- Batch Size: 64
- Epochs: 200
- Train/Val/Test Split: 70/15/15
- Random Seed: 42 (reproducibility)

Early Stopping:
- Monitors validation accuracy
- Saves best model checkpoint
- Prevents overfitting
```

## Explainability: Grad-CAM

### Gradient-weighted Class Activation Mapping

```python
Process:
1. Forward pass → Get predictions
2. Backward pass → Compute gradients
3. Global average pooling on gradients → Weights
4. Weighted combination of feature maps
5. ReLU + Normalization → Heatmap
6. Overlay on original image

Target Layer:
- EfficientNet: Last convolutional features
- ResNet: Layer 4 (final residual block)

Benefits:
- Visual explanation of predictions
- Highlights important regions
- Builds trust with clinicians
- Detects potential biases
```

## Data Flow

### Training Phase
```
Raw Images → CLAHE → Augmentation → Normalization → Model → Loss → Backprop
Metadata → Encoding → Normalization → Model → Loss → Backprop
```

### Inference Phase
```
Raw Images → CLAHE → Normalization → Model → Predictions
Metadata → Encoding → Normalization → Model → Predictions
                                          ↓
                                     Grad-CAM Visualization
```

## Performance Optimization

### Techniques Used

1. **Transfer Learning**
   - Pretrained ImageNet weights
   - Fine-tuned on medical images

2. **Regularization**
   - Dropout (0.3 in metadata, 0.5 in classifier)
   - Weight decay (0.01)
   - Data augmentation

3. **Batch Normalization**
   - Built into EfficientNet/ResNet
   - Stabilizes training
   - Allows higher learning rates

4. **Mixed Precision Training** (Optional)
   - Can be enabled for faster training
   - Reduces memory usage
   - Maintains accuracy

## Model Variants

### Supported Configurations

1. **EfficientNet-B0 + Metadata** (Default)
   - Best balance: accuracy vs speed
   - Recommended for most use cases

2. **ResNet50 + Metadata**
   - Higher accuracy potential
   - More parameters
   - Slower inference

3. **Image-only Models**
   - Remove metadata branch
   - Pure computer vision approach

## Evaluation Metrics

### Classification Metrics

```python
Primary Metrics:
- Accuracy: Overall correctness
- Precision: Positive predictive value
- Recall: Sensitivity/True positive rate
- F1 Score: Harmonic mean of precision/recall

Per-Class Metrics:
- Individual performance for each disease
- Confusion matrix
- Classification report

Clinical Relevance:
- High recall for disease classes (minimize false negatives)
- High precision to avoid false alarms
- Balanced F1 for overall performance
```

## Future Enhancements

### Potential Improvements

1. **Architecture**
   - Attention mechanisms
   - Multi-scale feature fusion
   - Ensemble of multiple models

2. **Data**
   - External validation datasets
   - Additional metadata (symptoms, lab results)
   - Longitudinal patient data

3. **Training**
   - Class balancing techniques
   - Advanced augmentation (AutoAugment)
   - Self-supervised pretraining

4. **Explainability**
   - Multiple Grad-CAM variants
   - Integrated Gradients
   - SHAP values for metadata

## References

### Key Papers

1. EfficientNet: Rethinking Model Scaling for CNNs
2. Deep Residual Learning for Image Recognition (ResNet)
3. Grad-CAM: Visual Explanations from Deep Networks
4. CLAHE: Contrast Limited Adaptive Histogram Equalization

### Medical AI Guidelines

- FDA guidance on AI/ML-based medical devices
- WHO AI for Health guidance
- Clinical validation standards

---

**Note**: This architecture is designed for research and educational purposes. Clinical deployment requires extensive validation, regulatory approval, and integration with clinical workflows.
