"""Model architecture for MediVisionAI."""
import torch
import torch.nn as nn
from torchvision import models
import config


class MediVisionModel(nn.Module):
    """
    MediVision model combining image features with metadata.
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=3, num_metadata_features=2):
        super(MediVisionModel, self).__init__()
        self.model_name = model_name
        
        # Load pretrained backbone
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
            # Remove the final classifier
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            num_features = self.backbone.fc.in_features
            # Remove the final fc layer
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, metadata):
        """
        Forward pass.
        
        Args:
            x: Image tensor [batch_size, 3, H, W]
            metadata: Metadata tensor [batch_size, num_metadata_features]
        
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Extract image features
        img_features = self.backbone(x)
        
        # Process metadata
        meta_features = self.metadata_fc(metadata)
        
        # Combine features
        combined = torch.cat([img_features, meta_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output


def build_model(model_name=None, num_classes=None, num_metadata_features=2):
    """
    Build and return the model.
    
    Args:
        model_name: Name of the backbone model
        num_classes: Number of output classes
        num_metadata_features: Number of metadata features
    
    Returns:
        Model instance
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    model = MediVisionModel(
        model_name=model_name,
        num_classes=num_classes,
        num_metadata_features=num_metadata_features
    )
    
    return model
