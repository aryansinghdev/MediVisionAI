"""Grad-CAM visualization script for MediVisionAI."""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import config
from src.models.model import build_model
from src.dataset import get_transforms


class GradCAM:
    """Grad-CAM implementation for model visualization."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get the target layer
        if hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, 'features'):
                # EfficientNet
                target = self.model.backbone.features
            elif hasattr(self.model.backbone, 'layer4'):
                # ResNet
                target = self.model.backbone.layer4
            else:
                raise ValueError("Could not find appropriate layer for Grad-CAM")
        else:
            raise ValueError("Model structure not supported")
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate(self, input_image, metadata, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor [1, 3, H, W]
            metadata: Metadata tensor [1, num_features]
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            Grad-CAM heatmap
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image, metadata)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate heatmap
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), class_idx
    
    def visualize(self, image, heatmap, alpha=0.5):
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (numpy array)
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor
        
        Returns:
            Visualization image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to color
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = (1 - alpha) * image + alpha * heatmap_color
        overlayed = np.uint8(overlayed)
        
        return overlayed


def generate_gradcam(args):
    """
    Generate Grad-CAM visualization.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
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
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=config.GRADCAM_LAYER)
    
    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    original_image = cv2.imread(args.image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # Prepare input
    transform = get_transforms(is_training=False)
    image_pil = Image.fromarray(original_image)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Prepare metadata (example values)
    metadata = torch.tensor([[args.age / 100.0, 0 if args.gender == 'M' else 1]], 
                           dtype=torch.float32).to(device)
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    heatmap, predicted_class = gradcam.generate(input_tensor, metadata)
    
    # Visualize
    visualization = gradcam.visualize(original_image, heatmap)
    
    # Create figure
    class_names = ['Normal', 'Pneumonia', 'COVID-19']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title(f'Overlay (Predicted: {class_names[predicted_class]})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(config.OUTPUT_DIR, 'gradcam.jpg')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualization')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--age', type=int, default=50,
                        help='Patient age')
    parser.add_argument('--gender', type=str, default='M', choices=['M', 'F'],
                        help='Patient gender')
    
    args = parser.parse_args()
    generate_gradcam(args)
