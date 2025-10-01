"""Image preprocessing utilities including CLAHE."""
import cv2
import numpy as np
from PIL import Image
import config


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image.
    
    Args:
        image: Input image (numpy array or PIL Image)
    
    Returns:
        Processed image with CLAHE applied
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    
    # Apply CLAHE
    enhanced = clahe.apply(gray)
    
    # Convert back to 3 channels if original was color
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced


def preprocess_image(image_path, use_clahe=True):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        use_clahe: Whether to apply CLAHE preprocessing
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE if enabled
    if use_clahe:
        image = apply_clahe(image)
    
    # Resize to target size
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    return image


def normalize_image(image):
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0
