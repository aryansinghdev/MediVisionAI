"""Synthetic data generation utilities."""
import os
import numpy as np
import pandas as pd
from PIL import Image
import config


def generate_metadata(num_samples, output_path=None, seed=42):
    """
    Generate metadata CSV file with age and gender information.
    
    Args:
        num_samples: Number of samples to generate metadata for
        output_path: Path to save metadata CSV (default: config.METADATA_CSV)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with metadata
    """
    np.random.seed(seed)
    
    if output_path is None:
        output_path = config.METADATA_CSV
    
    # Generate metadata
    metadata = {
        'image_id': [f'image_{i:04d}.jpg' for i in range(num_samples)],
        'age': np.random.randint(config.AGE_RANGE[0], config.AGE_RANGE[1], num_samples),
        'gender': np.random.choice(config.GENDERS, num_samples),
        'label': np.random.choice(['Normal', 'Pneumonia', 'COVID-19'], num_samples)
    }
    
    df = pd.DataFrame(metadata)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Metadata generated and saved to: {output_path}")
    
    return df


def generate_synthetic_data(num_samples=None, output_dir=None, seed=42):
    """
    Generate synthetic chest X-ray images for testing.
    
    Args:
        num_samples: Number of synthetic images to generate
        output_dir: Directory to save synthetic images
        seed: Random seed for reproducibility
    
    Returns:
        Path to the directory containing synthetic images
    """
    np.random.seed(seed)
    
    if num_samples is None:
        num_samples = config.NUM_SYNTHETIC_SAMPLES
    
    if output_dir is None:
        output_dir = os.path.join(config.DATA_DIR, 'synthetic')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic chest X-ray images...")
    
    for i in range(num_samples):
        # Generate synthetic grayscale image (simulating X-ray)
        # Use different patterns for different classes
        image = np.random.randint(50, 200, 
                                   (config.SYNTHETIC_IMAGE_SIZE, 
                                    config.SYNTHETIC_IMAGE_SIZE), 
                                   dtype=np.uint8)
        
        # Add some structure to make it look more realistic
        # Add circular pattern (simulating chest cavity)
        center_y, center_x = config.SYNTHETIC_IMAGE_SIZE // 2, config.SYNTHETIC_IMAGE_SIZE // 2
        y, x = np.ogrid[:config.SYNTHETIC_IMAGE_SIZE, :config.SYNTHETIC_IMAGE_SIZE]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (config.SYNTHETIC_IMAGE_SIZE // 3)**2
        image[mask] = image[mask] * 0.7 + 50
        
        # Add some noise
        noise = np.random.randint(-20, 20, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        image_rgb = np.stack([image] * 3, axis=-1)
        
        # Save image
        img = Image.fromarray(image_rgb)
        img.save(os.path.join(output_dir, f'image_{i:04d}.jpg'))
    
    print(f"Synthetic images generated in: {output_dir}")
    
    # Generate corresponding metadata
    generate_metadata(num_samples, seed=seed)
    
    return output_dir
