"""Quick system test for MediVisionAI."""
import os
import sys

def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import cv2
        import sklearn
        import matplotlib
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    try:
        from src.utils.data_generator import generate_synthetic_data, generate_metadata
        
        # Generate small test dataset
        generate_synthetic_data(num_samples=10, seed=42)
        
        # Check files exist
        assert os.path.exists('data/synthetic/image_0000.jpg')
        assert os.path.exists('data/metadata.csv')
        
        print("✓ Data generation successful")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_model_building():
    """Test model building."""
    print("\nTesting model building...")
    try:
        from src.models.model import build_model
        
        model = build_model(model_name='efficientnet_b0', num_classes=3)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Model built successfully ({num_params:,} parameters)")
        return True
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing functions."""
    print("\nTesting preprocessing...")
    try:
        from src.utils.preprocessing import apply_clahe, preprocess_image
        import numpy as np
        
        # Test CLAHE on synthetic image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        enhanced = apply_clahe(image)
        
        assert enhanced.shape == image.shape
        
        print("✓ Preprocessing successful")
        return True
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset...")
    try:
        from src.dataset import MediVisionDataset, get_transforms
        import pandas as pd
        
        # Load metadata
        df = pd.read_csv('data/metadata.csv')
        
        # Create dataset
        transform = get_transforms(is_training=False)
        dataset = MediVisionDataset(
            df.head(5), 
            'data/synthetic', 
            transform=transform,
            use_clahe=True
        )
        
        # Test loading
        image, metadata, label = dataset[0]
        
        print(f"✓ Dataset working (loaded {len(dataset)} samples)")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("MediVisionAI System Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_model_building,
        test_preprocessing,
        test_dataset
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✓ All tests passed! System is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
