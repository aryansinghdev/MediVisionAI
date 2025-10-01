"""Dataset class for MediVisionAI."""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from src.utils.preprocessing import apply_clahe


class MediVisionDataset(Dataset):
    """
    Dataset class for chest X-ray images with metadata.
    """
    def __init__(self, metadata_df, image_dir, transform=None, use_clahe=True):
        """
        Args:
            metadata_df: DataFrame containing image metadata
            image_dir: Directory containing images
            transform: Optional transform to be applied on images
            use_clahe: Whether to apply CLAHE preprocessing
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.use_clahe = use_clahe
        
        # Label mapping
        self.label_map = {'Normal': 0, 'Pneumonia': 1, 'COVID-19': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Gender encoding
        self.gender_map = {'M': 0, 'F': 1}
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        # Get metadata
        row = self.metadata_df.iloc[idx]
        image_id = row['image_id']
        age = row['age']
        gender = row['gender']
        label = row['label']
        
        # Load image
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        
        # Apply CLAHE if enabled
        if self.use_clahe:
            image = apply_clahe(np.array(image))
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare metadata features
        age_normalized = age / 100.0  # Normalize age to [0, 1]
        gender_encoded = self.gender_map.get(gender, 0)
        metadata = torch.tensor([age_normalized, gender_encoded], dtype=torch.float32)
        
        # Encode label
        label_encoded = self.label_map.get(label, 0)
        
        return image, metadata, label_encoded


def get_transforms(is_training=True):
    """
    Get image transforms for training or validation/testing.
    
    Args:
        is_training: Whether transforms are for training
    
    Returns:
        torchvision transforms
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def prepare_data_loaders(metadata_path, image_dir, batch_size=None):
    """
    Prepare train, validation, and test data loaders.
    
    Args:
        metadata_path: Path to metadata CSV file
        image_dir: Directory containing images
        batch_size: Batch size for data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Shuffle and split data
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Create datasets
    train_dataset = MediVisionDataset(
        train_df, image_dir, 
        transform=get_transforms(is_training=True),
        use_clahe=config.USE_CLAHE
    )
    
    val_dataset = MediVisionDataset(
        val_df, image_dir,
        transform=get_transforms(is_training=False),
        use_clahe=config.USE_CLAHE
    )
    
    test_dataset = MediVisionDataset(
        test_df, image_dir,
        transform=get_transforms(is_training=False),
        use_clahe=config.USE_CLAHE
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
