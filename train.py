"""Training script for MediVisionAI."""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from src.models.model import build_model
from src.dataset import prepare_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, metadata, labels in pbar:
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, metadata, labels in pbar:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        metadata_path=config.METADATA_CSV,
        image_dir=args.image_dir,
        batch_size=config.BATCH_SIZE
    )
    
    # Build model
    print(f"Building model: {config.MODEL_NAME}")
    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        num_metadata_features=len(config.METADATA_FEATURES)
    )
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    print(f"\nTraining completed! Final model saved: {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MediVisionAI model')
    parser.add_argument('--image_dir', type=str, default='data/synthetic',
                        help='Directory containing images')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    train(args)
