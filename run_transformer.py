"""
run_transformer.py - Training Pipeline Orchestrator for Vision Transformer

This file serves as an alternative training pipeline using the Vision Transformer architecture.
It preserves the original CNN implementation in run.py while providing an improved training
setup with the new model.

Project Structure:
- run_transformer.py (this file): Training pipeline for Vision Transformer
- vision_transformer.py: Vision Transformer model architecture
- run.py: Original CNN training pipeline
- dataset.py: Data loading and preprocessing

Key Changes from Previous Version:
1. Switched to Vision Transformer architecture for better pattern recognition
2. Updated to Binary Cross Entropy loss (more appropriate for classification)
3. Improved optimizer and learning rate scheduling
4. Added early stopping and model checkpointing
5. Implemented test predictions writer

Dependencies:
- torch: Core deep learning framework and utilities
- vision_transformer: Custom ViT model implementation
- dataset: Custom data loading utilities
"""

from dataset import HDF5Dataset
from vision_transformer import VisionTransformer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import os


def accuracy(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate binary classification accuracy.
    
    Args:
        outputs: Raw model outputs (logits)
        y: Ground truth labels
        
    Returns:
        Classification accuracy as a value between 0 and 1
    """
    y_hat = (torch.sigmoid(outputs) > .5).float()
    return (y_hat == y).float().mean()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device
) -> Tuple[float, float]:
    """Train the model for one epoch.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler (optional)
        device: Device to run the training on (CPU/GPU)
        
    Returns:
        Tuple of (average_loss, average_accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate accuracy
        with torch.no_grad():
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
            
        # Calculate loss and backpropagate
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item()
        
    return running_loss / len(data_loader), running_acc / len(data_loader)


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model performance on the validation set.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader containing validation data
        device: Device to run the validation on
        
    Returns:
        Average validation accuracy for the epoch
    """
    model.eval()
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
    
    val_acc = running_acc / len(data_loader)
    print(f"Validation Accuracy: {val_acc:.4f}")
    return val_acc


def write_test_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> None:
    """Write model predictions for the test set to a file.
    
    Args:
        model: The trained neural network model
        data_loader: DataLoader containing test data
        device: Device to run inference on
    """
    model.eval()
    predictions = []
    
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Generate predictions
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(preds.cpu().numpy().flatten().tolist())
    
    # Write predictions to file
    with open('outputs/predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred)}\n")


if __name__ == "__main__":
    # Training hyperparameters
    epochs = 100
    batch_size = 64  # Reduced from 128 to help with regularization
    learning_rate = 5e-4  # Adjusted for AdamW optimizer
    patience = 10  # Early stopping patience
    
    # Model architecture parameters
    input_dim = 2          # Number of input channels (density and recording_date)
    num_classes = 1        # Binary classification
    input_shape = 28       # Input image size (28x28)
    patch_size = 4         # Size of image patches
    emb_dim = 64          # Embedding dimension
    depth = 3             # Number of transformer blocks
    num_heads = 4         # Number of attention heads
    mlp_dim = 128         # Hidden dimension for MLP
    dropout = 0.1         # Dropout probability
    weight_decay = 0.01   # L2 regularization factor
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading setup
    train_dataset = HDF5Dataset("data/train.hdf5")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataset = HDF5Dataset("data/valid.hdf5")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataset = HDF5Dataset("data/test_no_labels.hdf5")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model initialization
    model = VisionTransformer(
        in_channels=input_dim,
        patch_size=patch_size,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler with warmup
    def warmup_cosine_schedule(step):
        warmup_steps = 100
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, epochs * len(train_loader) - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validation
        val_acc = validate_one_epoch(model, val_loader, device)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'outputs/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model for predictions
    model.load_state_dict(torch.load('outputs/best_model.pt'))
    
    # Generate predictions for test set
    write_test_predictions(model, test_loader, device)
    print("Test predictions written to outputs/predictions.txt")