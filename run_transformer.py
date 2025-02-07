"""
run_transformer.py - Training Pipeline Orchestrator for Vision Transformer

This file serves as the training pipeline using a simplified Vision Transformer architecture.
The model complexity has been intentionally reduced to prevent overfitting, with:
- Smaller embedding dimension (32)
- Fewer attention heads (2)
- Single transformer block
- Increased dropout (0.2)

Project Structure:
- run_transformer.py (this file): Training pipeline for Vision Transformer
- vision_transformer.py: Simplified Vision Transformer model architecture
- dataset.py: Data loading and preprocessing

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
from typing import Tuple, Optional, Dict, List
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Custom exception for training-related errors."""
    pass


def check_data_files(paths: List[str]) -> None:
    """Verify that all required data files exist and are accessible.
    
    Args:
        paths: List of file paths to check
        
    Raises:
        FileNotFoundError: If any required file is missing
        PermissionError: If any file cannot be accessed
    """
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read data file: {path}")


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
) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler (optional)
        device: Device to run the training on (CPU/GPU)
        
    Returns:
        Dictionary containing metrics:
        - loss: Average training loss
        - accuracy: Average training accuracy
        - learning_rate: Final learning rate of the epoch
        
    Raises:
        TrainingError: If NaN loss is encountered
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for i, (inputs, labels) in enumerate(data_loader):
        try:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy
            with torch.no_grad():
                acc = accuracy(outputs, labels)
                running_acc += acc.item()
                
            # Calculate MSE loss with sigmoid activation
            loss = F.mse_loss(torch.sigmoid(outputs), labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                raise TrainingError("NaN loss encountered during training")
                
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
            running_loss += loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TrainingError(f"GPU out of memory. Try reducing batch size. Error: {str(e)}")
            raise e
            
    metrics = {
        'loss': running_loss / len(data_loader),
        'accuracy': running_acc / len(data_loader),
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    return metrics


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance on the validation set.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader containing validation data
        device: Device to run the validation on
        
    Returns:
        Dictionary containing metrics:
        - loss: Average validation loss
        - accuracy: Average validation accuracy
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate metrics with MSE loss
            loss = F.mse_loss(torch.sigmoid(outputs), labels)
            acc = accuracy(outputs, labels)
            
            running_loss += loss.item()
            running_acc += acc.item()
    
    metrics = {
        'loss': running_loss / len(data_loader),
        'accuracy': running_acc / len(data_loader)
    }
    
    logger.info(f"Validation Metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def write_test_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: str = 'outputs/predictions.txt'
) -> None:
    """Write model predictions for the test set to a file.
    
    Args:
        model: The trained neural network model
        data_loader: DataLoader containing test data
        device: Device to run inference on
        output_path: Path to save predictions (default: 'outputs/predictions.txt')
        
    Raises:
        OSError: If unable to create output directory or write to file
    """
    model.eval()
    predictions = []
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory: {str(e)}")
    
    # Generate predictions
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(preds.cpu().numpy().flatten().tolist())
    
    # Write predictions to file
    try:
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(f"{int(pred)}\n")
    except OSError as e:
        raise OSError(f"Failed to write predictions to file: {str(e)}")


if __name__ == "__main__":
    try:
        # Training hyperparameters
        epochs = 100
        batch_size = 32  # Reduced from 64 to help prevent overfitting
        learning_rate = 1e-4  # Reduced from 5e-4 for more stable training
        patience = 15  # Increased from 10 to allow more exploration
        
        # Model architecture parameters (simplified)
        input_dim = 1          # Number of input channels (density only)
        num_classes = 1        # Binary classification
        input_shape = 28       # Input image size (28x28)
        patch_size = 4         # Size of image patches
        emb_dim = 32          # Reduced embedding dimension (was 64)
        depth = 1             # Single transformer block (was 3)
        num_heads = 2         # Reduced number of attention heads (was 4)
        mlp_dim = 64         # Reduced MLP dimension (was 128)
        dropout = 0.2         # Increased dropout probability (was 0.1)
        weight_decay = 0.1    # Increased from 0.01 for stronger regularization
        
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Check data files exist
        data_files = [
            "data/train.hdf5",
            "data/valid.hdf5",
            "data/test_no_labels.hdf5"
        ]
        check_data_files(data_files)
        
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
        
        # Model initialization with simplified architecture
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
        
        # Optimizer setup with SGD and momentum
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Linear learning rate scheduler
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=len(train_loader)*epochs
        )
        
        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            # Training
            train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"LR: {train_metrics['learning_rate']:.6f}"
            )
            
            # Validation
            val_metrics = validate_one_epoch(model, val_loader, device)
            
            # Save metrics history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'learning_rate': train_metrics['learning_rate']
            })
            
            # Early stopping check
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'outputs/best_model.pt')
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model for predictions
        model.load_state_dict(torch.load('outputs/best_model.pt'))
        
        # Generate predictions for test set
        write_test_predictions(model, test_loader, device)
        logger.info("Test predictions written to outputs/predictions.txt")
        
        # Save training history
        try:
            import json
            with open('outputs/training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2)
            logger.info("Training history saved to outputs/training_history.json")
        except Exception as e:
            logger.warning(f"Failed to save training history: {str(e)}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise