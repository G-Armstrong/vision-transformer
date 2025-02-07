"""
run.py - Training Pipeline Orchestrator

This file serves as the main entry point and orchestrator for the deep learning pipeline.
It ties together the model architecture and data loading components, handling the training
process, validation, and test predictions.

Project Structure:
- run.py (this file): Orchestrates the training process and ties everything together
- CNN_classifier.py: Provides the neural network model architecture
- dataset.py: Handles data loading and preprocessing

This script coordinates the entire machine learning workflow:
1. Configures training hyperparameters and model architecture settings
2. Initializes the CNN model from CNN_classifier.py
3. Sets up data loading using HDF5Dataset from dataset.py
4. Manages the training loop, including:
   - Model training and parameter updates
   - Validation checks
   - Learning rate scheduling
   - Progress monitoring and reporting
5. Generates predictions on the test set

The script is designed to be run as the main entry point of the project:
    python run.py

Dependencies:
- torch: Core deep learning framework and utilities
- CNN_classifier: Custom CNN model implementation
- dataset: Custom data loading utilities
"""

from dataset import HDF5Dataset
from CNN_classifier import BasicCNNClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def accuracy(outputs, y):
    """Calculate binary classification accuracy.
    
    Args:
        outputs (torch.Tensor): Raw model outputs (logits)
        y (torch.Tensor): Ground truth labels
        
    Returns:
        float: Classification accuracy as a value between 0 and 1
    """
    y_hat = (torch.sigmoid(outputs) > .5).float()  # Convert logits to binary predictions
    return (y_hat == y).float().mean()  # Calculate accuracy


def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    """Train the model for one epoch.
    
    Performs one complete pass through the training data, updating model parameters
    using backpropagation and computing training metrics.
    
    Args:
        model (nn.Module): The neural network model to train
        data_loader (DataLoader): DataLoader containing training data
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to run the training on (CPU/GPU)
        
    Returns:
        tuple: (average_loss, average_accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate accuracy (no gradient needed)
        with torch.no_grad():
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
            
        # Calculate loss and backpropagate
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        
        # Update parameters and learning rate
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        
    # Return average metrics for the epoch
    return running_loss / len(data_loader), running_acc / len(data_loader)


def validate_one_epoch(model, data_loader, device):
    """Evaluate model performance on the validation set.
    
    Performs a complete pass through the validation data without updating model parameters,
    computing accuracy metrics to monitor model performance.
    
    Args:
        model (nn.Module): The neural network model to evaluate
        data_loader (DataLoader): DataLoader containing validation data
        device (torch.device): Device to run the validation on (CPU/GPU)
        
    Returns:
        float: Average validation accuracy for the epoch
    """
    running_loss = 0.0
    with torch.no_grad():  # No gradient computation needed for validation
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, labels)
            running_loss += acc.item()
    
    val_acc = running_loss / len(data_loader)
    print("Validation Accuracy: ", val_acc)
    return val_acc


def write_test_predictions(model, data_loader, device):
    """Write model predictions for the test set to a file.
    
    Args:
        model (nn.Module): The trained neural network model
        data_loader (DataLoader): DataLoader containing test data
        device (torch.device): Device to run inference on (CPU/GPU)
    
    Note:
        Predictions are written to 'outputs/predictions.txt' with one prediction per line
    """
    pass  # TO BE FILLED IN BY APPLICANT


if __name__ == "__main__":
    # Training hyperparameters
    epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    
    # Model architecture parameters
    input_dim = 2          # Number of input channels (density and recording_date)
    num_classes = 1        # Binary classification
    input_shape = 28       # Input image size (28x28)
    proj_dim = 32         # Base projection dimension for conv layers
    mlp_dim = 128         # Hidden dimension for MLP classifier
    dropout = 0.2         # Dropout probability
    weight_decay = 1e-3   # L2 regularization factor
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading setup
    train_dataset = HDF5Dataset("data/train.hdf5")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
    val_dataset = HDF5Dataset("data/valid.hdf5")
    val_loader = DataLoader(val_dataset, batch_size=4, drop_last=True, num_workers=2)
    
    # Model initialization
    model = BasicCNNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        input_shape=input_shape,
        dropout=dropout
    ).to(device)
    
    # Optimizer and learning rate scheduler setup
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=1e-5,
        total_iters=len(train_loader) * epochs
    )
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss}")
    
    # Generate predictions for test set
    write_test_predictions(model, test_loader, device)
