"""
run.py - Training Pipeline Implementation

This script implements the training pipeline for a binary classification task using
a CNN model. It coordinates two main components:

1. dataset.py:
   - Provides the HDF5Dataset class for efficient data loading
   - Handles both training and validation data streams
   - Processes 28x28 images with density and recording_date channels

2. CNN_classifier.py:
   - Provides the BasicCNNClassifier model architecture
   - Processes the loaded data through convolutional layers
   - Outputs binary predictions for classification

The script orchestrates these components by:
- Initializing data loaders with appropriate batch sizes
- Setting up the model with proper input dimensions
- Managing the training loop with loss computation and optimization
- Validating model performance
- Generating test predictions
"""

from dataset import HDF5Dataset
from CNN_classifier import BasicCNNClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def accuracy(outputs, y):
    """Calculate binary classification accuracy from model outputs.
    
    Converts model logits to binary predictions using sigmoid activation
    and 0.5 threshold, then compares with ground truth labels.
    
    Args:
        outputs (torch.Tensor): Raw model outputs (logits)
        y (torch.Tensor): Ground truth binary labels
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    y_hat = (torch.sigmoid(outputs) > .5).float()  # Convert logits to binary predictions
    return (y_hat == y).float().mean()  # Calculate accuracy as mean of correct predictions


def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    """Execute one training epoch.
    
    Processes all batches in the training data loader:
    1. Forwards data through the model
    2. Computes loss and accuracy
    3. Updates model parameters via backpropagation
    4. Adjusts learning rate using scheduler
    
    Args:
        model (BasicCNNClassifier): The neural network model being trained
        data_loader (DataLoader): Provides batched training data from HDF5Dataset
        optimizer (torch.optim.Optimizer): Updates model parameters
        scheduler (torch.optim.lr_scheduler._LRScheduler): Adjusts learning rate
        device (torch.device): Device (CPU/GPU) to run computations on
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy) averaged over all batches
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_acc = 0.0
    
    # Process each batch in the training data
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        
        outputs = model(inputs)  # Forward pass through the model
        
        # Calculate accuracy (no gradient needed for metrics)
        with torch.no_grad():
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
            
        # Compute loss and update model
        loss = F.mse_loss(outputs, labels)  # Mean squared error loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        scheduler.step()  # Adjust learning rate
        
        running_loss += loss.item()
    
    # Return average loss and accuracy for the epoch
    return running_loss / len(data_loader), running_acc / len(data_loader)


def validate_one_epoch(model, data_loader, device):
    """Evaluate model performance on validation data.
    
    Processes all validation batches with gradient computation disabled:
    1. Forwards data through the model
    2. Computes accuracy metrics
    3. Does not update model parameters
    
    Args:
        model (BasicCNNClassifier): The neural network model to evaluate
        data_loader (DataLoader): Provides batched validation data
        device (torch.device): Device (CPU/GPU) to run computations on
        
    Returns:
        float: Average validation accuracy for the epoch
    """
    running_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation for validation
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
    """Generate and save predictions for test data.
    
    Args:
        model (BasicCNNClassifier): Trained model to use for predictions
        data_loader (DataLoader): Provides batched test data
        device (torch.device): Device (CPU/GPU) to run computations on
        
    Note:
        Predictions are saved to 'outputs/predictions.txt'
        with one prediction per line
    """
    pass  # TO BE FILLED IN BY APPLICANT


if __name__ == "__main__":
    # Training hyperparameters
    epochs = 100          # Total number of training epochs
    batch_size = 128      # Number of samples per training batch
    learning_rate = 1e-3  # Initial learning rate for optimizer
    
    # Model configuration parameters
    input_dim = 2         # Number of input channels (density and recording_date)
    num_classes = 1       # Binary classification task
    input_shape = 28      # Size of input images (28x28)
    proj_dim = 32        # Projection dimension for CNN layers
    mlp_dim = 128        # Hidden dimension for classifier head
    dropout = 0.2        # Dropout probability for regularization
    weight_decay = 1e-3  # L2 regularization strength
    
    # Setup compute device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize data loaders
    # Training data with larger batch size for efficiency
    train_dataset = HDF5Dataset("data/train.hdf5")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
    
    # Validation data with smaller batch size for more frequent evaluation
    val_dataset = HDF5Dataset("data/valid.hdf5")
    val_loader = DataLoader(val_dataset, batch_size=4, drop_last=True, num_workers=2)
    
    # Initialize model, optimizer, and learning rate scheduler
    model = BasicCNNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        input_shape=input_shape,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,    # Initial learning rate multiplier
        end_factor=1e-5,   # Final learning rate multiplier
        total_iters=len(train_loader) * epochs  # Total number of scheduler steps
    )
    
    # Training loop
    for epoch in range(epochs):
        # Train for one epoch and get metrics
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        # Evaluate on validation set
        val_loss = validate_one_epoch(model, val_loader, device)
        # Report progress
        print(f"Epoch {epoch} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss}")
    
    # Generate predictions on test set
    write_test_predictions(model, test_loader, device)
    print("Test predictions written to outputs/predictions.txt")