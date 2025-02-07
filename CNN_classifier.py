"""
CNN_classifier.py - Neural Network Model Architecture

This file defines the core neural network architecture used in the binary classification task.
It serves as the model component in the project's deep learning pipeline.

Project Structure:
- CNN_classifier.py (this file): Defines the neural network model architecture
- dataset.py: Handles data loading and preprocessing from HDF5 files
- run.py: Orchestrates the training process and ties everything together

The BasicCNNClassifier defined here is instantiated and trained in run.py using data
loaded through the HDF5Dataset class from dataset.py. The model processes 28x28 images
with 2 channels (density and recording_date) to make binary predictions.

Dependencies:
- torch: Main deep learning framework
- torch.nn: Neural network layers and utilities
- torch.nn.functional: Activation functions and operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNNClassifier(torch.nn.Module):
    """A Convolutional Neural Network (CNN) classifier for image processing tasks.
    
    This CNN architecture consists of two convolutional blocks followed by a fully connected
    classifier head. Each convolutional block includes batch normalization and dropout for
    regularization. The network is designed to process square input images and output
    class predictions.
    
    Architecture:
        - Conv Block 1: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - Conv Block 2: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - Classifier: Flatten -> Linear -> ReLU -> Dropout -> Linear
    
    Args:
        input_dim (int): Number of input channels in the image (default: 2)
        proj_dim (int): Base projection dimension for conv layers (default: 16)
        mlp_dim (int): Hidden dimension size for the MLP classifier head (default: 128)
        num_classes (int): Number of output classes (default: 1 for binary classification)
        input_shape (int): Size of the input image (assumes square input) (default: 28)
        dropout (float): Dropout probability for regularization (default: 0.1)
    """
    def __init__(self, 
                input_dim:int=2,
                proj_dim:int=16,
                mlp_dim:int=128,
                num_classes:int=1,
                input_shape:int=28,
                dropout:float=0.1):
        super(BasicCNNClassifier, self).__init__()
        self.input_shape = input_shape
        self.proj_dim = proj_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # First convolutional block
        # Input: (batch_size, input_dim, input_shape, input_shape)
        # Output: (batch_size, proj_dim, input_shape/2, input_shape/2)
        self.conv1 = nn.Conv2d(input_dim, proj_dim, kernel_size=3, padding=1,)
        self.bn1 = nn.BatchNorm2d(proj_dim)
        self.dropout1 = nn.Dropout2d(p=dropout)
        
        # Second convolutional block
        # Input: (batch_size, proj_dim, input_shape/2, input_shape/2)
        # Output: (batch_size, proj_dim*2, input_shape/4, input_shape/4)
        self.conv2 = nn.Conv2d(proj_dim, proj_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(proj_dim*2)
        self.dropout2 = nn.Dropout2d(p=dropout)
        
        # MLP classifier head
        self.flatten = nn.Flatten()
        # Calculate input features for the linear layer based on the conv output shape
        # After 2 max pooling layers, spatial dimensions are reduced by factor of 4
        self.fc1 = nn.Linear(proj_dim*2*(input_shape//4)**2, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, num_classes)
        
    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim, input_shape, input_shape)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)          # Apply convolution
        x = self.bn1(x)           # Normalize activations
        x = F.relu(x)             # Apply non-linearity
        x = F.max_pool2d(x, 2)    # Reduce spatial dimensions
        x = self.dropout1(x)      # Apply dropout for regularization
        
        # Second conv block
        x = self.conv2(x)          # Apply convolution
        x = self.bn2(x)           # Normalize activations
        x = F.relu(x)             # Apply non-linearity
        x = F.max_pool2d(x, 2)    # Reduce spatial dimensions
        x = self.dropout2(x)      # Apply dropout for regularization
        
        # MLP classifier
        x = self.flatten(x)       # Flatten for fully connected layers
        x = F.relu(self.fc1(x))   # First fully connected layer with ReLU
        x = F.dropout(x, p=0.5, training=self.training)  # Additional dropout
        x = self.fc2(x)          # Final classification layer
        
        return x