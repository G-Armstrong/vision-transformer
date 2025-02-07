"""
dataset.py - Data Loading and Processing

This file handles the loading and preprocessing of HDF5 data files for the deep learning pipeline.
It provides a PyTorch Dataset implementation that efficiently manages data access patterns
and memory usage.

Project Structure:
- dataset.py (this file): Handles data loading and preprocessing from HDF5 files
- CNN_classifier.py: Defines the neural network model that processes this data
- run.py: Uses this dataset class to feed data to the model during training

The HDF5Dataset class defined here serves as the data backbone of the project, providing
efficient access to the training, validation, and test datasets. It supports both
memory-efficient streaming and full data loading modes, feeding the image data to the
CNN model defined in CNN_classifier.py through DataLoaders configured in run.py.

Data Structure:
The HDF5 files are expected to contain 28x28 images with two channels:
- density: First channel representing density measurements
- recording_date: Second channel containing temporal information
- labels: Binary classification targets

Dependencies:
- h5py: For reading HDF5 files
- numpy: For efficient array operations
- torch.utils.data: For Dataset implementation
"""

import h5py
import numpy as np
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Custom Dataset class for loading data from HDF5 files.
    
    This dataset class is designed to handle HDF5 files containing image data
    with density and recording_date channels, along with corresponding labels.
    It supports both memory-efficient streaming and full data loading into memory.
    
    Args:
        path (str): Path to the HDF5 file
        load_into_memory (bool): If True, loads entire dataset into memory during
            initialization. If False, streams data from disk as needed. (default: True)
    
    Dataset Structure:
    The HDF5 files are expected to contain 28x28 images with two channels:
    - density: First channel representing density measurements
    - recording_date: Second channel containing experimental conditions, may indicate whether a measurement was taken with or without the new procedure
    - labels: Binary classification targets
    """
    def __init__(self, path, load_into_memory=True):
        self.path = path
        self.data = h5py.File(path, "r")
        self.load_into_memory = load_into_memory
        
        if load_into_memory:
            # Load only the density channel
            self.inputs = np.expand_dims(self.data["density"][:], axis=1)  # Add channel dimension
            self.labels = self.data["labels"][:]

    def __len__(self):
        """Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        if self.load_into_memory:
            return self.inputs.shape[0]
        else:
            return self.data["density"].shape[0]

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_data, label) where:
                - input_data is a numpy array of shape (2, H, W) containing
                  the density and recording_date channels
                - label is a numpy array containing the corresponding label
        """
        if self.load_into_memory:
            # Return pre-loaded data
            return np.float32(self.inputs[idx]), np.float32(self.labels[idx])
        else:
            # Load only density data from disk on-the-fly
            x = np.expand_dims(self.data["density"][idx], axis=0)  # Add channel dimension
            return np.float32(x), np.float32(self.data["labels"][idx])