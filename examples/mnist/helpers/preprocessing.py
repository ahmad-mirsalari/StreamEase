import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def preprocess(array, row=28, column=28, transpose= False):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    array = array.astype("float32") / 255.0

    # Calculate padding values based on row and column parameters
    pad_left = max(0, (row - 28) // 2)
    pad_right = max(0, row - 28 - pad_left)
    pad_top = max(0, (column - 28) // 2)
    pad_bottom = max(0, column - 28 - pad_top)

    # Pad the images
    array = np.pad(array, ((0, 0), (pad_left, pad_right), (pad_top, pad_bottom)), mode='constant', constant_values=0)
    if transpose:
        array = np.transpose(array, (0, 2,1))

    return array

def torch_data_loader( data, batch_size):
    # Convert NumPy arrays to PyTorch tensors
    data_tensor = torch.from_numpy(data)  # Add channel dimension
    
    # Create DataLoader for easier batching
    data_dataset = TensorDataset(data_tensor)

    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader

def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    n = len(array1)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(array1[i])  # Display the padded image
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(array2[i])  # Display the padded image with noise
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()