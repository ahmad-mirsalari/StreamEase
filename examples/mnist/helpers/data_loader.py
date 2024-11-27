
from keras.datasets import mnist

import os
import sys
# Add the parent directory of mypackage to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from  .preprocessing import *
# from pre_process import *


def load_mnist_data(row=28, column=28, transpose = False):
    # Since we only need images from the dataset to encode and decode, we
    # won't use the labels.
    (train_data, _), (test_data, _) = mnist.load_data()

    # Normalize and reshape the data
    train_data = preprocess(train_data, row, column, transpose=transpose)
    test_data = preprocess(test_data, row, column, transpose=transpose)

    # Create a copy of the data with added noise
    noisy_train_data = noise(train_data)
    noisy_test_data = noise(test_data)

    # Display the train data and a version of it with added noise
    display(train_data[:10], noisy_train_data[:10])

    return train_data, test_data, noisy_train_data, noisy_test_data