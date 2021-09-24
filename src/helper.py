# Utility
import numpy as np
from numpy import random


def normalize(x):
    """Normalizes the given array."""
    return x / max(x)


def relu(x) -> np.ndarray:
    # do not use max(0.0, x)
    # as this is returning 0.0 instead of a numpy array
    return max(np.array([0.0]), x)


def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small
        # denominator can't be zero because it's 1+z
        z = np.exp(x)
        return z / (1 + z)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def random_float(low, high):
    return random.random()*(high-low) + low


def encode_one_hot(x, size):
    return np.eye(size)[x]
