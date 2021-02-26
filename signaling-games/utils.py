import numpy as np


def rand_argmax(x, axis=1):
    """
    Find the argmax along axis of an array by randomly breaking ties
    Args:
        x (np.ndarray): Matrix whose argmax needs to be found
        axis (int): Axis along which to find the argmax (default: 1)

    Returns:
        Argmax along axis
    """
    tie_break_x = np.random.random(x.shape) * (x == x.max(axis=axis, keepdims=True))
    return np.argmax(tie_break_x, axis=axis)


def vectorized_2d_choice(items, p, axis=1):
    """
    Similar to np.random.choice but p can be 2D to generate a choice for each row
    Args:
        items (np.ndarray): 1D array consisting of elements to choose from
        p (np.ndarray): Probability matrix. The axis given is considered to be the
            probabilities
        axis (int): Axis of p which should be considered as probabilities (default: 1)

    Returns:

    """
    if axis == 1:
        p = p.T
    s = p.cumsum(axis=0)
    r = np.random.rand(p.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]


def softmax(x, axis=0):
    """
    Softmax over an axis
    Args:
        x (np.ndarray): Input array
        axis (int): Axis over which softmax

    Returns:

    """

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def onehot(x, n_classes):
    """
    Onehot encoding
    Args:
        x (np.ndarray): Array with values
        n_classes (int): Total number of classes

    Returns:
        Onehot encoded array
    """
    x = x.flatten()
    x_oh = np.zeros((x.size, n_classes))
    x_oh[np.arange(x.size), x] = 1
    return x_oh
