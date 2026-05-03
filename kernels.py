import numpy as np


def rbf_kernel(X1, X2, gamma):
    """
    RBF kernel:
        K(x, y) = exp(-gamma ||x - y||^2)
    """
    sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dist = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * dist)


def laplace_kernel(X1, X2, gamma):
    """
    Laplace kernel:
        K(x, y) = exp(-gamma ||x - y||_1)
    """
    dist = np.sum(np.abs(X1[:, None] - X2[None, :]), axis=2)
    return np.exp(-gamma * dist)