import numpy as np


def make_moons(n_samples=1000, noise=0.1):
    n = n_samples // 2

    theta = np.random.rand(n) * np.pi
    x1 = np.c_[np.cos(theta), np.sin(theta)]

    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta) - 0.5]

    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)])

    X += noise * np.random.randn(*X.shape)

    return X, y


def train_test_split(X, y, test_ratio=0.3):
    n = len(X)
    idx = np.random.permutation(n)

    test_size = int(n * test_ratio)

    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return (X_train - mean)/std, (X_test - mean)/std
