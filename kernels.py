import numpy as np


def rbf_kernel(X1, X2, gamma):
    sq1 = np.sum(X1**2, axis=1).reshape(-1,1)
    sq2 = np.sum(X2**2, axis=1).reshape(1,-1)

    dist = sq1 + sq2 - 2 * X1 @ X2.T
    return np.exp(-gamma * dist)


class RFF:
    def __init__(self, D, sigma=1.0):
        self.D = D
        self.sigma = sigma

    def fit(self, X):
        d = X.shape[1]
        self.W = np.random.normal(0, 1/self.sigma, size=(d, self.D))
        self.b = np.random.uniform(0, 2*np.pi, size=self.D)

    def transform(self, X):
        return np.sqrt(2/self.D) * np.cos(X @ self.W + self.b)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
