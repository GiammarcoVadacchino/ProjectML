import numpy as np



class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self.sigmoid(z)

            grad_w = X.T @ (p - y) / n
            grad_b = np.mean(p - y)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        return (self.sigmoid(X @ self.w + self.b) > 0.5).astype(int)
    


class KernelRidge:
    def __init__(self, gamma=1.0, lam=1e-3):
        self.gamma = gamma
        self.lam = lam

    def fit(self, X, y):
        self.X = X
        K = rbf_kernel(X, X, self.gamma)

        n = len(K)
        self.alpha = np.linalg.solve(K + self.lam*np.eye(n), y)

    def predict(self, X):
        K = rbf_kernel(X, self.X, self.gamma)
        return (K @ self.alpha > 0.5).astype(int)
