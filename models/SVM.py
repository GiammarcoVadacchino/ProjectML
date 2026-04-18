import numpy as np



class LinearSVM:
    def __init__(self, lr=0.01, epochs=1000, C=1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C

    def fit(self, X, y):
        # convert labels {0,1} → {-1,1}
        y = np.where(y == 0, -1, 1)

        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(n):
                condition = y[i] * (np.dot(X[i], self.w) + self.b)

                if condition >= 1:
                    grad_w = self.w
                    grad_b = 0
                else:
                    grad_w = self.w - self.C * y[i] * X[i]
                    grad_b = -self.C * y[i]

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X):
        pred = np.dot(X, self.w) + self.b
        return (pred > 0).astype(int)