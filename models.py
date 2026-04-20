import numpy as np



class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr # learning rate
        self.epochs = epochs # number of epochs in training

    def sigmoid(self, z):
        # apply sigmoid in order to return a probability
        return 1 / (1 + np.exp(-z)) 

    def fit(self, X, y):
        n, d = X.shape # n is the number of examples, d is the number of features for each example
        self.w = np.zeros(d) # weight initialization (zero vector), NOTE: maybe other weight inizialization are better
        self.b = 0 # bias inizialiation

        # traning 
        for _ in range(self.epochs):
            # compute the prediction y = w * X + b
            z = X @ self.w + self.b 
            # normalize the prediction into a probability
            p = self.sigmoid(z)


            grad_w = X.T @ (p - y) / n
            grad_b = np.mean(p - y)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        return (self.sigmoid(X @ self.w + self.b) > 0.5).astype(int)
    

class RFFLogisticRegression:
    def __init__(self, D=100, gamma=1.0, lr=0.1, epochs=1000):
        """
        Logistic Regression con Random Fourier Features

        Parametri:
        - D: numero di feature random (dimensione spazio trasformato)
        - gamma: parametro kernel RBF
        - lr: learning rate
        - epochs: numero iterazioni
        """
        self.D = D
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def _init_random_features(self, d):
        """
        Inizializza le matrici random:
        W ~ N(0, 2γI)
        b ~ Uniform(0, 2π)
        """

        # matrice (D, d)
        self.W = np.sqrt(2 * self.gamma) * np.random.randn(self.D, d)

        # bias random (D,)
        self.b_rf = 2 * np.pi * np.random.rand(self.D)

    def _transform(self, X):
        """
        Applica Random Fourier Features:
        X -> Z(X)
        """

        # proiezione lineare + shift
        projection = X @ self.W.T + self.b_rf

        # applicazione coseno
        Z = np.sqrt(2 / self.D) * np.cos(projection)

        return Z

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape

        # inizializza RFF
        self._init_random_features(d)

        # trasforma i dati
        Z = self._transform(X)  # (n, D)

        # inizializza pesi nello spazio trasformato
        self.w = np.zeros(self.D)
        self.b = 0

        # training (identico alla logistic standard ma su Z)
        for _ in range(self.epochs):
            z = Z @ self.w + self.b
            p = self.sigmoid(z)

            grad_w = Z.T @ (p - y) / n
            grad_b = np.mean(p - y)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        # trasformazione dei nuovi dati
        Z = self._transform(X)

        # predizione
        return (self.sigmoid(Z @ self.w + self.b) > 0.5).astype(int)
    


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