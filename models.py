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
    


class KernelSVM:
    def __init__(self, lr=1e-3, epochs=1000, C=1.0, gamma=1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        # convert labels {0,1} → {-1,1}
        y = np.where(y == 0, -1, 1)

        self.X = X
        self.y = y

        n = X.shape[0]

        # kernel matrix (n x n)
        K = rbf_kernel(X, X, self.gamma)

        # Q = y_i y_j K_ij
        Q = np.outer(y, y) * K

        # inizializzazione alpha
        self.alpha = np.zeros(n)

        # -----------------------------
        # TRAINING (gradient ascent duale)
        # -----------------------------
        for _ in range(self.epochs):
            # gradiente: 1 - Q alpha
            grad = 1 - Q @ self.alpha

            # aggiornamento
            self.alpha += self.lr * grad

            # vincolo: 0 ≤ alpha ≤ C
            self.alpha = np.clip(self.alpha, 0, self.C)

            # vincolo: Σ alpha_i y_i = 0
            correction = np.dot(self.alpha, y) / np.sum(y**2)
            self.alpha -= correction * y

        # -----------------------------
        # calcolo bias
        # -----------------------------
        sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        if np.any(sv):
            i = np.where(sv)[0][0]
            self.b = y[i] - np.sum(self.alpha * y * K[:, i])
        else:
            self.b = 0

    def decision_function(self, X):
        # f(x) = Σ alpha_i y_i k(x, x_i) + b
        K = rbf_kernel(X, self.X, self.gamma)
        return K @ (self.alpha * self.y) + self.b

    def predict(self, X):
        pred = self.decision_function(X)
        return (pred > 0).astype(int)