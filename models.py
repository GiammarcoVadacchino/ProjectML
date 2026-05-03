import numpy as np
import kernels


class RandomFourierFeatures:
    def __init__(self, D=100, gamma=1.0):
        self.D = D
        self.gamma = gamma

    def fit(self, X):
        d = X.shape[1]
        self.W = np.sqrt(2 * self.gamma) * np.random.randn(self.D, d)
        self.b = 2 * np.pi * np.random.rand(self.D)

    def transform(self, X):
        projection = X @ self.W.T + self.b
        return np.sqrt(2.0 / self.D) * np.cos(projection)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RandomBinningFeatures:
    def __init__(self, D=100, gamma=1.0, buckets=8):
        self.D = D
        self.gamma = gamma
        self.buckets = buckets

    def fit(self, X):
        d = X.shape[1]

        self.delta = np.random.exponential(scale=1.0 / self.gamma,size=(self.D, d))
        self.delta = np.maximum(self.delta, 1e-8)

        self.offset = np.random.uniform(0, self.delta)

        self.hash_weights = np.random.randint(1,10_000,size=(self.D, d))

    def transform(self, X):
        n = X.shape[0]
        Z = np.zeros((n, self.D * self.buckets))

        for j in range(self.D):
            bins = np.floor((X + self.offset[j]) / self.delta[j]).astype(int)
            hashed = np.abs(bins @ self.hash_weights[j]) % self.buckets

            rows = np.arange(n)
            cols = j * self.buckets + hashed

            Z[rows, cols] = 1.0

        return Z / np.sqrt(self.D)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Perceptron:
    def __init__(self, lr=0.001, epochs=1000, feature_map="linear", D=100, gamma=1.0, buckets=32):
        """
        Perceptron binario from scratch.

        feature_map:
        - "linear"          -> Perceptron lineare classico
        - "rff"             -> Perceptron con Random Fourier Features
        - "random_binning"  -> Perceptron con Random Binning Features
        """

        self.lr = lr
        self.epochs = epochs
        self.feature_map = feature_map
        self.D = D
        self.gamma = gamma
        self.buckets = buckets

        self.mapper = None
        self.w = None
        self.b = None

    def _init_feature_map(self, X):
        """
        Inizializza e applica la trasformazione scelta sul training set.
        """

        if self.feature_map == "linear":
            self.mapper = None
            return X

        if self.feature_map == "rff":
            self.mapper = RandomFourierFeatures(D=self.D, gamma=self.gamma)
            return self.mapper.fit_transform(X)

        if self.feature_map == "random_binning":
            self.mapper = RandomBinningFeatures(D=self.D, gamma=self.gamma, buckets=self.buckets)
            return self.mapper.fit_transform(X)

        raise ValueError("feature_map deve essere: 'linear', 'rff' oppure 'random_binning'")

    def _transform(self, X):
        """
        Applica al test set la stessa trasformazione imparata sul training set.

        Importante:
        - sul training set si usa fit_transform()
        - sul test set si usa solo transform()

        Così le random features restano le stesse.
        """

        if self.mapper is None:
            return X

        return self.mapper.transform(X)

    def fit(self, X, y):
        """
        Allena il Perceptron.

        Il training avviene sempre in modo lineare, ma lo spazio cambia:

        - linear:
            f(x) = w^T x + b

        - rff:
            f(x) = w^T z_RFF(x) + b

        - random_binning:
            f(x) = w^T z_RB(x) + b
        """

        n_samples = X.shape[0]

        X_train = self._init_feature_map(X)

        n_features = X_train.shape[1]

        self.w = np.zeros(n_features)
        self.b = 0.0

        y_signed = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):

            indices = np.random.permutation(n_samples)

            for i in indices:

                score = X_train[i] @ self.w + self.b

                margin = y_signed[i] * score

                if margin <= 0:
                    self.w += self.lr * y_signed[i] * X_train[i]
                    self.b += self.lr * y_signed[i]

        return self

    def decision_function(self, X):
        """
        Calcola lo score del modello:

            f(x) = w^T phi(x) + b

        dove phi(x) può essere:
        - identità
        - Random Fourier Features
        - Random Binning Features
        """

        X_transformed = self._transform(X)
        return X_transformed @ self.w + self.b

    def predict(self, X):
        """
        Predice label in formato {0, 1}.
        """

        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)


class KernelSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1.0, gamma=1.0, kernel="rbf"):
        """
        SVM kernelizzata from scratch usando il problema duale.

        kernel:
        - "rbf"     -> K(x, z) = exp(-gamma * ||x - z||^2)
        - "laplace" -> K(x, z) = exp(-gamma * ||x - z||_1)
        """

        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None

    def _kernel_function(self, X1, X2):
        """
        Calcola la matrice kernel tra X1 e X2.
        """

        if self.kernel == "rbf":
            return kernels.rbf_kernel(X1, X2, gamma=self.gamma)

        if self.kernel == "laplace":
            return kernels.laplace_kernel(X1, X2, gamma=self.gamma)

        raise ValueError("kernel deve essere 'rbf' oppure 'laplace'")

    def fit(self, X, y):
        """
        Allena la SVM kernelizzata.
        """

        n_samples = X.shape[0]

        y_signed = np.where(y == 0, -1, 1)

        self.X_train = X
        self.y_train = y_signed

        K = self._kernel_function(X, X)

        Q = np.outer(y_signed, y_signed) * K

        self.alpha = np.zeros(n_samples)

        for _ in range(self.epochs):

            grad = 1 - Q @ self.alpha

            self.alpha += self.lr * grad

            self.alpha = np.clip(self.alpha, 0, self.C)

            correction = np.dot(self.alpha, y_signed) / np.sum(y_signed ** 2)
            self.alpha -= correction * y_signed

            self.alpha = np.clip(self.alpha, 0, self.C)

        support_vector_mask = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        if np.any(support_vector_mask):
            support_indices = np.where(support_vector_mask)[0]
            b_values = []

            for i in support_indices:
                decision_without_b = np.sum(self.alpha * y_signed * K[:, i])
                b_values.append(y_signed[i] - decision_without_b)

            self.b = np.mean(b_values)
        else:
            self.b = 0.0

        return self

    def decision_function(self, X):
        """
        f(x) = sum_i alpha_i y_i K(x_i, x) + b
        """

        K = self._kernel_function(X, self.X_train)
        return K @ (self.alpha * self.y_train) + self.b

    def predict(self, X):
        """
        Predice label in formato {0, 1}.
        """

        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)