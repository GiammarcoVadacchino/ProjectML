import numpy as np
import kernels

class LogisticRegression:
    def __init__(self, lr=0.001, epochs=1000, use_rff=False, D=100, gamma=1.0):
        self.lr = lr #learning rate 
        self.epochs = epochs # training epochs 
        self.use_rff = use_rff # true if we use RFF, else false
        self.D = D # finite dimension of the space created with RFF
        self.gamma = gamma # controls the scale of similarity and the complexity of decision boundary #TODO: need more info

    def sigmoid(self, z):
        # apply sigmoid in order to have a probability output
        return 1 / (1 + np.exp(-z))


    def _init_random_features(self, d):
        #initialize weight sampling from a gaussian distribution (cause i use gaussian kernel) W sampled from N(0,2*gamma)
        # np.sqrt(2 * self.gamma) scaling factor, np.random.randn(self.D, d) sampling from normal distribution
        self.W = np.sqrt(2 * self.gamma) * np.random.randn(self.D, d)
        # initialize bias from a uniform distribution, b sampled from (0,2pi)
        # 2 * np.pi scaling factorm, np.random.rand(self.D) sampling from a uniform distribution
        self.b_rf = 2 * np.pi * np.random.rand(self.D)

    def _transform(self, X):
        if not self.use_rff:
            #if don't use RFF don't transorm input
            return X

        # compute: W^T * X + b
        projection = X @ self.W.T + self.b_rf
        # compute z(x) = sqrt(2/D) * cos(W^T*x + b)
        Z = np.sqrt(2 / self.D) * np.cos(projection)
        return Z

    def fit(self, X, y):
        
        # n is the number of samples, d is the number of features
        n, d = X.shape

        # init RFF if necessary
        if self.use_rff:
            self._init_random_features(d)

        # transform input (only if use_rff if True)
        Z = self._transform(X)
        n, d_transformed = Z.shape

        # initialize weihts and bias for the linear model
        self.w = np.zeros(d_transformed)
        self.b = 0

        # training loop
        for _ in range(self.epochs):
            # compute the prediction: z = Z * W + b
            # NOTE: Z corresponds to X if use_rff is False, otherwise Z are the inputs obtained using RFF tranformation
            z = Z @ self.w + self.b
            p = self.sigmoid(z)

            # compute gradient: 1/n * Z^{T}(p - y)
            # p is the predicted probability, y is the true label, p - y corresponds to the error vector
            # in this way i compute how much each feature contribute to the error
            grad_w = Z.T @ (p - y) / n
            # average error across al samples
            grad_b = np.mean(p - y)

            # update rule for the weights
            self.w -= self.lr * grad_w
            # update rule for the bias
            self.b -= self.lr * grad_b

    # ------------------ PREDICT ------------------
    def predict(self, X):
        Z = self._transform(X)
        # threshold: if the probability is greater than 0.5 the output is 1, otherwise is 0.
        return (self.sigmoid(Z @ self.w + self.b) > 0.5).astype(int)
    

#TODO: fix this, got 0% accuracy, there is some problem, maybe implement a differnte "version" of SVM with rbf
class KernelSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1.0, gamma=1.0):
        self.lr = lr # learning rate
        self.epochs = epochs # epochs in traning
        # balance the trade off between maximize the margin and minimize classification error
        # objectivde function: min 1/2*(||w||)^2 + C sum(slack_variables)
        self.C = C 
        self.gamma = gamma # parameter of the gaussian kernel

    def fit(self, X, y):
        # SVM assume lables in {-1,1}
        y = np.where(y == 0, -1, 1)

        self.X = X
        self.y = y

        # get number of sampeles
        n = X.shape[0]

        # compute Gnam Matrix, so matrix n x n, with all entries K = K(x_i,x_j)
        #NOTE: doesn't scale with dataset size, O(n^2), need lot of storage memory
        K = kernels.rbf_kernel(X, X, self.gamma) 


        # compute this matrix for SVM dual objective
        # dual objective function: max_a = sum(a_i) - 1/2 * sum(a_i * a_j * Q), where Q = y_i * y_i * K(x_i,x_j)
        Q = np.outer(y, y) * K
        # init alpha
        self.alpha = np.zeros(n)

        # training loop
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


        sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        if np.any(sv):
            i = np.where(sv)[0][0]
            self.b = y[i] - np.sum(self.alpha * y * K[:, i])
        else:
            self.b = 0

    def decision_function(self, X):
        # f(x) = Σ alpha_i y_i k(x, x_i) + b
        K = kernels.rbf_kernel(X, self.X, self.gamma)
        return K @ (self.alpha * self.y) + self.b

    def predict(self, X):
        pred = self.decision_function(X)
        return (pred > 0).astype(int)