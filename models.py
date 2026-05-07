import numpy as np
import kernels


class RandomFourierFeatures:
    def __init__(self, D=100, gamma=1.0):
        self.D = D #number of finite dimensions
        self.gamma = gamma #Parameter of the RBF kernel, control the velocity of decrase of similarity among points

        #RBF kernels, points closer in the original space are similar in the new transformated space

    def fit(self, X):
        #get number of fetaures
        d = X.shape[1]

        #Build the W matrix, where each row is a random dircetion in the space
        #the number are sampled from a normal distribution, and the multiply them in order to approximate the rbf kernel
        self.W = np.sqrt(2 * self.gamma) * np.random.randn(self.D, d)

        #generate random bias from 0 to 2pi
        self.b = 2 * np.pi * np.random.rand(self.D)

    def transform(self, X):
        #transform original data in the Random Fourier space, each column represents a random projection of the original data
        projection = X @ self.W.T + self.b
        # z(x) = sqrt(2/D) * cos(Wx + b) is the formula for the random fourier featuers, np.sqrt(2.0 / self.D) this is a normalization factor
        return np.sqrt(2.0 / self.D) * np.cos(projection)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RandomBinningFeatures:
    def __init__(self, D=100, gamma=1.0, buckets=8):
        self.D = D #number of random partitions generated
        self.gamma = gamma # controls the mean dimensions of the cell
        self.buckets = buckets # number of hashed buckets for each cells, need to control the final dimensions

    def fit(self, X):
        #number of initial features
        d = X.shape[1]

        #Build a matrix (D,d), every row represent a different random grid, with random numbers genrated from an exponential distribution.
        #Scaled by 1/gamma
        #If gamma increase, delta becames smaller the cells are smaller and so we have smaller partitions
        #If gamma deacres, delta becames greater the cells are bigger and so we have bigger partitions
        self.delta = np.random.exponential(scale=1.0 / self.gamma,size=(self.D, d))
        #In order to avoid division by 0 or really small values
        self.delta = np.maximum(self.delta, 1e-8)
        #Generate a random offset for each cells in order to random move the cells in the space, in this way they are not aligned with the same input
        self.offset = np.random.uniform(0, self.delta)
        #Generate random weights, need to create an hash function that converts coordinates into a single discrete index
        self.hash_weights = np.random.randint(1,10_000,size=(self.D, d))

    def transform(self, X):
        #number of samples
        n = X.shape[0]
        #final matrix of features
        Z = np.zeros((n, self.D * self.buckets))

        #iterates on the random grids
        for j in range(self.D):
            #calculate the bins were the points fell
            bins = np.floor((X + self.offset[j]) / self.delta[j]).astype(int)
            #convert the coordinates into a single hashed bucket
            hashed = np.abs(bins @ self.hash_weights[j]) % self.buckets

            #each sample activate ONE feature for each grid
            rows = np.arange(n)
            cols = j * self.buckets + hashed

            #one hot encoding, set to 1 the activated feature
            Z[rows, cols] = 1.0

        #normalizations
        return Z / np.sqrt(self.D)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Perceptron:
    def __init__(self, lr=0.001, epochs=1000, feature_map="linear", D=100, gamma=1.0, buckets=32):

        self.lr = lr #learning rate
        self.epochs = epochs #epochs of training
        self.feature_map = feature_map #select the feature mapping
        self.D = D #number of finite dimensione if the mapping is not linear
        self.gamma = gamma #parameter if the mapping is not linear
        self.buckets = buckets #number of buckets if the mapping is not linear

        self.mapper = None #save the mapping to use
        self.w = None #weights
        self.b = None #bias

    def _init_feature_map(self, X):

        #initialize and apply the transormation choosen on the training set

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
        if self.mapper is None:
            return X

        return self.mapper.transform(X)

    def fit(self, X, y):
        #number of samples
        n_samples = X.shape[0]
        #apply the choosen mapping
        X_train = self._init_feature_map(X)
        #number of features
        n_features = X_train.shape[1]

        #init bias and weights
        self.w = np.zeros(n_features)
        self.b = 0.0

        #convert label
        y_signed = np.where(y == 0, -1, 1)

        #iterate over epochs
        for _ in range(self.epochs):
            #permute the indices at each iteration
            indices = np.random.permutation(n_samples)

            #iterate over the samples
            for i in indices:
                #calculate the score = w^T x_i + b
                score = X_train[i] @ self.w + self.b
                #calculate the margin, if is positive the sample is correctly classified, if it is <= 0 the point is misclassified or on the decision boundaries
                margin = y_signed[i] * score
                
                if margin <= 0:
                    #update parameters if the point is misclassified
                    self.w += self.lr * y_signed[i] * X_train[i]
                    self.b += self.lr * y_signed[i]

        return self

    def decision_function(self, X):
        #calculate the score of the model f(x) = w^T phi(x) + b, phi(x) is the choosen transformation
        X_transformed = self._transform(X)
        return X_transformed @ self.w + self.b

    def predict(self, X):
        #if scores is grater or equal to 0 it predicts 1, otherwise 0
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)


class KernelSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1.0, gamma=1.0, kernel="rbf"):

        self.lr = lr #learning rate
        self.epochs = epochs #train epochs
        self.C = C #regularization paremeter, if it is high, penalize more the error, if it is low penaize less the error
        self.gamma = gamma #parameter of the kernel, control how much fast deacrese the similarity between points
        self.kernel = kernel #selected kernel

        self.alpha = None #coefficients of the dual SVM, a coefficient for each sample
        self.b = 0.0 #bias 
        self.X_train = None #train set sample
        self.y_train = None #train set label

    def _kernel_function(self, X1, X2):
        
        #Kernel RBF/Gaussiano: K(x,z) = exp(-gamma * ||x - z||^2)
        if self.kernel == "rbf":
            return kernels.rbf_kernel(X1, X2, gamma=self.gamma)

        #Kernel Laplaciano: K(x,z) = exp(-gamma * ||x - z||_1)
        if self.kernel == "laplace":
            return kernels.laplace_kernel(X1, X2, gamma=self.gamma)

        raise ValueError("kernel deve essere 'rbf' oppure 'laplace'")

    def fit(self, X, y):
        #number of samples
        n_samples = X.shape[0]
        #convert the label
        y_signed = np.where(y == 0, -1, 1)

        self.X_train = X
        self.y_train = y_signed

        #calcolate the gram matrix, containing the transfromations of the kernell, K[i, j] = K(x_i, x_j)
        #contain all the similarity between points
        K = self._kernel_function(X, X)

        #matrix: Q_ij = y_i * y_j * K(x_i, x_j)
        Q = np.outer(y_signed, y_signed) * K

        #set coefficients for each samples
        self.alpha = np.zeros(n_samples)

        #training epochs
        #maximize this: sum_i alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)
        for _ in range(self.epochs):
            #Gradient w.r.t alpha, if is positive alpha tends to get higher, otherwise not
            grad = 1 - Q @ self.alpha
            #upgrade coefficients with gradient ascent
            self.alpha += self.lr * grad
            #force each alpha between 0 and C
            self.alpha = np.clip(self.alpha, 0, self.C)

            #correction in order to have that sum_i alpha_i y_i = 0
            correction = np.dot(self.alpha, y_signed) / np.sum(y_signed ** 2)
            self.alpha -= correction * y_signed

            #after the update some alphas may be violated the constraints, so we have to force again
            self.alpha = np.clip(self.alpha, 0, self.C)

        #identify the support vectors using point with a_i between 0 and C
        support_vector_mask = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        
        
        #for any support vector we have y_i = sum_j alpha_j y_j K(x_j, x_i) + b and so b = y_i - sum_j alpha_j y_j K(x_j, x_i)
        if np.any(support_vector_mask):
            #indices of support vectors
            support_indices = np.where(support_vector_mask)[0]
            b_values = []


            for i in support_indices:
                #decision function withoout considering the bias
                decision_without_b = np.sum(self.alpha * y_signed * K[:, i])
                #value of b using support vector i
                b_values.append(y_signed[i] - decision_without_b)

            #mean of the biases
            self.b = np.mean(b_values)
        else:
            self.b = 0.0

        return self

    def decision_function(self, X):


        #calcuate kernel matrix
        K = self._kernel_function(X, self.X_train)
        #each prediction use all the training points weighetd by alpha_i * y_i
        #in practice we use the support vectors beacuse alphas close to zero have a small impact
        return K @ (self.alpha * self.y_train) + self.b

    def predict(self, X):
        
        #calculate the scores
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0) #if the score >= 0 then the prediction is 1, else 0