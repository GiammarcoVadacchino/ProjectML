import numpy as np
import matplotlib.pyplot as plt
import time
import dataset 
import models
import kernels
from sklearn.datasets import fetch_covtype



def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_experiment(D_values):
    data = fetch_covtype()
    X, y = data.data, data.target 

    """

    500K samples: 

    - Linear model: acc = 0.309  time = 26s
    - SVM rbf kernel: acc = 0.364 time = 689s
    - Linear model with RFF: 
        - D = 10: acc = 0.364??  time = 6s
        - D = 50: acc = ??  time = 26s
        - D = 100: acc = ??  time = 42s
        - D = 500: acc = ??  time = 280s
        - D = 1000: acc = ??  time = 887s


    400K samples:

    - Linear model: acc = 0.309  time = 13s
    - SVM rbf kernel: acc = 0.364 time = 439s
    - Linear model with RFF: 
        - D = 10: acc = 0.364??  time = 4s
        - D = 50: acc = ??  time = 16s
        - D = 100: acc = ??  time = 31s
        - D = 500: acc = ??  time = 146s
        - D = 1000: acc = ??  time = 271s
    
    """

    idx = np.random.choice(len(X),10_000, replace=False)
    X, y = X[idx], y[idx]
    
    print(f"Input size: {X.shape}\n Output size: {y.shape}") 
    print(f"X : {X}\n y: {y}") 
    X_train, X_test, y_train, y_test = dataset.train_test_split(X, y)
    X_train, X_test = dataset.standardize(X_train, X_test)

    t0 = time.time()
    lin = models.LogisticRegression()
    lin.fit(X_train, y_train)
    t1 = time.time()

    lin_acc = accuracy(y_test, lin.predict(X_test))
    print("Linear:", lin_acc, "time:", t1-t0)

    t0 = time.time()
    ker = models.KernelSVM()
    ker.fit(X_train, y_train)
    t1 = time.time()

    ker_acc = accuracy(y_test, ker.predict(X_test))
    print("Kernel:", ker_acc, "time:", t1-t0)

    results = []

    for dimension in D_values:
        t0 = time.time()

        #rff = kernels.RFF(D)
        #Z_train = rff.fit_transform(X_train)
        #Z_test = rff.transform(X_test)

        model = models.LogisticRegression(use_rff=True, D=dimension)
        model.fit(X_train, y_train)

        t1 = time.time()

        acc = accuracy(y_test, model.predict(X_test))

        results.append((dimension, acc, t1-t0))
        print("RFF D=", dimension, "acc=", acc, "time=", t1-t0)

    return results, lin_acc, ker_acc



def plot_results(results, lin_acc, ker_acc):
    D = [r[0] for r in results]
    acc = [r[1] for r in results]
    time_vals = [r[2] for r in results]

    plt.figure()
    plt.plot(D, acc, marker='o', label="RFF")
    plt.axhline(lin_acc, linestyle="--", label="Linear")
    plt.axhline(ker_acc, linestyle="--", label="Kernel")

    plt.xscale("log")
    plt.xlabel("D")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs Random Features")
    plt.show()

    plt.figure()
    plt.plot(D, time_vals, marker='o')
    plt.xscale("log")
    plt.xlabel("D")
    plt.ylabel("Time")
    plt.title("Runtime vs D")
    plt.show()



if __name__ == "__main__":
    D_values = [10, 50, 100, 500, 1000]

    results, lin_acc, ker_acc = run_experiment(D_values)
    plot_results(results, lin_acc, ker_acc)