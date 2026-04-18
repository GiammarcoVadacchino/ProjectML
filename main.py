import numpy as np
import matplotlib.pyplot as plt
import time
import dataset.dataset as data
import models.linear as linear
import models.SVM as svm
import kernels.kernels as kernel


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_experiment(D_values):
    X, y = data.make_moons(n_samples=2000, noise=0.2)
    X_train, X_test, y_train, y_test = data.train_test_split(X, y)
    X_train, X_test = data.standardize(X_train, X_test)

    # ---- Linear ----
    t0 = time.time()
    lin = linear.LogisticRegression()
    lin.fit(X_train, y_train)
    t1 = time.time()

    lin_acc = accuracy(y_test, lin.predict(X_test))
    print("Linear:", lin_acc, "time:", t1-t0)

    # ---- Kernel ----
    t0 = time.time()
    ker = svm.LinearSVM()
    ker.fit(X_train, y_train)
    t1 = time.time()

    ker_acc = accuracy(y_test, ker.predict(X_test))
    print("Kernel:", ker_acc, "time:", t1-t0)

    # ---- RFF ----
    results = []

    for D in D_values:
        t0 = time.time()

        rff = kernel.RFF(D)
        Z_train = rff.fit_transform(X_train)
        Z_test = rff.transform(X_test)

        model = linear.LogisticRegression()
        model.fit(Z_train, y_train)

        t1 = time.time()

        acc = accuracy(y_test, model.predict(Z_test))

        results.append((D, acc, t1-t0))
        print("RFF D=", D, "acc=", acc, "time=", t1-t0)

    return results, lin_acc, ker_acc


# =========================
# PLOT
# =========================

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


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    D_values = [10, 50, 100, 500, 1000]

    results, lin_acc, ker_acc = run_experiment(D_values)
    plot_results(results, lin_acc, ker_acc)