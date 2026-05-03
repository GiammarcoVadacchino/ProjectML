import numpy as np
from sklearn.model_selection import KFold
import models
from utils import accuracy, error_rate
import time


def cross_validate_svm(X, y, C_values, gamma_values, kernel, k=5, lr=0.01, epochs=1000):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_score = -1.0
    best_params = None
    results = {}

    for C in C_values:
        for gamma in gamma_values:
            fold_scores = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = models.KernelSVM(
                    lr=lr,
                    epochs=epochs,
                    C=C,
                    gamma=gamma,
                    kernel=kernel
                )

                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                fold_scores.append(accuracy(y_val, y_pred))

            mean_score = np.mean(fold_scores)
            results[(C, gamma)] = mean_score

            print(f"C={C}, gamma={gamma} -> CV accuracy={mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    "C": C,
                    "gamma": gamma
                }

    return best_params



def train_evaluate_svm(X_train,y_train,X_test,y_test,C_values,gamma_values, kernel):

    results = {}
    print(f"I'm using {kernel} kernel")
    start_total = time.perf_counter()
    best_params = cross_validate_svm(X_train,y_train, C_values=C_values,gamma_values=gamma_values,kernel=kernel)
    svm_rbf = models.KernelSVM(C = best_params["C"], gamma=best_params["gamma"],kernel=kernel)
    start_train = time.perf_counter()
    svm_rbf.fit(X_train,y_train)
    end_train = time.perf_counter()
    end_total = time.perf_counter()

    y_train_pred = svm_rbf.predict(X_train)
    y_test_pred = svm_rbf.predict(X_test)
    train_error = error_rate(y_train,y_train_pred)
    test_error = error_rate(y_test,y_test_pred)
    acc = accuracy(y_test_pred,y_test)

    training_time = end_train - start_train
    total_time = end_total - start_total

    results[kernel] = [best_params["gamma"], train_error,test_error, training_time, total_time, acc]

    return results



def train_evaluate_perceptron(X_train,y_train,X_test,y_test,best_gamma, feature_map_name, D_values):

    results = {}

    if feature_map_name == "linear":

        p_linear = models.Perceptron()

        # training
        start_train = time.perf_counter()
        p_linear.fit(X_train, y_train)
        end_train = time.perf_counter()
        training_time = end_train - start_train

        # predizioni
        y_train_pred = p_linear.predict(X_train)
        y_test_pred = p_linear.predict(X_test)

        # metriche
        p_train_error = error_rate(y_train, y_train_pred)
        p_test_error = error_rate(y_test, y_test_pred)
        p_acc = accuracy(y_test, y_test_pred)

        results["linear"] = [p_train_error, p_test_error, p_acc, training_time]

        return results

    for d in D_values:
        p_rff = models.Perceptron(D = d, feature_map=feature_map_name, gamma=best_gamma)

        # training
        start_train = time.perf_counter()
        p_rff.fit(X_train, y_train)
        end_train = time.perf_counter()
        training_time = end_train - start_train

        # predizioni
        y_train_pred = p_rff.predict(X_train)
        y_test_pred = p_rff.predict(X_test)

        

        # metriche
        p_train_error = error_rate(y_train, y_train_pred)
        p_test_error = error_rate(y_test, y_test_pred)
        p_acc = accuracy(y_test, y_test_pred)

        results[d] = [p_train_error, p_test_error, p_acc,training_time]

    return results
