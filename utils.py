import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def error_rate(y_true, y_pred):
    return 1.0 - accuracy(y_true, y_pred)


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_test = int(len(X) * test_size)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return (X_train - mean)/std, (X_test - mean)/std



def print_stats(results):

    for d, values in results.items():
        if len(values) == 4 and d == "linear":
            #linear perceptron
            train_err, test_err, acc, training_time = values
            print(f"Train err: {train_err:.4f} | Test err: {test_err:.4f} | Training Time: {training_time:.4f} | Acc: {acc:.4f}")
        elif len(values) == 4:
            #perceptron with feauters map
            train_err, test_err, acc, training_time = values
            print(f"D={d:4d} | Train err: {train_err:.4f} | Test err: {test_err:.4f} | Training Time: {training_time:.4f} | Acc: {acc:.4f}")
        else:
            #SVM
            best_gamma, train_err, test_err, training_time,total_time, acc = values
            print(f"Best Gamma={best_gamma:4} | Train err: {train_err:.4f} | Test err: {test_err:.4f} | Training Time: {training_time:.4f} | Total (train + CV) Time: {total_time:.4f} |Acc: {acc:.4f}")
        
    return 
