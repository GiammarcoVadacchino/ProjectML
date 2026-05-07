import numpy as np
from sklearn.datasets import make_classification
from ucimlrepo import fetch_ucirepo



SEED = 42
np.random.seed(SEED)


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


def aggregate_results(results_list):
    """
    Aggrega una lista di risultati con stessa struttura.
    Ritorna:
        mean_results, std_results
    """

    first = results_list[0]

    if isinstance(first, dict):
        mean_dict = {}
        std_dict = {}

        for key in first.keys():
            values = [res[key] for res in results_list]
            mean_dict[key], std_dict[key] = aggregate_results(values)

        return mean_dict, std_dict

    if isinstance(first, (list, tuple)):
        mean_values = []
        std_values = []

        for i in range(len(first)):
            values = [res[i] for res in results_list]
            mean_i, std_i = aggregate_results(values)
            mean_values.append(mean_i)
            std_values.append(std_i)

        return type(first)(mean_values), type(first)(std_values)

    if isinstance(first, (int, float, np.integer, np.floating)):
        values = np.array(results_list, dtype=float)
        return np.mean(values), np.std(values)

    return first, None


def print_stats_mean_std(mean_results, std_results, title="RESULTS"):
    """
    Stampa risultati in formato orizzontale: media ± std
    """

    print(f"\n===== {title} MEAN ± STD =====")

    for key in mean_results:
        print(f"\n{key}")

        mean_value = mean_results[key]
        std_value = std_results[key]

        if isinstance(mean_value, (list, tuple)):

            headers = [f"m{i}" for i in range(len(mean_value))]
            print(" | ".join(headers))

            values_row = []
            for i, val in enumerate(mean_value):
                std = std_value[i]

                if isinstance(val, (int, float, np.integer, np.floating)):
                    values_row.append(f"{val:.4f} ± {std:.4f}")
                else:
                    values_row.append(str(val))

            print(" | ".join(values_row))

        else:
            print(f"{mean_value:.4f} ± {std_value:.4f}")


def run_scalability_experiment(sizes, C_values, gamma_values, best_gamma_rbf, best_gamma_laplace, D_rff=500, D_binning=100):
    from train import train_evaluate_perceptron,train_evaluate_svm

    scalability_results = {
        "size": [],
        "svm_rbf_time": [],
        "svm_laplace_time": [],
        "perceptron_rff_time": [],
        "perceptron_rb_time": [],
        "svm_rbf_acc": [],
        "svm_laplace_acc": [],
        "perceptron_rff_acc": [],
        "perceptron_rb_acc": []
    }

    for size in sizes:

        print(f"\n===== SCALABILITY EXPERIMENT - SIZE {size} =====")

        np.random.seed(SEED)

        X, y = make_classification(
            n_samples= 2500,
            n_features=10,
            n_informative=4,
            n_redundant=3,
            flip_y=0.05,
            class_sep=0.8,
            random_state=SEED
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=SEED)
        X_train, X_test = standardize(X_train, X_test)

        svm_rbf_results = train_evaluate_svm(X_train, y_train, X_test, y_test, C_values, gamma_values, kernel="rbf")
        svm_laplace_results = train_evaluate_svm(X_train, y_train, X_test, y_test, C_values, gamma_values, kernel="laplace")

        np.random.seed(SEED)
        p_rff_results = train_evaluate_perceptron(X_train, y_train, X_test, y_test, best_gamma_rbf, "rff", [D_rff])

        np.random.seed(SEED)
        p_rb_results = train_evaluate_perceptron(X_train, y_train, X_test, y_test, best_gamma_laplace, "random_binning", [D_binning])

        scalability_results["size"].append(size)

        scalability_results["svm_rbf_time"].append(svm_rbf_results["rbf"][3])
        scalability_results["svm_laplace_time"].append(svm_laplace_results["laplace"][3])
        scalability_results["perceptron_rff_time"].append(p_rff_results[D_rff][3])
        scalability_results["perceptron_rb_time"].append(p_rb_results[D_binning][3])

        scalability_results["svm_rbf_acc"].append(svm_rbf_results["rbf"][5])
        scalability_results["svm_laplace_acc"].append(svm_laplace_results["laplace"][5])
        scalability_results["perceptron_rff_acc"].append(p_rff_results[D_rff][2])
        scalability_results["perceptron_rb_acc"].append(p_rb_results[D_binning][2])

    return scalability_results



def get_real_dataset(n_samples=2500, seed=42):
    # fetch dataset 
    default_of_credit_card_clients = fetch_ucirepo(id=350)

    # data as pandas DataFrame
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets

    #from datafram to array 1D
    y = y.iloc[:, 0]

    #sampling
    if n_samples is not None and n_samples < len(X):
        sampled = X.sample(n=n_samples, random_state=seed)
        X = sampled
        y = y.loc[sampled.index]

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y


