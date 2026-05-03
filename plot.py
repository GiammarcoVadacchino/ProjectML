import numpy as np
import matplotlib.pyplot as plt
from models import KernelSVM, Perceptron
from utils import error_rate


def plot_test_error_vs_features(results, label="Perceptron + RFF", save_path=None):
    
    D_values = sorted(results.keys())
    test_errors = [results[D][1] for D in D_values]

    plt.figure(figsize=(8, 5))
    plt.plot(D_values, test_errors, marker="o", label=label)

    plt.xlabel("Number of features (D)")
    plt.ylabel("Test error")
    plt.title("Test error vs number of features")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_kernel_comparison(results_kernel, results_linear, results_approx, kernel,
                           approx_label="RFF",
                           save_path=None):

    # -------- Approx (RFF / Binning) --------
    D_values = sorted(results_approx.keys())

    approx_train = [results_approx[D][0] for D in D_values]
    approx_test = [results_approx[D][1] for D in D_values]

    # -------- Kernel (SVM) --------
    kernel_train = results_kernel[kernel][1]
    kernel_test = results_kernel[kernel][2]

    # -------- Linear (Perceptron) --------
    linear_train = results_linear["linear"][0]
    linear_test = results_linear["linear"][1]

    # posizione x per punti fissi
    x_pos = sum(D_values) / len(D_values)

    # -------- Plot --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # ===== LEFT: TEST ERROR =====
    axes[0].plot(D_values, approx_test, marker="o", label=f"Perceptron + {approx_label}")
    axes[0].scatter(x_pos, kernel_test, marker="x", s=100, label="SVM (exact kernel)")
    axes[0].scatter(x_pos, linear_test, marker="s", label="Perceptron (linear)")

    axes[0].set_title("Test error")
    axes[0].set_xlabel("Number of features (D)")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ===== RIGHT: TRAIN ERROR =====
    axes[1].plot(D_values, approx_train, marker="o", label=f"Perceptron + {approx_label}")
    axes[1].scatter(x_pos, kernel_train, marker="x", s=100, label="SVM (exact kernel)")
    axes[1].scatter(x_pos, linear_train, marker="s", label="Perceptron (linear)")

    axes[1].set_title("Train error")
    axes[1].set_xlabel("Number of features (D)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Kernel approximation comparison")
    plt.tight_layout()

    # -------- Save --------
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_runtime_vs_accuracy(results_kernel, results_linear, results_approx, kernel,
                             approx_label="RFF", save_path=None):

    # -------- Approx (RFF / Binning) --------
    D_values = sorted(results_approx.keys())

    times = [results_approx[D][3] for D in D_values]
    accs = [results_approx[D][2] for D in D_values]

    # -------- Kernel --------
    kernel_time = results_kernel[kernel][3]
    kernel_acc = results_kernel[kernel][5]
    kernel_total_time = results_kernel[kernel][4]

    # -------- Linear --------
    linear_time = results_linear["linear"][3]
    linear_acc = results_linear["linear"][2]

    # -------- Plot --------
    plt.figure(figsize=(8, 5))

    # curva trade-off
    plt.plot(times, accs, marker="o", label=f"Perceptron + {approx_label}")

    # punti fissi
    plt.scatter(kernel_time, kernel_acc, marker="x", s=100, label="SVM (exact kernel)")
    plt.scatter(linear_time, linear_acc, marker="s", label="Perceptron (linear)")
    plt.scatter(kernel_total_time, kernel_acc, marker="*", s=150, label="SVM (+CV)")


    plt.xlabel("Training time")
    plt.ylabel("Accuracy")
    plt.title("Runtime vs Accuracy Trade-off")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xscale("log")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def study_kernel_bandwidth(X_train, y_train, X_test, y_test, gammas, D=200, buckets=32, lr=0.001, epochs=1000, C=1.0, svm_kernel="rbf", save_path=None):

    results = {
        "svm": {"train_err": [], "test_err": []},
        "rff": {"train_err": [], "test_err": []},
        "rb":  {"train_err": [], "test_err": []}
    }

    for gamma in gammas:

        # -------- SVM --------
        svm = KernelSVM(lr=lr, epochs=epochs, C=C, gamma=gamma, kernel=svm_kernel)
        svm.fit(X_train, y_train)

        results["svm"]["train_err"].append(error_rate(y_train, svm.predict(X_train)))
        results["svm"]["test_err"].append(error_rate(y_test, svm.predict(X_test)))

        # -------- Perceptron + RFF --------
        perc_rff = Perceptron(lr=lr, epochs=epochs, feature_map="rff", D=D, gamma=gamma)
        perc_rff.fit(X_train, y_train)

        results["rff"]["train_err"].append(error_rate(y_train, perc_rff.predict(X_train)))
        results["rff"]["test_err"].append(error_rate(y_test, perc_rff.predict(X_test)))

        # -------- Perceptron + RB --------
        perc_rb = Perceptron(lr=lr, epochs=epochs, feature_map="random_binning", D=D, gamma=gamma, buckets=buckets)
        perc_rb.fit(X_train, y_train)

        results["rb"]["train_err"].append(error_rate(y_train, perc_rb.predict(X_train)))
        results["rb"]["test_err"].append(error_rate(y_test, perc_rb.predict(X_test)))

    # -------- Plot --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # TEST
    axes[0].plot(gammas, results["svm"]["test_err"], marker="o", label=f"SVM {svm_kernel}")
    axes[0].plot(gammas, results["rff"]["test_err"], marker="s", label=f"Perceptron + RFF (D={D})")
    axes[0].plot(gammas, results["rb"]["test_err"], marker="^", label=f"Perceptron + RB (D={D})")
    axes[0].set_title("Test error")
    axes[0].set_xlabel("Gamma")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # TRAIN
    axes[1].plot(gammas, results["svm"]["train_err"], marker="o", label=f"SVM {svm_kernel}")
    axes[1].plot(gammas, results["rff"]["train_err"], marker="s", label=f"Perceptron + RFF (D={D})")
    axes[1].plot(gammas, results["rb"]["train_err"], marker="^", label=f"Perceptron + RB (D={D})")
    axes[1].set_title("Train error")
    axes[1].set_xlabel("Gamma")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")

    plt.suptitle(f"Effect of kernel bandwidth ({svm_kernel})")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
