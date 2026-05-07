from train import *
from utils import *
import numpy as np
from plot import *
from sklearn.datasets import make_classification

SEED = 42
np.random.seed(SEED)


RUN_SCALABILITY_EXPERIMENT = False #flag
USE_REAL_DATASET = True #switch if you want to use synthetic dataset

def main():

    #params for multiple run experiments of random features
    n_runs = 5
    seeds = [SEED + i for i in range(n_runs)]


    if USE_REAL_DATASET:
        X, y = get_real_dataset(n_samples=2500, seed=SEED)
    else:
        #crate synthetic non linear dataset
        X, y = make_classification(
            n_samples= 2500,
            n_features=10,
            n_informative=4,
            n_redundant=3,
            flip_y=0.05,
            class_sep=0.8,
            random_state=SEED
        )

    sizes = [500, 1000, 2500, 4000, 8000, 15000]

    #Train and test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=SEED)
    X_train, X_test = standardize(X_train, X_test)

    D_rff_values = [25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000] # d values for RFF
    D_binning_values = [10, 25, 50, 100, 250, 500, 1000, 2000] # da values for RB (the matrix is then d * buckets, this explain the fact that i use smaller values of d)
    gamma_values = np.logspace(-2, 1, 6) 
    C_values = np.logspace(-2, 2, 6)


    #run linear perceptron
    p_linear_results = train_evaluate_perceptron(X_train, y_train, X_test, y_test, best_gamma=None, feature_map_name="linear", D_values=None)
    print("\n===== PERCEPTRON LINEARE =====")
    print_stats(p_linear_results)

    #run SVM with RBF kernel
    svm_rbf_results = train_evaluate_svm(X_train, y_train, X_test, y_test, C_values, gamma_values, kernel="rbf")
    print("\n===== SVM RBF =====")
    print_stats(svm_rbf_results)

    #run SVM with Laplace kernel
    svm_laplace_results = train_evaluate_svm(X_train, y_train, X_test, y_test, C_values, gamma_values, kernel="laplace")
    print("\n===== SVM LAPLACE =====")
    print_stats(svm_laplace_results)

    #take the best gammas after 5-fold-cross-validation
    best_gamma_rbf = svm_rbf_results["rbf"][3]
    best_gamma_laplace = svm_laplace_results["laplace"][3]



    all_p_rff_results = []
    all_p_random_binning_results = []

    #multiple runs for random features
    for run_id, seed in enumerate(seeds):

        print(f"\n\n========== RUN {run_id + 1}/{n_runs} - SEED {seed} ==========")

        np.random.seed(seed)

        #run Random Fourier Features
        p_rff_results_run = train_evaluate_perceptron(X_train, y_train, X_test, y_test, best_gamma_rbf, "rff", D_rff_values)
        print("\n===== PERCEPTRON RFF - SINGLE RUN =====")
        print_stats(p_rff_results_run)
        all_p_rff_results.append(p_rff_results_run)

        np.random.seed(seed)

        #run Random Binning Features
        p_random_binning_results_run = train_evaluate_perceptron(X_train, y_train, X_test, y_test, best_gamma_laplace, "random_binning", D_binning_values)
        print("\n===== PERCEPTRON RANDOM BINNING - SINGLE RUN =====")
        print_stats(p_random_binning_results_run)
        all_p_random_binning_results.append(p_random_binning_results_run)


    #aggregate and print results (mean +- std)
    p_rff_results, p_rff_std = aggregate_results(all_p_rff_results)
    p_random_binning_results, p_random_binning_std = aggregate_results(all_p_random_binning_results)

    print_stats_mean_std(p_rff_results, p_rff_std, title="PERCEPTRON RFF")
    print_stats_mean_std(p_random_binning_results, p_random_binning_std, title="PERCEPTRON RANDOM BINNING")



    real = "" #flag to distinguish if i use real or synthetic data
    if USE_REAL_DATASET: real = "real"

    plot_test_error_vs_features(p_rff_results,label="Perceptron RFF", save_path=f"plots/test_error_vs_features_rff_{real}")
    plot_test_error_vs_features(p_random_binning_results, label="Perceptron RB", save_path=f"plots/test_error_vs_features_rb_{real}")

    plot_kernel_comparison(svm_rbf_results, p_linear_results, p_rff_results, kernel="rbf", save_path=f"plots/kernel_compariso_rbf_rff_{real}")
    plot_kernel_comparison(svm_laplace_results, p_linear_results, p_random_binning_results, kernel="laplace", approx_label="RB", save_path=f"plots/kernel_compariso_laplace_rb_{real}")

    plot_runtime_vs_accuracy(svm_rbf_results, p_linear_results, p_rff_results, kernel="rbf", save_path=f"plots/runtime_vs_accuracy_rbf_rff_{real}")
    plot_runtime_vs_accuracy(svm_laplace_results, p_linear_results, p_random_binning_results, kernel="laplace", approx_label="RB", save_path=f"plots/runtime_vs_accuracy_laplace_rb_{real}")

    gammas = np.logspace(-3, 1, 8)

    study_kernel_bandwidth(X_train, y_train, X_test, y_test, gammas, D=200, svm_kernel="rbf", save_path=f"plots/bandwidth_rbf_{real}")
    study_kernel_bandwidth(X_train, y_train, X_test, y_test, gammas, D=200, svm_kernel="laplace", save_path=f"plots/bandwidth_laplace_{real}")

    if RUN_SCALABILITY_EXPERIMENT:

        scalability_results = run_scalability_experiment(
            sizes=sizes,
            C_values=C_values,
            gamma_values=gamma_values,
            best_gamma_rbf=best_gamma_rbf,
            best_gamma_laplace=best_gamma_laplace,
            D_rff=500,
            D_binning=100
        )

        plot_scalability_results(
            scalability_results,
            save_path="plots/scalability_training_time"
        )
    
if __name__ == "__main__":
    main()