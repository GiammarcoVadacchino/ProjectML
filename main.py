
from sklearn.datasets import make_moons
from train import *
from utils import *
import numpy as np
from plot import *


SEED = 42
np.random.seed(SEED)


def main():

    sizes = [500, 1000, 2000, 4000, 8000, 15000]

    #TODO: whene i finshed the pipeline with synthetic dataset, rerune the pipeline with real world data
    #NOTE: 5k samples for exps
    X, y = make_moons(n_samples=5000,noise=0.8,random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,seed=SEED)

    X_train, X_test = standardize(X_train, X_test)

    D_rff_values = [25, 50, 100, 250, 500, 1000, 2000]
    D_binning_values = [10, 25, 50, 100, 250, 500] #NOTE: explain this
    gamma_values = np.logspace(-2, 1, 6) # need to be tuned
    C_values = np.logspace(-2, 2, 6) # need to be tuned

    #RUN PERCEPTRON LINEAR save train error, test error, accuracy, time

    p_linear_results = train_evaluate_perceptron(X_train,y_train,X_test,y_test,best_gamma=None,feature_map_name="linear",D_values=None)
    print("\n===== PERCEPTRON LINEARE =====")
    print_stats(p_linear_results)


    #RUN SVM RBF (5-fold-cv) save train error, test error, accuracy, time and gamma values
    svm_rbf_results = train_evaluate_svm(X_train,y_train,X_test,y_test,C_values,gamma_values, kernel = "rbf")
    print_stats(svm_rbf_results)

    #RUN SVM LAPLACE (5-fold-cv) save train error, test error, accuracy, time and gamma values
    svm_laplace_results = train_evaluate_svm(X_train,y_train,X_test,y_test,C_values,gamma_values, kernel = "laplace")
    print_stats(svm_laplace_results)


    best_gamma_rbf = svm_rbf_results["rbf"][3] #NOTE: best gamma founed in CV with rbf kernel has to be used in perceptron with RFF
    best_gamma_laplace = svm_laplace_results["laplace"][3] #NOTE: best gamma founed in CV with laplace kernel has to be used in perceptron with random binning features


    #RUN PERCEPTRON RFF save train error, test error, accuracy, time and D values
    p_rff_results = train_evaluate_perceptron(X_train,y_train,X_test,y_test,best_gamma_rbf,"rff",D_rff_values)
    print("\n===== PERCEPTRON RFF =====")
    print_stats(p_rff_results)

    #RUN PERCEPTRON BINNING, save train error, test error, accuracy, time and D values
    p_random_binning_results = train_evaluate_perceptron(X_train,y_train,X_test,y_test,best_gamma_laplace,"random_binning",D_binning_values)
    print("\n===== PERCEPTRON RANDOM BINNING =====")
    print_stats(p_random_binning_results)


    #PLOTS and STORE


    #Insight into representation vs computation trade-offs
    #TODO: plot test error vs number of features (for rff)
    #TODO: plot test error vs number of features (for binning)
    plot_test_error_vs_features(p_rff_results, label = "Perceptron RFF",save_path='plots/test_error_vs_features_rff')
    plot_test_error_vs_features(p_random_binning_results, label = "Perceptron RB",save_path='plots/test_error_vs_features_rb')

    

    #Evidence that random features approximate kernel performance, Comparison between linear, approximate kernel, and exact kernel models, Compare different feature map constructions
    #TODO: comparison with exact kernel method, (comparison between, train/test error, of svm_rbf, perceptron, percetron rff)
    plot_kernel_comparison(svm_rbf_results,p_linear_results,p_rff_results,kernel="rbf",save_path="plots/kernel_compariso_rbf_rff")
    #TODO: comparison with exact kernel method, (comparison between, train/test error, of svm_laplace, perceptron, percetron binning)
    plot_kernel_comparison(svm_laplace_results,p_linear_results,p_random_binning_results,kernel="laplace",approx_label="RB",save_path="plots/kernel_compariso_laplace_rb")


    #Analyze runtime vs accuracy trade-off
    #TODO: plot runtime vs accuracy (x training time, y acc) NO CV (all models so 5 lines) 2 subplots, one with training and one with test error
    plot_runtime_vs_accuracy(svm_rbf_results,p_linear_results,p_rff_results,kernel="rbf",save_path="plots/runtime_vs_accuracy_rbf_rff")
     #TODO: plot runtime vs accuracy (x total time, y acc) WITH CV (all models so 5 lines)
    plot_runtime_vs_accuracy(svm_laplace_results,p_linear_results,p_random_binning_results,kernel="laplace",approx_label="RB",save_path="plots/runtime_vs_accuracy_laplace_rb")
   

    #Scalabily of perceptron with featrues maps w.r.t SVM 
    #TODO: plot runtime vs dataset size (prove scalability of features transformations)

    """
    asse x = dataset size (n)
    asse y = training time
    curve:
    SVM rbf
    SVM laplace
    Perceptron + RFF
    Perceptron + RB
    """

    #Study effect of kernel bandwidth
    #TODO: plot SVM rbf and perceptron with RFF (x = gamma (log scale), y = train/test error)
    gammas = np.logspace(-3, 1, 8)
    study_kernel_bandwidth(X_train, y_train, X_test, y_test,gammas,D=200,svm_kernel="rbf",save_path="plots/bandwidth_rbf.png")
    study_kernel_bandwidth(X_train, y_train, X_test, y_test,gammas,D=200,svm_kernel="laplace",save_path="plots/bandwidth_laplace.png")


if __name__ == "__main__":
    main()