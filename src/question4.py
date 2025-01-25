import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed, parallel_backend
from kernel_perceptron import (
    load_data,
    split_data,
    train_ovr,
    predict_ovr,
    train_ovo,
    predict_ovo,
    polynomial_kernel,
    gaussian_kernel,
    precompute_kernel_matrix_vectorized
)

# precompute kernel matrices for cross-validation
def precompute_kernel_cache(x_train, values, kernel_func):
    return {value: precompute_kernel_matrix_vectorized(x_train, kernel_func, param=value) for value in values}

# function for cross-validation
def cross_validate(x_train, y_train, values, kernel_func, method, max_epochs=50, patience=5, run_id=0):
    print(f"[DEBUG][Run {run_id}] Starting cross-validation...")
    fold_size = len(x_train) // 5
    errors = {value: [] for value in values}

    # precompute kernel matrices
    kernel_cache = precompute_kernel_cache(x_train, values, kernel_func)

    def evaluate_fold(fold):
        print(f"[DEBUG][Run {run_id}] Evaluating fold {fold + 1}/5...")
        val_indices = range(fold * fold_size, (fold + 1) * fold_size)
        mask = np.ones(len(x_train), dtype=bool)
        mask[val_indices] = False
        train_indices = np.where(mask)[0]

        x_train_fold, y_train_fold = x_train[train_indices], y_train[train_indices]
        x_val_fold, y_val_fold = x_train[val_indices], y_train[val_indices]

        fold_errors = {}
        for value in values:
            # use precomputed kernel matrix for the current parameter value
            kernel_matrix = kernel_cache[value][train_indices][:, train_indices]
            val_kernel_matrix = kernel_cache[value][val_indices][:, train_indices]

            classifiers = train_ovr(
                x_train_fold, y_train_fold, kernel_func, param=value,
                run_id=run_id, max_epochs=max_epochs, patience=patience
            ) if method == "ovr" else train_ovo(
                x_train_fold, y_train_fold, kernel_func, param=value,
                run_id=run_id, max_epochs=max_epochs, patience=patience
            )

            y_val_pred = predict_ovr(
                x_val_fold, classifiers, x_train, kernel_func, param=value,
                run_id=run_id, kernel_matrix=val_kernel_matrix
            ) if method == "ovr" else predict_ovo(
                x_val_fold, classifiers, kernel_func, param=value,
                run_id=run_id
            )

            fold_errors[value] = 100 * np.mean(y_val_fold != y_val_pred)
        return fold_errors

    # parallelize fold evaluations
    with parallel_backend('loky', n_jobs=-1):  # Prevent nested parallelism
        fold_results = Parallel(n_jobs=-1)(delayed(evaluate_fold)(fold) for fold in range(5))

    # aggregate results
    for fold_error in fold_results:
        for value, error in fold_error.items():
            errors[value].append(error)

    # compute average errors and find the best parameter
    avg_errors = {value: np.mean(errors[value]) for value in values}
    best_value = min(avg_errors, key=avg_errors.get)
    print(f"[DEBUG][Run {run_id}] Cross-validation completed. Best value: {best_value}")
    return best_value

# Reusable function for training and evaluation
def train_and_evaluate(x_train, y_train, x_test, y_test, kernel_func, method, best_param, max_epochs=50, patience=5, run_id=0):
    print(f"[DEBUG][Run {run_id}] Training with best_param={best_param}...")

    if method == "ovr":
        classifiers = train_ovr(
            x_train, y_train, kernel_func, param=best_param,
            run_id=run_id, max_epochs=max_epochs, patience=patience
        )
        y_train_pred = predict_ovr(x_train, classifiers, x_train, kernel_func, param=best_param, run_id=run_id)
        y_test_pred = predict_ovr(x_test, classifiers, x_train, kernel_func, param=best_param, run_id=run_id)
    elif method == "ovo":
        classifiers = train_ovo(
            x_train, y_train, kernel_func, param=best_param,
            run_id=run_id, max_epochs=max_epochs, patience=patience
        )
        y_train_pred = predict_ovo(x_train, classifiers, kernel_func, param=best_param, run_id=run_id)
        y_test_pred = predict_ovo(x_test, classifiers, kernel_func, param=best_param, run_id=run_id)
    else:
        raise ValueError(f"Unsupported method: {method}")

    train_error = 100 * np.mean(y_train != y_train_pred)
    test_error = 100 * np.mean(y_test != y_test_pred)
    return train_error, test_error, y_test_pred

# function for aggregated confusion matrix computation and visualization
def compute_and_visualize_confusion_matrix(conf_matrices, labels, save_path=None):
    num_classes = len(labels)
    confusion_counts = np.sum(conf_matrices, axis=0)

    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    confusion_rates = confusion_counts / row_sums
    np.fill_diagonal(confusion_rates, 0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_rates, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Aggregated Confusion Matrix (Error Rates)')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# function for hardest-to-predict samples visualization
def visualize_hardest_samples(x_test, y_test, y_pred, sample_indices, save_prefix="hardest_sample"):
    for i, idx in enumerate(sample_indices):
        sample = x_test[idx].reshape(16, 16) 
        true_label = y_test[idx]
        predicted_label = y_pred[idx]

        plt.figure(figsize=(4, 4))
        plt.imshow(sample, cmap='gray')
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        save_path = f"{save_prefix}_{i + 1}.png"
        plt.savefig(save_path)
        plt.close()

def process(num_runs, values, kernel_func, method, max_epochs=50, patience=5, save_path=None, visualize_hardest=False, zipcombo_path="../data/zipcombo.dat"):
    x, y = load_data(zipcombo_path)
    labels = np.unique(y)
    
    # initialize lists to store results
    train_errors, test_errors, best_params = [], [], []
    conf_matrices = []
    misclassified_indices = []

    for run in range(num_runs):
        run_start_time = time.time()
        
        # split data into training and testing sets
        x_train, y_train, x_test, y_test = split_data(x, y)
        
        # perform cross-validation to find the best parameter
        best_param = cross_validate(x_train, y_train, values, kernel_func, method, max_epochs=max_epochs, patience=patience, run_id=run + 1)
        best_params.append(best_param)  # Track best parameters
        
        # train and evaluate the model with the best parameter
        train_error, test_error, y_test_pred = train_and_evaluate(
            x_train, y_train, x_test, y_test, kernel_func, method, best_param, max_epochs=max_epochs, patience=patience, run_id=run + 1
        )
        train_errors.append(train_error)
        test_errors.append(test_error)
        conf_matrices.append(confusion_matrix(y_test, y_test_pred, labels=labels))
        misclassified_indices.extend(np.where(y_test != y_test_pred)[0])
        
        run_end_time = time.time()
        print(f"[DEBUG] Run {run + 1}/{num_runs} completed in {run_end_time - run_start_time:.2f} seconds.")

    # compute mean and standard deviation for d*, train errors, and test errors
    d_mean_std = (np.mean(best_params), np.std(best_params))
    train_mean_std = (np.mean(train_errors), np.std(train_errors))
    test_mean_std = (np.mean(test_errors), np.std(test_errors))

    # display results
    print(f"[+] Best d* (Mean ± STD): {d_mean_std[0]:.2f} ± {d_mean_std[1]:.2f}")
    print(f"[+] Train Error (Mean ± STD): {train_mean_std[0]:.2f} ± {train_mean_std[1]:.2f}")
    print(f"[+] Test Error (Mean ± STD): {test_mean_std[0]:.2f} ± {test_mean_std[1]:.2f}")

    # (optional) save and visualize confusion matrix
    if save_path:
        compute_and_visualize_confusion_matrix(conf_matrices, labels, save_path)

    # (optional) visualize hardest-to-predict samples
    if visualize_hardest:
        hardest_indices = misclassified_indices[:5]
        visualize_hardest_samples(x_test, y_test, y_test_pred, hardest_indices)

# instance for question 4, 5, and 6
def question4_5_6():
    process(
        num_runs=20,
        values=range(1, 8),
        kernel_func=polynomial_kernel,
        method="ovr",
        save_path="confusion_matrix.png",
        visualize_hardest=True,
    )
    
if __name__ == "__main__":
    # question 4, 5, and 6
    question4_5_6()