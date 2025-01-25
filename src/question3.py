import time 
import numpy as np
from joblib import Parallel, delayed
from kernel_perceptron import (
    load_data,
    split_data,
    train_ovr,
    predict_ovr,
    train_ovo,
    predict_ovo,
    polynomial_kernel,
)

# function to run a single training/testing task
def single_run(x_train, y_train, x_test, y_test, kernel_func, param, method, run_id):
    print(f"[DEBUG][Run {run_id}] Training with param={param} using {method}...")

    # fixed max_epochs and patience for early stopping
    max_epochs = 50
    patience = 5

    # train and predict based on the method
    if method == "ovr":
        classifiers = train_ovr(x_train, y_train, kernel_func, param, run_id, max_epochs=max_epochs, patience=patience)
        train_pred = predict_ovr(x_train, classifiers, x_train, kernel_func, param, run_id)
        test_pred = predict_ovr(x_test, classifiers, x_train, kernel_func, param, run_id)
    elif method == "ovo":
        classifiers = train_ovo(x_train, y_train, kernel_func, param, run_id, max_epochs=max_epochs, patience=patience)
        train_pred = predict_ovo(x_train, classifiers, kernel_func, param, run_id)
        test_pred = predict_ovo(x_test, classifiers, kernel_func, param, run_id)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # calculate errors
    train_error = np.mean(train_pred != y_train)
    test_error = np.mean(test_pred != y_test)

    # log error difference for parameter evaluation
    error_diff = abs(test_error - train_error)
    print(f"[EVAL][Run {run_id}] Param={param}: Train Error={train_error:.4f}, Test Error={test_error:.4f}, Error Difference={error_diff:.4f}")

    # log if this parameter performs the best so far (handled in aggregation)
    return train_error, test_error

# function to run experiments across parameters for a single run
def run_experiment(x, y, kernel_func, param_values, method, train_ratio, run_id):
    #print(f"[DEBUG][Run {run_id}] Splitting data for training/testing...")
    start_time = time.time()  # start timing

    # split data into training and testing sets
    x_train, y_train, x_test, y_test = split_data(x, y, train_ratio)

    # parallelize across parameters
    #print(f"[DEBUG][Run {run_id}] Starting parallel training across params...")
    results = Parallel(n_jobs=-1)(
        delayed(single_run)(x_train, y_train, x_test, y_test, kernel_func, param, method, run_id)
        for param in param_values
    )

    # separate train and test errors
    train_errors, test_errors = zip(*results)
    end_time = time.time()  # end timing
    print(f"[DEBUG][Run {run_id}] Completed in {end_time - start_time:.2f} seconds.")
    return train_errors, test_errors

# function to aggregate results across multiple runs
def aggregate_results(num_runs, x, y, kernel_func, param_values, method, train_ratio):
    all_train_errors = []
    all_test_errors = []
    best_test_error = float('inf')
    best_param = None

    # parallelize across runs
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(x, y, kernel_func, param_values, method, train_ratio, run_id)
        for run_id in range(num_runs)
    )

    # collect errors from all runs
    for run_id, (train_errors, test_errors) in enumerate(results):
        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)

        # identify best parameter for this run
        for i, param in enumerate(param_values):
            if test_errors[i] < best_test_error:
                best_test_error = test_errors[i]
                best_param = param
                print(f"[BEST][Run {run_id}] New Best Param={param} with Test Error={best_test_error:.4f}")

    print(f"[DEBUG] Completed all runs.")
    print(f"[SUMMARY] Best Gaussian Parameter: {best_param}, Test Error: {best_test_error:.4f}")
    return np.array(all_train_errors), np.array(all_test_errors)

# function to process and display results
def process_results(all_train_errors, all_test_errors, param_values, description):
    # calculate mean and standard deviation for train and test errors
    train_mean_std = [
        (np.mean(all_train_errors[:, i]), np.std(all_train_errors[:, i]))
        for i in range(len(param_values))
    ]
    test_mean_std = [
        (np.mean(all_test_errors[:, i]), np.std(all_test_errors[:, i]))
        for i in range(len(param_values))
    ]

    # display results
    print(f"\nResults for {description}:")
    print("Param | Train Error (Mean ± STD) | Test Error (Mean ± STD)")
    for i, param in enumerate(param_values):
        train_mean, train_std = train_mean_std[i]
        test_mean, test_std = test_mean_std[i]
        print(f"{param} | {train_mean:.4f}±{train_std:.4f} | {test_mean:.4f}±{test_std:.4f}")

# main function
def main(kernel_func, param_values, method, description, train_ratio=0.8, num_runs=20, data_file="../data/zipcombo.dat"):
    print(f"[DEBUG] Loading data from {data_file}...")

    # load dataset
    x, y = load_data(data_file)

    # aggregate results across runs
    print(f"[DEBUG] Starting experiments with {method}...")
    all_train_errors, all_test_errors = aggregate_results(
        num_runs=num_runs,
        x=x,
        y=y,
        kernel_func=kernel_func,
        param_values=param_values,
        method=method,
        train_ratio=train_ratio
    )

    # process and display results
    process_results(all_train_errors, all_test_errors, param_values, description)
    print(f"[DEBUG] Experiments for {method} completed.")

if __name__ == "__main__":
    # q3: polynomial kernel with ovr
    main(
        kernel_func=polynomial_kernel,
        param_values=range(1, 8), 
        method="ovr",
        description="OvR with Polynomial Kernel"
    )