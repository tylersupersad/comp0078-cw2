import numpy as np
from numba import jit
from joblib import Parallel, delayed

# polynomial kernel function
@jit
def polynomial_kernel(p, q, d):
    if d <= 0:
        raise ValueError("\n[!] d must be a positive integer")
    return (np.dot(p, q)) ** d

# gaussian kernel function
@jit
def gaussian_kernel(p, q, c):
    if c <= 0:
        raise ValueError("\n[!] c must be a positive float")
    diff = p - q
    return np.exp(-c * np.dot(diff, diff))

# function to load and preprocess .dat files
def load_data(file_path):
    data = np.loadtxt(file_path)
    y = data[:, 0]  # extract labels
    x = data[:, 1:]  # extract features
    x = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1  # scale features
    return x, y

# splits data into training and testing sets
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    split_point = int(train_ratio * len(indices))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

# precomputes the kernel matrix for training using optimized vectorized operations
def precompute_kernel_matrix_vectorized(x, kernel_func, param):
    # number of samples in the input data
    n_samples = x.shape[0]
    # initialize an empty kernel matrix
    kernel_matrix = np.zeros((n_samples, n_samples))
    # compute the kernel values for each pair of samples
    for i in range(n_samples):
        # calculate the kernel values for the upper triangle (including diagonal)
        kernel_matrix[i, i:] = [kernel_func(x[i], x[j], param) for j in range(i, n_samples)]
        # copy the values to the lower triangle for symmetry
        kernel_matrix[i:, i] = kernel_matrix[i, i:]
    # return the fully computed kernel matrix
    return kernel_matrix

# computes the kernel matrix for test data
def compute_test_kernel_matrix(x_test, x_train, kernel_func, param):
    # number of samples in the test data
    n_test = x_test.shape[0]
    # number of samples in the training data
    n_train = x_train.shape[0]
    # initialize an empty kernel matrix for test vs training samples
    kernel_matrix = np.zeros((n_test, n_train))
    # compute the kernel values for each test and training pair
    for i in range(n_test):
        kernel_matrix[i, :] = [kernel_func(x_test[i], x_train[j], param) for j in range(n_train)]
    # return the computed kernel matrix
    return kernel_matrix

# early stopping logic
def early_stopping(training_error, best_error, patience_counter, patience):
    if training_error < best_error:
        best_error = training_error
        patience_counter = 0
    else:
        patience_counter += 1

    stop = patience_counter >= patience
    return stop, best_error, patience_counter

# train one-vs-rest kernel perceptron with precomputed kernel matrix
def train_ovr(x, y, kernel_func, param, run_id, max_epochs=50, patience=5):
    classes = np.unique(y)
    kernel_matrix = precompute_kernel_matrix_vectorized(x, kernel_func, param)

    classifiers = {}
    for c in classes:
        # create binary labels: 1 for class c, -1 for others
        binary_labels = np.where(y == c, 1, -1)
        alpha = train_kernel_perceptron_with_matrix(
            kernel_matrix, binary_labels, max_epochs, patience, run_id, class_label=c
        )
        classifiers[c] = (alpha, binary_labels)
        print(f"\n[+] Run {run_id}: Trained classifier for class {c}")
    return classifiers

# trains kernel perceptron using a precomputed kernel matrix
def train_kernel_perceptron_with_matrix(kernel_matrix, y, max_epochs=50, patience=5, run_id=0, class_label=None):
    # number of training samples
    n_samples = kernel_matrix.shape[0]
    # initialize alpha values for each sample
    alpha = np.zeros(n_samples)
    # track the best error observed during training
    best_error = float('inf')
    # initialize patience counter for early stopping
    patience_counter = 0

    # iterate through epochs
    for epoch in range(max_epochs):
        errors = 0

        # iterate through each sample
        for i in range(n_samples):
            # compute prediction using precomputed kernel values
            prediction = np.sum(alpha * y * kernel_matrix[:, i])
            # update alpha if prediction is incorrect
            if np.sign(prediction) != y[i]:
                alpha[i] += 1
                errors += 1

        # update best error and reset patience counter if improved
        if errors < best_error:
            best_error = errors
            patience_counter = 0
        else:
            # increment patience counter if no improvement
            patience_counter += 1

        # stop training early if patience limit is reached
        if patience_counter >= patience:
            print(f"[DEBUG] Early stopping for class {class_label} at epoch {epoch}.")
            break

    # return the final alpha values
    return alpha

# predicts labels using one-vs-rest classifiers
def predict_ovr(x_test, classifiers, x_train, kernel_func, param, run_id, kernel_matrix=None):
    # use precomputed kernel matrix if provided
    if kernel_matrix is not None:
        test_kernel_matrix = kernel_matrix
    else:
        # compute kernel matrix for test data
        test_kernel_matrix = compute_test_kernel_matrix(x_test, x_train, kernel_func, param)

    # initialize list to store confidence scores for each classifier
    confidences = []
    # iterate through classifiers to compute confidence scores
    for c, (alpha, y_train) in classifiers.items():
        # compute confidence scores for the current classifier
        confidence = test_kernel_matrix.dot(alpha * y_train)
        confidences.append(confidence)

    # convert list of confidences to a numpy array and transpose
    confidences = np.array(confidences).T
    # determine the class with the highest confidence for each sample
    predictions = np.argmax(confidences, axis=1)
    # return the predicted labels
    return predictions

# one-vs-one kernel perceptron training with optimization
def train_ovo(x, y, kernel_func, param, run_id, max_epochs=50, patience=5):
    classes = np.unique(y)
    pairs = [(c1, c2) for i, c1 in enumerate(classes) for c2 in classes[i + 1:]]
    classifiers = {}

    def train_pair(c1, c2):
        indices = (y == c1) | (y == c2)
        x_pair, y_pair = x[indices], y[indices]
        binary_labels = np.where(y_pair == c1, 1, -1)
        alpha = np.zeros(len(x_pair))

        # precompute the kernel matrix
        kernel_matrix = np.array([
            [kernel_func(x_pair[i], x_pair[j], param) for j in range(len(x_pair))]
            for i in range(len(x_pair))
        ])

        print(f"[DEBUG][Run {run_id}] Starting training for pair ({c1}, {c2}).")
        best_error = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            errors = 0
            # update alpha for misclassified samples
            for j in range(len(x_pair)):
                prediction = np.sum(alpha * binary_labels * kernel_matrix[j])
                if np.sign(prediction) != binary_labels[j]:
                    alpha[j] += 1
                    errors += 1

            # debug alpha sum and epoch errors
            alpha_sum = np.sum(alpha)
            print(f"[DEBUG][Run {run_id}] Pair ({c1}, {c2}), Epoch {epoch}, Errors: {errors}, Alpha sum: {alpha_sum}")

            # early stopping logic
            if errors < best_error:
                best_error = errors
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"[DEBUG][Run {run_id}] Early stopping for pair ({c1}, {c2}) at epoch {epoch}.")
                break

        return (c1, c2), (alpha, x_pair, binary_labels)

    # Train classifiers for all pairs in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_pair)(c1, c2) for c1, c2 in pairs
    )

    for pair, model in results:
        classifiers[pair] = model

    print(f"[DEBUG][Run {run_id}] OvO classifiers trained.")
    return classifiers

def predict_ovo(x_test, classifiers, kernel_func, param, run_id):
    print(f"[DEBUG][Run {run_id}] Predicting using OvO classifiers with vectorization...")

    # create a mapping from class labels to indices
    unique_classes = np.unique([key[0] for key in classifiers.keys()] + [key[1] for key in classifiers.keys()])
    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    # initialize votes array
    votes = np.zeros((len(x_test), num_classes))

    # precompute kernel matrix for all training data in classifiers
    for (c1, c2), (alpha, x_train, y_train) in classifiers.items():
        idx_c1, idx_c2 = class_to_index[c1], class_to_index[c2]

        # efficient kernel computation between x_test and x_train
        kernel_matrix = np.dot(x_test, x_train.T) ** param if kernel_func.__name__ == "polynomial_kernel" else \
            np.exp(-param * (np.sum(x_test ** 2, axis=1, keepdims=True) - 2 * np.dot(x_test, x_train.T) + np.sum(x_train ** 2, axis=1).T))

        # compute scores
        scores = kernel_matrix @ (alpha * y_train)

        # update votes based on scores
        votes[scores > 0, idx_c1] += 1
        votes[scores <= 0, idx_c2] += 1

    # determine final predictions based on majority vote
    predictions = np.argmax(votes, axis=1)

    # map back to original class labels
    index_to_class = {idx: label for label, idx in class_to_index.items()}
    predictions = np.array([index_to_class[idx] for idx in predictions])

    return predictions