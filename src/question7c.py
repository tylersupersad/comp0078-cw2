from question4 import process
from kernel_perceptron import gaussian_kernel

if __name__ == "__main__":
    # question 7(c)
    process(
        num_runs=20,
        values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        kernel_func=gaussian_kernel,
        method="ovr",
    )