from question4 import process
from kernel_perceptron import polynomial_kernel

if __name__ == "__main__":
    # question 8(c)
    process(
        num_runs=20,
        values=range(1, 8),
        kernel_func=polynomial_kernel,
        method="ovo",
    )