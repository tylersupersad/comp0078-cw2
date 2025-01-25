from question3 import main
from kernel_perceptron import gaussian_kernel

if __name__ == "__main__":
    # question 7(b): gaussian kernel with ovr
    main(
        kernel_func=gaussian_kernel,
        param_values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        method="ovr",
        description="OvR with Gaussian Kernel"
    )