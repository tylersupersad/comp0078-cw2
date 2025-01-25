from question3 import main
from kernel_perceptron import polynomial_kernel

if __name__ == "__main__":
    # question 8(b): polynomial kernel with ovo
    main(
        kernel_func=polynomial_kernel,
        param_values=range(1, 8),  # d = 1 to 7
        method="ovo",
        description="OvO with Polynomial Kernel"
    )