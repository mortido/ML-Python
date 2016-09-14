import numpy as np


def simple_function():
    # y = 1.5x - 2
    X = np.linspace(0, 5, 10)
    Y = 1.5 * X - 2
    return X, Y


def generate_simple_data():
    n = 200
    X = np.random.uniform(0.5, 4.5, (n, 2))
    idx = 1.5 * X[:, 0] >= X[:, 1] + 2 + np.random.uniform(-0.5, 0.5, n)
    Y = np.zeros(n)
    Y[idx] = 1
    return X, Y
