import numpy as np


def add_ones_column(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))


def check_result(y, y_prediction):
    m = y.shape[0]
    error = y - y_prediction
    error_squared = error ** 2
    exp = np.sum(error) / m
    exp_squared = exp ** 2
    var = np.sum(error_squared - exp_squared) / m
    return exp, var
