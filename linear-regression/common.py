import numpy as np

__training_proportions = 2/3;

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


def load_concrete_data():
    raw_data = np.loadtxt(open("Concrete_Data.csv", "rb"), delimiter=",", skiprows=1)
    x = raw_data[:, :-1]
    #x = add_ones_column(x)
    y = raw_data[:, -1:]
    return x, y


def get_training_set(x_all, y_all):
    # 2/3 of all data is training set.
    m_training = int(x_all.shape[0] * __training_proportions)
    x = x_all[:m_training]
    y = y_all[:m_training]
    return x, y


def get_test_set(x_all, y_all):
    # 1/3 of all data is test set.
    m_training = int(x_all.shape[0] * __training_proportions)
    x = x_all[m_training:]
    y = y_all[m_training:]
    return x, y
