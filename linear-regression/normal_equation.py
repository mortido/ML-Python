import numpy as np


def normal_equation(x, y):
    # return theta
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
