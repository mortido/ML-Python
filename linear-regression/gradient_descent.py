import numpy as np


def gradient_descent(x, y, a, iterations):
    # return theta
    theta = np.zeros((x.shape[1], 1))
    m = x.shape[0]
    for i in range(0, iterations):
        gradient = x.T.dot(x.dot(theta)-y) / m

        # for test purpose
        cost = np.sum((x.dot(theta)-y) ** 2) / (2 * m)
        print("Iter %d | Cost: %f" % (i, cost))

        theta = theta - a * gradient
    return theta


def normalize_features(x):
    mean = x.mean(0)
    std = x.std(0, ddof = 1)
    x_norm = (x-mean) / std;
    return x_norm, mean, std

def denormalize_features(x, mean, std):
    x_real = x * std + mean
    return x_real