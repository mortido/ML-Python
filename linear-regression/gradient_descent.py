import numpy as np

MAX_ITERATIONS = 10000


def gradient_descent(x, y, a, epsilon):
    theta = np.zeros((x.shape[1], 1))
    m = x.shape[0]
    prev_cost = np.sum((x.dot(theta)-y) ** 2) / (2 * m)
    print("Start cost: %f" % prev_cost)
    for i in range(0, MAX_ITERATIONS):
        gradient = x.T.dot(x.dot(theta)-y) / m
        theta = theta - a * gradient
        cost = np.sum((x.dot(theta)-y) ** 2) / (2 * m)
        print("Iter %d | Cost: %f" % (i, cost))
        if abs(prev_cost - cost) <= epsilon:
            return theta
        prev_cost = cost
    return theta


def normalize_features(x, x_mean=None, x_range=None):
    if x_mean is None:
        x_mean = x.mean(0)
    if x_range is None:
        x_range = x.max(0)-x.min(0)

    x_norm = (x-x_mean) / x_range
    return x_norm, x_mean, x_range