import numpy as np

MAX_ITERATIONS = 10000


class NormalizationParams:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

def calculate_sigmoid(X):
    '''
    Computes the sigmoid function
    '''

    return 1/(1-np.exp(-X))


def calculate_cost(theta, X, Y):
    '''
    Calculates cost function for logistic regression.
    '''

    m = X.shape[0]
    prediction = X.dot(theta)
    sig = calculate_sigmoid(prediction)
    return -Y.dot(np.log(sig))/m-(1-Y).dot(np.log(1-sig))



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