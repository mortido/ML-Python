import numpy as np


class LogisticRegression(object):
    """
    Class contains logistic regression implementation
    """

    def __init__(self, X, Y):
        m, n = X.shape
        self.m = m

        # normalize
        self.x_mean = X.mean(0)
        self.x_std = X.std(0, ddof=1)
        x = (X - self.x_mean) / self.x_std

        # add ones column
        ones = np.ones((m, 1))
        self.X = np.hstack((ones, x))

        self.Y = np.array(Y)
        self.theta = np.zeros((n + 1, 1))

    def __sigmoid(self, X):
        """
        Computes the sigmoid function
        """

        return 1 / (1 + np.exp(-X))

    def __cost(self, lam):
        """
        Calculates cost function for logistic regression
        """

        prediction = self.X.dot(self.theta)
        sig = self.__sigmoid(prediction)
        cost1 = (-self.Y).T.dot(np.log(sig))
        cost2 = (1.0 - self.Y).T.dot(np.log(1 - sig))
        cost = cost1 - cost2
        if lam:
            cost += 0.5 * lam * np.sum(self.theta[1:] ** 2)

        return cost / self.m

    def __gradient_descent(self, max_iter, alpha, lam, print_log):
        """
        Executes gradient descent optimization to find best suited theta
        """

        for i in range(max_iter):
            prediction = self.X.dot(self.theta)
            sig = self.__sigmoid(prediction)
            grad = self.X.T.dot(sig - self.Y)
            if lam:
                grad[1:] += lam * self.theta[1:]
            grad = grad / self.m
            self.theta -= alpha * grad

            if print_log:
                print("Iteration", i, "\tcost=", self.__cost(lam))

    def run(self, max_iter, alpha, lam, print_log=False):
        """
        Runs logistical regression
        """
        self.__gradient_descent(max_iter, alpha, lam, print_log)

    def predict(self, X):
        """
        Predicts output with calculated theta value
        """

        # normalize
        x = (X - self.x_mean) / self.x_std

        # add ones column
        ones = np.ones((X.shape[0], 1))
        x = np.hstack((ones, x))

        prediction = self.__sigmoid(x.dot(self.theta))
        np.putmask(prediction, prediction >= 0.5, 1.0)
        np.putmask(prediction, prediction < 0.5, 0.0)

        return prediction
