import numpy as np


class LogisticRegression(object):
    """
    Class contains logistic regression implementation
    """

    def __init__(self, X, Y, feature_mapping_degree=1):
        self.m = X.shape[0]

        # normalize
        self.x_mean = X.mean(0)
        self.x_std = X.std(0, ddof=1)
        x = (X - self.x_mean) / self.x_std

        # apply feature mapping if required
        self.feature_mapping_degree = feature_mapping_degree
        x = self.__map_features(x)

        # add ones column
        ones = np.ones((self.m, 1))
        self.X = np.hstack((ones, x))

        self.Y = np.array(Y)
        self.theta = np.zeros((self.X.shape[1], 1))

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

        # NASTY HACK!!!!
        s1 = sig == 1.
        s0 = sig == 0.
        sig[s1] = 0.9999999
        sig[s0] = 0.0000001

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

    def __combinations(self, cur_n, max_n, cur_i, max_i):
        out = []
        for i in range(cur_n, max_n):
            next_combs = [[]] if cur_i == max_i else self.__combinations(i, max_n, cur_i + 1, max_i)
            for comb in next_combs:
                out.append([i] + comb)
        return out

    def __feature_combinations(self, n):
        out = []
        for i in range(self.feature_mapping_degree):
            out += self.__combinations(0, n, 0, i)
        return out

    def __map_features(self, X):
        """
        Feature mapping for nonlinear solution.
        """

        if self.feature_mapping_degree > 1:
            combinations = self.__feature_combinations(X.shape[1])
            XN = np.empty((X.shape[0], len(combinations)))
            for i, c in enumerate(combinations):
                XN[:, i] = X[:, c].prod(1)
            return XN
        else:
            return X

    def run(self, max_iter, alpha, lam, print_log=False):
        """
        Runs logistical regression
        """
        self.__gradient_descent(max_iter, alpha, lam, print_log)

    def predict_raw(self, X):
        """
        Predicts Y value without adjusting to 0, 1
        """
        # normalize
        x = (X - self.x_mean) / self.x_std

        # apply feature mapping if required
        x = self.__map_features(x)

        # add ones column
        ones = np.ones((X.shape[0], 1))
        x = np.hstack((ones, x))

        prediction = self.__sigmoid(x.dot(self.theta))
        return prediction

    def predict(self, X):
        """
        Predicts output with calculated theta value
        """

        prediction = self.predict_raw(X)
        np.putmask(prediction, prediction >= 0.5, 1.0)
        np.putmask(prediction, prediction < 0.5, 0.0)

        return prediction
