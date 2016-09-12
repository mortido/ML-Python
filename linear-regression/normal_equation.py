#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def normal_equation(x, y):
    # return theta
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


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


if __name__ == "__main__":
    # CONCRETE DATA
    print("Concrete data set test:")
    raw_data = np.loadtxt(open("Concrete_Data.csv", "rb"), delimiter=",", skiprows=1)
    X = raw_data[:, :-1]
    Y = raw_data[:, -1:]

    # get data set examples count
    m_training = int(2 * X.shape[0] / 3)

    # training set
    X_training = add_ones_column(X[:m_training])
    Y_training = Y[:m_training]

    # test set
    X_test = add_ones_column(X[m_training:])
    Y_test = Y[m_training:]

    # find theta
    theta = normal_equation(X_training, Y_training)

    print("theta = ")

    # check results
    exp1, var1 = check_result(Y_test, X_test.dot(theta))
    print("Expected value (error): ", exp1)
    print("Variance: ", var1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # try visual result check (no so impressive)
    ax.scatter(X_training[:, 1],
               X_training[:, 6],
               Y_training,
               marker='^',
               color='green',
               s=10,
               alpha=0.85,
               label='real data')

    ax.scatter(X_training[:, 1],
               X_training[:, 6],
               X_training.dot(theta),
               marker='x',
               color='red',
               s=10,
               alpha=0.85,
               label='prediction')

    ax.set_zlabel('Strength')
    plt.title('Concrete data set')
    plt.show()

    # SYNTHETIC DATA
    print("Synthetic data test: f(x) = -0.5x + 3.5")

    def func(x):
        return -0.5 * x + 3.5

    v_func = np.vectorize(func)
    X = np.linspace(0, 8, 200)
    Y_real = v_func(X)
    Y = Y_real + np.random.uniform(-2, 2, X.shape[0])
    Xt = add_ones_column(X.reshape(-1,1))
    theta = normal_equation(Xt, Y)
    print("theta = ", theta)
    Y_prediction = Xt.dot(theta)

    exp2, var2 = check_result(Y, Y_prediction)
    print("Expected value (error): ", exp2)
    print("Variance: ", var2)

    plt.plot(X, Y, 'go', alpha=0.7)
    plt.plot(X, Y_real, 'g', linewidth=2.0)
    plt.plot(X, Y_prediction, 'r', linewidth=2.0)
    plt.show()
