#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


import common as cm


def normal_equation(x, y):
    # return theta
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


if __name__ == "__main__":
    # CONCRETE DATA
    print("Concrete data set test:")
    raw_data = np.loadtxt(open("Concrete_Data.csv", "rb"), delimiter=",", skiprows=1)
    X = raw_data[:, :-1]
    Y = raw_data[:, -1:]

    # get data set examples count
    m_training = int(2 * X.shape[0] / 3)

    # training set
    X_training = cm.add_ones_column(X[:m_training])
    Y_training = Y[:m_training]

    # test set
    X_test = cm.add_ones_column(X[m_training:])
    Y_test = Y[m_training:]

    # find theta
    theta = normal_equation(X_training, Y_training)

    print("theta = ", theta)

    # check results
    exp1, var1 = cm.check_result(Y_test, X_test.dot(theta))
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
    theta_syn = np.array([3.5, -0.5]).reshape((-1, 1))
    X = np.linspace(0, 8, 200).reshape((-1, 1))
    X = cm.add_ones_column(X)
    Y_real = X.dot(theta_syn)
    Y = Y_real + np.random.uniform(-2, 2, X.shape[0]).reshape((-1, 1))

    theta = normal_equation(X, Y)
    print("theta = ", theta)

    Y_prediction = X.dot(theta)
    exp2, var2 = cm.check_result(Y, Y_prediction)
    print("Expected value (error): ", exp2)
    print("Variance: ", var2)

    plt.plot(X[:, 1], Y, 'go', alpha=0.7)
    plt.plot(X[:, 1], Y_real, 'g', linewidth=2.0)
    plt.plot(X[:, 1], Y_prediction, 'r', linewidth=2.0)
    plt.show()
