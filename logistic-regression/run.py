#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import common as cm
from logistic_regression import LogisticRegression


def draw_markets(X, Y):
    pos = Y == 1
    neg = Y == 0
    markets1, = plt.plot(X[pos, 0], X[pos, 1], 'ro', alpha=0.7)
    markets2, = plt.plot(X[neg, 0], X[neg, 1], 'b^', alpha=0.7)
    return markets1, markets2

if __name__ == "__main__":
    # SYNTHETIC DATA
    print("Synthetic data test: 1.5*x1-x2-2>=0")

    X, Y = cm.generate_simple_data()

    plt.figure(1)
    plt.subplot(121)
    m1, m2 = draw_markets(X, Y)
    plt.axis((0, 5, 0, 6))
    plt.title("Simple Synthetic data")

    lr = LogisticRegression(X, Y.reshape((-1, 1)))
    lr.run(100, 1, 0)

    # Draw boundary line
    plot_x = np.array([0.0, 5.0])
    plot_x_norm = (plot_x - lr.x_mean[0]) / lr.x_std[0]
    plot_y_norm = -(lr.theta[1] * plot_x_norm + lr.theta[0]) / lr.theta[2]
    plot_y = plot_y_norm * lr.x_std[1]+lr.x_mean[1]
    boundary_line, = plt.plot(plot_x, plot_y, 'y', linewidth=2.0)

    plt.legend([m1, m2, boundary_line],
               ['First class', 'Second class', 'Logistic regression'])

    plt.subplot(122)

    # plt.axis((0, 10, 0, 12))
    plt.title("Complex Synthetic data\nRed - Under / Black - Over / Green - OK")

    X, Y = cm.generate_complex_data()
    m1, m2 = draw_markets(X, Y)

    lr_over = LogisticRegression(X, Y.reshape((-1, 1)), feature_mapping_degree=6)
    lr_under = LogisticRegression(X, Y.reshape((-1, 1)), feature_mapping_degree=6)
    lr = LogisticRegression(X, Y.reshape((-1, 1)), feature_mapping_degree=6)
    iterations = 500
    alpha = 3
    lr_over.run(iterations, alpha, 0.)
    lr_under.run(iterations, alpha, 30.)
    lr.run(iterations, alpha, 1.)

    # Draw boundary lines
    n = 500
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)

    X = cm.cartesian((x1, x2))

    Y = lr.predict_raw(X)
    plt.contour(x1, x2, Y.reshape((n, n)), (0.5,), colors="green")
    Y = lr_under.predict_raw(X)
    plt.contour(x1, x2, Y.reshape((n, n)), (0.5,), colors="red")
    Y = lr_over.predict_raw(X)
    plt.contour(x1, x2, Y.reshape((n, n)), (0.5,), colors="black")

    plt.show()

    print("haberman.data.txt")
    # Data rows were randomly ordered before use.
    data = np.loadtxt("haberman.data.txt", delimiter=",")
    Y = data[:, -1]
    Y -= 1
    Y.shape = (-1, 1)
    X = data[:, :-1]
    m_training = int(2 * X.shape[0] / 3)
    lr = LogisticRegression(X[:m_training, :], Y[:m_training, :], feature_mapping_degree=6)
    lr.run(500, .003, 1., print_log=True)
    Y_test = lr.predict(X[m_training:, :])
    test = Y_test == Y[m_training:, :]
    result = (np.sum(np.ones(test.shape)[test]) / test.shape[0]) * 100
    print("Result accuracy: {0:.2f}%".format(result))
