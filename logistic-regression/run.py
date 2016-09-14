#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import common as cm
import gradient_descent as gd
from logistic_regression import LogisticRegression

if __name__ == "__main__":
    # SYNTHETIC DATA
    print("Synthetic data test: 1.5*x1-x2-2>=0")

    X, Y = cm.generate_simple_data()
    pos = Y == 1
    neg = Y == 0

    first_class_markets, = plt.plot(X[pos, 0], X[pos, 1], 'ro', alpha=0.7)
    second_class_markets, = plt.plot(X[neg, 0], X[neg, 1], 'b^', alpha=0.7)
    plt.axis((0, 5, 0, 6))
    plt.title("Synthetic data")

    lr = LogisticRegression(X, Y.reshape((-1, 1)))
    lr.run(100, 1, 0, print_log=True)
    plot_x = np.array([0.0, 5.0])
    plot_x_norm = (plot_x - lr.x_mean[0]) / lr.x_std[0]
    plot_y_norm = -(lr.theta[1] * plot_x_norm + lr.theta[0]) / lr.theta[2]
    plot_y = plot_y_norm * lr.x_std[1]+lr.x_mean[1]

    trained_line, = plt.plot(plot_x, plot_y, 'y', linewidth=2.0)
    #x1, x2 = cm.simple_function()
    #real_line, = plt.plot(x1, x2, 'g', linewidth=2.0)

    plt.legend([first_class_markets, second_class_markets, trained_line],
               ['First class', 'Second class', 'Logistic regression'])
    plt.show()