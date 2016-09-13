#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import common as cm


if __name__ == "__main__":
    # SYNTHETIC DATA
    print("Synthetic data test: 1.5*x1-x2-2>=0")

    x1, x2 = cm.simple_function()
    X, Y = cm.generate_simple_data()
    pos = Y == 1
    neg = Y == 0

    real_line, = plt.plot(x1, x2, 'g', linewidth=2.0)
    first_class_markets, = plt.plot(X[pos, 0], X[pos, 1], 'ro', alpha=0.7)
    second_class_markets, = plt.plot(X[neg, 0], X[neg, 1], 'b^', alpha=0.7)
    plt.axis((0, 5, 0, 6))
    plt.title("Synthetic data")
    plt.legend([real_line, first_class_markets, second_class_markets], ['Real', 'First class', 'Second class'])
    plt.show()
