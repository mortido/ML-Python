#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import common as cm
import gradient_descent as gd
import normal_equation as ne

GD_ITERATIONS = 400
GD_ALPHA = 0.03


if __name__ == "__main__":
    # CONCRETE DATA
    print("Concrete data set test:")
    X, Y = cm.load_concrete_data()

    # training set
    X_training, Y_training = cm.get_training_set(X, Y)
    X_training = cm.add_ones_column(X_training)

    # test set
    X_test, Y_test = cm.get_test_set(X, Y)
    X_test = cm.add_ones_column(X_test)

    # find thetas
    ne_theta = ne.normal_equation(X_training, Y_training)
    gd_theta = gd.gradient_descent(X_training, Y_training, GD_ALPHA, GD_ITERATIONS)

    print("[Normal equation] theta = ", ne_theta)
    print("[Gradient descent] theta = ", gd_theta)

    # check results
    ne_prediction = X_test.dot(ne_theta)
    gd_prediction = X_test.dot(gd_theta)

    exp, var = cm.check_result(Y_test, ne_prediction)
    print("[Normal equation] Expected value (error): ", exp)
    print("[Normal equation] Variance: ", var)

    exp, var = cm.check_result(Y_test, gd_prediction)
    print("[Gradient descent] Expected value (error): ", exp)
    print("[Gradient descent] Variance: ", var)
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # try visual result check (no so impressive)
    ax.scatter(X_test[:, 1],
               X_test[:, 6],
               Y_test,
               marker='^',
               color='green',
               s=10,
               alpha=0.85,
               label='Real data')

    ax.scatter(X_test[:, 1],
               X_test[:, 6],
               ne_prediction,
               marker='x',
               color='red',
               s=10,
               alpha=0.85,
               label='NE prediction')

    ax.scatter(X_test[:, 1],
               X_test[:, 6],
               gd_prediction,
               marker='x',
               color='blue',
               s=10,
               alpha=0.85,
               label='GD prediction')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    ax.set_xlabel('Cement (component 1)')
    ax.set_ylabel('Coarse Aggregate (component 6)')
    ax.set_zlabel('Concrete compressive strength')
    plt.title('Concrete data set (2d projection)')
    plt.show()'''

    # SYNTHETIC DATA
    print("Synthetic data test: f(x) = -0.5x + 3.5")
    theta_syn = np.array([3.5, -0.5]).reshape((-1, 1))
    X = np.linspace(0, 8, 200).reshape((-1, 1))
    X = cm.add_ones_column(X)
    Y_real = X.dot(theta_syn)
    Y = Y_real + np.random.uniform(-2, 2, X.shape[0]).reshape((-1, 1))

    ne_theta = ne.normal_equation(X, Y)
    X_norm = np.ones(X.shape)
    X_norm[:, 1:], mean, std = gd.normalize_features(X[:, 1:])
    gd_theta = gd.gradient_descent(X, Y, 0.1, 25)
    gd_theta.


    ne_prediction = X.dot(ne_theta)
    gd_prediction = X.dot(gd_theta)

    exp, var = cm.check_result(Y, ne_prediction)
    print("[Normal equation] Expected value (error): ", exp)
    print("[Normal equation] Variance: ", var)

    exp, var = cm.check_result(Y, gd_prediction)
    print("[Gradient descent] Expected value (error): ", exp)
    print("[Gradient descent] Variance: ", var)

    plt.plot(X[:, 1], Y, 'go', alpha=0.7)
    real_line, = plt.plot(X[:, 1], Y_real, 'g', linewidth=2.0)
    ne_line, = plt.plot(X[:, 1], ne_prediction, 'r', linewidth=2.0)
    gd_line, = plt.plot(X[:, 1], gd_prediction, 'b', linewidth=2.0)
    plt.legend([real_line, ne_line, gd_line], ['Real', 'Normal equation', 'Gradient descent'])
    plt.show()
