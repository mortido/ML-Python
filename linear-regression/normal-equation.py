#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    raw_data = np.loadtxt(open("Concrete_Data.csv", "rb"), delimiter=",", skiprows=1)
    X = raw_data[:, :-1]
    Y = raw_data[:, -1:]

    # get data set examples count
    m = X.shape[0]
    m_training = int(2 * m / 3)
    m_test = m - m_training

    # training set
    X_training = np.hstack((np.ones((m_training, 1)), X[:m_training]))
    Y_training = Y[:m_training]

    # test set
    X_test = np.hstack((np.ones((m_test, 1)), X[m_training:]))
    Y_test = Y[m_training:]

    # features count
    n = X.shape[1]

    # find theta
    theta = np.linalg.inv(X_training.T.dot(X_training)).dot(X_training.T).dot(Y_training)

    # check results
    J = np.sum((X_test.dot(theta)-Y_test)**2)/(2*m_test)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # visual result check
    ax.scatter(X_training[:, 1],
               X_training[:, 6],
               Y_training,
               marker='^',
               color='red',
               s=10,
               alpha=0.85,
               label='real data')

    ax.scatter(X_training[:, 1],
               X_training[:, 6],
               X_training.dot(theta),
               marker='x',
               color='green',
               s=10,
               alpha=0.85,
               label='prediction')

    ax.set_zlabel('Strength')
    plt.title('Concrete data set')
    plt.show()
