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
    J = np.sum(((X_test.dot(theta)-Y_test)**2))/(2*m_test)
    #J2 = np.sum(((X_training.dot(theta)-Y_training)**2))/(2*m_training)
    #plt.plot(X[:,2],X[:,0],Y, 'ro')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(X[:,2],X[:,0],zs=Y.flat, zdir='z', label='zs=0, zdir=z')
    plt.show()
    v = 1
