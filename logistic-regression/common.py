import numpy as np


def generate_simple_data():
    n = 200
    X = np.random.uniform(0.5, 4.5, (n, 2))
    idx = 1.5 * X[:, 0] >= X[:, 1] + 2 + np.random.uniform(-0.5, 0.5, n)
    Y = np.zeros(n)
    Y[idx] = 1
    return X, Y


def generate_complex_data():
    n = 200
    X = np.random.uniform(-4, 4, (n, 2))
    R = np.cumsum(np.random.uniform(-0.2, 0.2, n))
    idx = (0.7 * (X[:, 0] - 0)) ** 2 + (0.5 * (X[:, 1] - 0)) ** 2 <= 2.5 + R
    Y = np.zeros(n)
    Y[idx] = 1
    return X, Y


def cartesian(arrays, out=None):
    """
    SOURCE: http://stackoverflow.com/a/1235363

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out
