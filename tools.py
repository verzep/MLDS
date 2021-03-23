import numpy as np


def _MFNN_t(X, Y, n):
    '''
    Compute the Mutual False Nearest Neighbors  for time intex n.
    The data should be given as matrices where the first dimension is time s.t X[t] is a point in space.

    :param X: A matrix with dimension (time_steps, X_space_dimension)
    :param Y: A matrix with dimension (time_steps, Y_space_dimension)
    :param n: The time index at which you want to compute the MFNN
    :return: the MFNN(n)
    '''

    x_n = X[n]
    y_n = Y[n]

    # find the NN of the drive
    n_NND = np.argpartition(np.linalg.norm(X - x_n, axis=1), 1)[1]
    x_n_NND = X[n_NND]
    y_n_NND = Y[n_NND]

    # find the NN of the responde
    n_NNR = np.argpartition(np.linalg.norm(Y - y_n, axis=1), 1)[1]
    x_n_NNR = X[n_NNR]
    y_n_NNR = Y[n_NNR]

    R = (np.linalg.norm(y_n - y_n_NND) * np.linalg.norm(x_n - x_n_NNR)
         / (np.linalg.norm(x_n - x_n_NND) * np.linalg.norm(y_n - y_n_NNR))
         )
    return R


def MFNN(X, Y, transient_length=None):
    '''
    Compute the Mutual False Nearest Neighbors doing the temporal average.

    :param X: A matrix with dimension ( X_space_dimension, time_steps)
    :param Y: A matrix with dimension ( Y_space_dimension, time_steps)
    :param transient_length: the number of initial point to discard. If None, the 10% will be discarded.
    :return: MFNN
    '''

    data = []
    stop = X.shape[1]
    if transient_length is None:
        start = stop // 10
    else:
        start = transient_length

    for i in range(start, stop):
        # Note that `_MFNN_t` uses the transpose data matrix!!!
        data.append(_MFNN_t(X.T, Y.T, i))

    return data
