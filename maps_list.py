import numpy as np


def logistic(x, r):
    return r * x * (1 - x)


def tent(x, r):
    m = np.min([x, 1. - x], 0)
    return r * m


def shift(x, r):
    return r * (2. * x % 1.)


def quadratic(x, r):
    return x ** 2 - r
