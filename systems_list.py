import numpy as np


def lorenz(t, X, sigma=10, beta=8 / 3, rho=28):
    """
    The Lorenz-63 system equations

    :param t: the time (just for consistency)
    :param X: the x,y and z coordinates of the Lorenz System
    :param sigma, beta, rho: the parameters of the Lorenz System
    :return: Xdot (as xdot,ydot,zdot)
    """
    x, y, z = X

    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z

    return [xdot, ydot, zdot]


def rossler(t, X, a=0.1, b=0.1, c=14):
    """
    The Rossler system equation

    :param t: the time (just for consistency)
    :param X: the x,y and z coordinates of the Rossler System
    :param a,b,c: the parameters of the Lorenz System
    :return: Xdot (as xdot,ydot,zdot)
    """

    x, y, z = X

    xdot = -y - z
    ydot = x + a * y
    zdot = b + z * (x - c)

    return [xdot, ydot, zdot]


def vanderpol(t, X, mu=8.53, A=1.2, omega=2 * 3.1415 / 10):
    '''
    The van del pol system equation

    :param t: the time (just for consistency)
    :param X: the x and v coordinates of the Van der Pol System
    :param mu, A, omega: the parameters Van der Pol System
    :return: Xdot (as xdot,vdot)
    '''
    x, v = X

    xdot = v
    vdot = mu * (1 - x ** 2) * v - x + A * np.sin(omega * t)

    return [xdot, vdot]


def lotkavolterra(t, X, a=1.5, b=1, c=3, d=1):
    '''
    The Lotka-Volterra system equation

    :param t: the time (just for consistency)
    :param X: the x and y coordinates of the Lotka-Volterra  System
    :param a, b, c, d: the parameters of the Lotka-Volterra  System
    :return: Xdot (as xdot,ydot)
    '''

    x, y = X
    return [a * x - b * x * y, -c * y + d * x * y]


def lorenz_driver(t, X, sigma=10, beta=8 / 3, rho=28, beta_p=8 / 3, rho_p=28):
    '''

    The Lorenz-63 system equations and a replica system which uses x as a drive.

    :param t: the time (just for consistency)
    :param X: the x,y and z coordinates of the Lorenz System and the y_p,z_p coordinate of the replica
    :param sigma, beta, rho, beta_p, rho_p: the parameters of the Lorenz System and its replica
    :return: Xdot (as xdot,ydot,zdot, y_pdot, z_pdot)
    '''

    x, y, z, y_p, z_p = X

    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z

    # the replica has the same x-coordinate as the original system
    y_pdot = x * (rho_p - z_p) - y_p
    z_pdot = x * y_p - beta_p * z_p

    return [xdot, ydot, zdot, y_pdot, z_pdot]


def lorenz_bidirectional(t, X, c=.1, sigma=10, beta=8 / 3, rho=28, sigma_p=10, beta_p=8 / 3, rho_p=28):
    '''
    Two Lorenz-63 system are coupled

    :param t: the time (just for consistency)
    :param X: the x,y and z coordinates of the Lorenz System and the x_p, y_p,z_p coordinate of the replica
    :param c: the coupling strength
    :param sigma, beta, rho, sigma_p,  beta_p, rho_p: the parameters of the Lorenz System and its replica
    :return: Xdot (as xdot,ydot,zdot, x_p, y_pdot, z_pdot)
    '''

    x, y, z, x_p, y_p, z_p = X

    xdot = sigma * (y - x) + c * (x_p - x)
    ydot = x * (rho - z) - y + c * (y_p - y)
    zdot = x * y - beta * z + c * (z_p - z)

    x_pdot = sigma_p * (y_p - x_p) - c * (x_p - x)
    y_pdot = x_p * (rho_p - z_p) - y_p - c * (y_p - y)
    z_pdot = x_p * y_p - beta_p * z_p - c * (z_p - z)

    return [xdot, ydot, zdot, x_pdot, y_pdot, z_pdot]


def SIR(t, X, beta=1.5, gamma=1):
    '''
    The SIR model system equation

    :param t: the time (just for consistency)
    :param X: the s, i and r coordinates of the system
    :param beta, gamma: the parameters of the SIR system
    :return: Xdot (as sdot,idot,rdot)
    '''

    s, i, r = X

    sdot = - beta * i * (1 - r - i)
    idot = - gamma * i + beta * i * (1 - r - i)
    rdot = gamma * i

    return [sdot, idot, rdot]
