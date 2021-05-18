import numpy as np
import matplotlib.pyplot as plt

import mapsList


class Map:

    def __init__(self, F, dimension: int, param, name: str):
        ''' A class for maps.

        A map is a discrete-time dynamical system.
        It is characterized by a function F: X -> X, where X is the space in which the map acts.

        Parameters
        ----------
        F:
            The function characterizing the map such that F(x(t)) = x(t+1)
        dimension:
            the dimensionality of the space X
        param
            the parameters of the map
        name: string
            The name of the map
        '''

        self.dimension = dimension
        self.param = param
        self.F = F
        self.name = name
        self.n_iter = None

    def trajectory(self, x_0: np.array, n_iter: int) -> np.array:
        '''
        return a trajectory of the map.

        Parameters
        ----------
        x_0 : np.array[dimension]
            An arraywith the initial conditions of the system.
            Its dimension must be the same of the map.
        n_iter: int
            the number of iteration that the map has to do.

        Returns
        -------
        X : np.array
            np.array with dimensions (dimension, n_iter+1)
        '''
        self.n_iter = n_iter

        X = np.zeros((self.dimension, self.n_iter))
        X[:, 0] = x_0
        x = x_0

        for k in range(1, n_iter):
            y = self.F(x, self.param)
            X[:, k] = y
            x = y

        self.X = X
        return X

    def _repeated(self, x, n: int):
        ''' Applies the map 'n' times to a point x.
        This means computing F^n(x)

        Parameters
        ----------
        x
            The point starting from which the map is iterated
        n
            The number of iteration
        Returns
        -------
        F^n(x), i.e., the map iterated n times

        '''
        for _ in range(n):
            x = self.F(x, self.param)
            # print x
        return x

    def plot_map(self, n_iter=1, x_min=0, x_max=1):
        ''' A function plotting the graph of the map.

        The map must be 1-dimensional!
        The x-axis represent x(t) while the y-axis is x(t+1) = F(x(t)).
        It can be used to plot multiple iteration of a map, i.e., plotting G(x) = F^n(x).


        Parameters
        ----------
        n_iter : int, default=1
            number of iteration of the map.
        x_min,x_max : int, default = 0,1
            The min and max value to be visualized, for both axis.
        '''
        x = np.arange(x_min, x_max, 0.0001)
        Fig = plt.figure(figsize=(5, 5))
        ax = Fig.add_subplot(111)

        ax.plot(x, self._repeated(x, n_iter))
        ax.plot(x, x, 'k')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)

        ax.set_xlabel('x[n]')
        ax.set_ylabel('x[n+1]')

        if n_iter == 1:
            ax.set_title('{} map'.format(self.name, n_iter), fontsize=16)
        else:
            ax.set_title('{} map iterated {} times'.format(self.name, n_iter), fontsize=16)

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------


class Logistic(Map):
    def __init__(self, r=3.5):
        ''' Logistic map F(x) = r*x*(1-x)

        Parameters
        ----------
        r : float, default = 3.5
            the value of the parameter r of the map.
        '''
        super().__init__(mapsList.logistic, 1, r, "Logistic")


# -----------------------------------------------------------
class Tent(Map):
    def __init__(self, r=2):
        ''' Tent map  F(x) = r*min(x, 1-x)
        Parameters
        ----------
        r : float, default = 2
            the value of the parameter r of the map.
        '''
        super().__init__(mapsList.tent, 1, r, "Tent")


# ---------------------------------------------------------------

class Shift(Map):
    def __init__(self, r=1):
        ''' Shift map F(x) = r*( 2*x mod(1) )

        Parameters
        ----------
        r: float, default = 1
            the value of the parameter r of the map.
        '''
        super().__init__(mapsList.shift, 1, r, "Shift")


# -------------------------------------------------


class Quadratic(Map):
    def __init__(self, r=2):
        ''' Quadratic map F(x) = x**2 - r

        Parameters
        ----------
        r: float, default = 2
            the value of the parameter r of the map.
        '''
        super().__init__(mapsList.quadratic, 1, r, "Quadratic")
