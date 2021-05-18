import numpy as np
import scipy.integrate as spi

import systemsList


class DynSys:
    def __init__(self, f, dimension: int, param, name: str):
        self.f = f
        self.dimension = dimension
        self.param = param
        self.name = name
        self.X = None
        self.times = None

    def trajectory(self, x_0, t_final, dt, t_0=0):
        t_eval = np.arange(t_0, t_final, dt)
        sol = spi.solve_ivp(fun=self.f, t_span=[t_0, t_final], y0=x_0, t_eval=t_eval, args=self.param,
                            # method = 'LSODA' # TO DO!!!!!!!!!!!!!!!!
                            rtol=10 ** (-12), atol=10 ** (-12) * np.ones_like((1, 1, 1))
                            )

        self.X = sol.y
        self.times = sol.t

        return sol.y


# -------------------------------------------
class Lorenz(DynSys):
    def __init__(self, param = (10,8/3,28)):
        super(Lorenz, self).__init__(f=systemsList.lorenz, dimension=3, param=param, name="Lorenz")


# -----------------------------------------

class Rossler(DynSys):
    def __init__(self, param = (0.1,0.1, 14)):
        super(Rossler, self).__init__(f=systemsList.rossler, dimension=3, param=param, name="Rossler")


# ---------------------------------------

class VanDerPol(DynSys):
    def __init__(self, param = (8.53, 1.2, 2 * 3.1415 / 10)):
        super(VanDerPol, self).__init__(f=systemsList.vanderpol, dimension=2, param=param,
                                        name="Van der Pol")


# -----------------------------------------
class LotkaVolterra(DynSys):
    def __init__(self, param= (1.5,1,3,1)):
        super(LotkaVolterra, self).__init__(f=systemsList.lotkavolterra, dimension=2, param=param,
                                            name="Lotka-Volterra")


# --------------------------
class SIR(DynSys):
    def __init__(self, param=(1.5,1)):
        super(SIR, self).__init__(f=systemsList.SIR, dimension=2, param=param, name="SIR")
