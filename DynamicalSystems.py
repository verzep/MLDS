import numpy as np
import scipy.integrate as spi

import systems_list


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
    def __init__(self, param):
        super(Lorenz, self).__init__(f=systems_list.lorenz, dimension=3, param=param, name="Lorenz")


# -----------------------------------------

class Rossler(DynSys):
    def __init__(self, param):
        super(Rossler, self).__init__(f=systems_list.rossler, dimension=3, param=param, name="Rossler")


# ---------------------------------------

class VanDerPol(DynSys):
    def __init__(self, param):
        super(VanDerPol, self).__init__(f=systems_list.vanderpol, dimension=2, param=param,
                                        name="Van der Pol")


# -----------------------------------------
class LotkaVolterra(DynSys):
    def __init__(self, param):
        super(LotkaVolterra, self).__init__(f=systems_list.lotkavolterra, dimension=2, param=param,
                                            name="Lotka-Volterra")


# --------------------------
class SIR(DynSys):
    def __init__(self, param):
        super(SIR, self).__init__(f=systems_list.SIR, dimension=2, param=param, name="SIR")
