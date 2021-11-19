"""
Dynamic Solow Model
--------------------
Implementation of the general Solow model
"""

__author__ = "Karl Naumann-Woleske"
__version__ = "0.0.1"
__license__ = "MIT"

import pickle

import numpy as np
import pandas as pd
from cython_base.step_functions import general, supply
from scipy.optimize import minimize


class SolowModel(object):
    def __init__(self, params: dict, xi_args=None):
        """ Class for the dynamic Solow model"""

        # Parametrization
        if xi_args is None:
            xi_args = dict(decay=0.2, diffusion=1.0)
        self.params = params
        self.xi_args = xi_args
        # Arguments for later
        self.path, self.sbars, self.t_end, self.seed = None, None, None, None
        self.asymp_rates = None

    def simulate(self, initial_values: np.ndarray, t_end: float,
                 interval: float = 0.1, seed: int = 40,
                 case: str = 'general') -> pd.DataFrame:
        """ Simulation of the dynamic solow model, wrapping a Cython
        implementation of the main integration function. Maximal simulation size
        depends on the amount of RAM available.

        Parameters
        ----------
        initial_values  :   np.ndarray
            Order of variables: [y, ks, kd, s, h, switch, xi]
        t_end           :   float
            Duration of the simulation, recommend >1e6, 1e7 takes ~11 sec
        interval        :   float
            Integration interval, 0.1 by default
        seed            :   int
            Numpy random seed, 40 by default
        force_supply : bool (default False)
            Enforce the supply case k = ks

        Returns
        -------
        path    :   pd.DataFrame
            DataFrame of the integration path indexed by t

        """
        assert case in ['supply', 'demand', 'general'], "Unknown case"
        # Initialise and pre-compute random dW
        np.random.seed(seed)
        stoch = np.random.normal(0, 1, int(t_end / interval))
        # Initialise array for the results
        values = np.zeros((int(t_end), 7), dtype=float)
        values[0, :] = initial_values
        # Simulation via Cython function
        if case == 'supply':
            path = supply(interval, int(1 / interval), stoch, values,
                          **self.xi_args, **self.params)
        elif case == 'general':
            path = general(interval, int(1 / interval), stoch, values,
                           **self.xi_args, **self.params)
        # Output and save arguments
        self.seed = seed
        self.t_end = t_end
        columns = ['y', 'ks', 'kd', 's', 'h', 'switch', 'xi']
        self.path = pd.DataFrame(path, columns=columns)
        return self.path

    def asymptotics(self) -> list:
        """ Empirically calculate the asymptotic growth rates, and average
        sentiment levels

        Returns
        -------
        asymptotics :   list
            order of params [psi_y, psi_ks, psi_kd, sbar_hat]
        """
        assert self.path is not None, "Simulation run required first"

        # OLS estimate of the growth rates
        y = self.path.y.values[:, None]
        n = y.shape[0]
        x = np.hstack([np.ones((n, 1)), np.asarray(np.arange(n)[:, None])])
        beta_y = np.linalg.inv(x.T @ x) @ x.T @ y

        y = self.path.ks.values[:, None]
        n = y.shape[0]
        x = np.hstack([np.ones((n, 1)), np.asarray(np.arange(n)[:, None])])
        beta_ks = np.linalg.inv(x.T @ x) @ x.T @ y

        y = self.path.kd.values[:, None]
        n = y.shape[0]
        x = np.hstack([np.ones((n, 1)), np.asarray(np.arange(n)[:, None])])
        beta_kd = np.linalg.inv(x.T @ x) @ x.T @ y

        sbar = self.path.s.mean()
        return [beta_y[1,0], beta_ks[1,0], beta_kd[1,0], sbar]

    def save(self, item: str = 'model', folder: str = 'computations'):
        """ Export the model, or an aspect of the model, to a folder as a
        pickle object

        Parameters
        ----------
        item    :   str
            currently supported "model" for object, and "path" for dataframe
        folder  :   str
        """
        item = dict(model=[self, '.obj'], path=[self.path, '.df'])[item]
        file = open(self._name(folder) + item[1], 'wb')
        pickle.dump(item[0], file)
        file.close()

    def _name(self, folder: str = 'computations/') -> str:
        """ Auto-generate a filename for the model based on its parameters

        Parameters
        ----------
        folder  :str

        Returns
        -------
        name    :str
        """
        p = self.params
        name = '_'.join([
            'general', 't{:05.0e}'.format(self.t_end),
            'g{:05.0f}'.format(p['gamma']),
            'e{:07.1e}'.format(p['epsilon']),
            'c1_{:03.1f}'.format(p['c1']), 'c2_{:07.1e}'.format(p['c2']),
            'b1_{:03.1f}'.format(p['beta1']), 'b2_{:03.1f}'.format(p['beta2']),
            'ty{:03.0f}'.format(p['tau_y']), 'ts{:03.0f}'.format(p['tau_s']),
            'th{:02.0f}'.format(p['tau_h']),
            'lam{:01.2f}'.format(p['saving0']), 'dep{:07.1e}'.format(p['dep']),
            'tech{:04.2f}'.format(p['tech0']), 'rho{:04.2f}'.format(p['rho']),
        ])
        return folder + name

    def _s_sol(self) -> list:
        """ Add the solution to the sentiment equilibria on the basis of the
        asymptotic growth rates

        Returns
        -------
        sbars : list
        """
        # Check
        if self.asymp_rates is None:
            self.asymptotics()
        # Solve
        m = self.params['beta2'] * np.tanh(
                self.params['gamma'] * self.asymp_rates[0])
        y = lambda s: np.abs(np.arctanh(s) - self.params['beta1'] * s - m)
        intersects = []
        # Search through four quadrants
        for bnds in [(-0.99, -0.5), (-0.5, 0), (0, 0.5), (0.5, 0.99)]:
            temp = minimize(y, 0.5 * sum(bnds), bounds=(bnds,), tol=1e-15)
            intersects.append(temp.x[0])
        self.sbars = [max(intersects), min(intersects)]
        return [max(intersects), min(intersects)]
