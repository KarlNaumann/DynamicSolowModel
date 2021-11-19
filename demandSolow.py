"""
Demand Solow Class
--------------------------
Class representing the demand limiting case for the Dynamic Solow Model
presented in the paper "Capital Demand Driven Business Cycles: Mechanism and
Effects" by Naumann-Woleske et al. 2021.
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from numdifftools import Jacobian
from scipy.optimize import minimize

from cython_base.step_functions import demand


class DemandSolow(object):
    def __init__(self, params: dict, xi_args: dict):
        """Implementation of the demand limiting case, where capital is set to
        capital demand across time.

        Parameters
        -----------
        params  :   dict
            dictionary of parameters, must contain [tech0, epsilon, rho, tau_y,
            tau_s, tau_h, c1, c2, b1, b2, gamma
        xi_args :   dict
            dictionary of parameters for the Ornstein-Uhlenbeck process

        Methods
        -------
        simulate()      :   solve initial value problem
        visualise_y(i)  :   matplotlib plot of production

        """
        self.params = params
        self.xi_args = xi_args
        self.path = None

        self._crit_point_info = None

    def simulate(self, initial_values: np.ndarray, interval: float = 0.1,
                 t_end: float = 1e5, xi: bool = True,
                 seed: int = 42) -> pd.DataFrame:
        """ Simulating the classic Solow model using Scipy with Runge-Kute
        4th order

        Parameters
        ----------
        initial_values : np.ndarray
            order is [y, z, kd, s, h, xi]
        interval : float (default 0.1)
            interval across which the noise is taken, generally 0.1
        t_end : float (default 1e5)
            duration of the simulation
        xi : bool (default True)
            Whether to use noise or consider the static case
        seed : float (default 42)
            Random seed for the simulation

        Returns
        -------
        path    :   pd.DataFrame
        """
        # Initialise and pre-compute random noise process
        np.random.seed(seed)
        if xi:
            stoch = np.random.normal(0, 1, int(t_end / interval))
        else:
            stoch = np.zeros(int(t_end / interval))

        # Initialise array for the results
        values = np.zeros((int(t_end), 7), dtype=float)
        values[0, :6] = initial_values[:6]
        if len(initial_values) == 6:
            # Allow for backward compatibility of the code (no ks case)
            values[0, 6] = values[0, 2]
        else:
            values[0, 6] = initial_values[-1]

        count = int(1 / interval)
        # Simulation via Cython function
        path = demand(interval, count, stoch, values,
                      **self.params, **self.xi_args)
        # Output and save arguments
        self.seed = seed
        self.t_end = t_end
        columns = ['y', 'z', 'kd', 's', 'h', 'xi', 'ks']
        self.path = pd.DataFrame(path, columns=columns)
        return self.path

    def get_critical_points(self) -> dict:
        """ Obtain the information about the critical points in the model that
        includes the coordinates in (s,h,z) space, classification, eigenvalues,
        and eigenvectors of the Jacobian at that point.

        Returns
        -------
        critical_points : dict
            dictionary of (s,h,z) keys and information for each point
        """
        x = self._critical_points()
        self._crit_point_info = self._point_classification(x)
        return self._crit_point_info

    def sh_phase(self, count: int = 9, ax=None, save: str = ''):
        """ Generate a phase diagram in the (s,h)-plane that shows the focii,
        and some sample trajectories.

        Parameters
        ----------
        count : int
            number of different trajectories to start from each axis. Ideally an
            odd number
        ax : matplotlib axes object (default None)
            pass an ax object to plot the phase diagram on
        save : str (default '')
            filename where to save the output. Will save as a .pdf. If '' then
            will show the plot instead.
        """
        if ax is None:
            # Generate a new figure
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(6, 6)
            output = True
        else:
            output = False

        # Initial points for the sample trajectories
        if self.path is not None:
            start = self.path.iloc[0, :]
            z0, k0 = start['z'], start['kd']
        else:
            z0, k0 = 0, 1

        for i in np.linspace(-0.8, 0.8, count):
            # Plot all the trajectories that are possible
            x0 = np.array([1, z0, k0, -1, i, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='gray', linewidth=0.5, linestyle='-')
            x0 = np.array([1, z0, k0, i, -1, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='gray', linewidth=0.5, linestyle='-')
            x0 = np.array([1, z0, k0, 1, i, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='gray', linewidth=0.5, linestyle='-')
            x0 = np.array([1, z0, k0, i, 1, 0])
            path = self.simulate(x0, t_end=1e4, xi=False)
            ax.plot(path.s, path.h, color='gray', linewidth=0.5, linestyle='-')

        ax = self.plot_critical_points(ax)

        # ax.legend(ncol=1, loc=4, frameon=False)
        ax.set_xlabel(r's')
        ax.set_ylabel(r'h')
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.set_yticks(np.linspace(-1, 1, 5))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        info = (self.params['gamma'], self.params['c2'])
        ax.set_title(r'$\gamma={:.0f},~c_2={:.0e}$'.format(*info))

        plt.tight_layout()

        if save != '' and output:
            if '.png' not in save:
                save += '.png'
            plt.savefig(save, bbox_inches='tight')
        elif output:
            plt.show()

    def plot_critical_points(self, ax, pos: tuple = ('s', 'h'),
                             zorder: int = 2):
        """ Plot the critical points of the system on a given set of axes

        Parameters
        ----------
        ax : matplotlib axes
        pos : 2 or 3-tuple of str {'s', 'h', 'z}, default: ('s', 'h')
            which points to plot
        zorder : int (default: 2)
            layer on which to plot

        Returns
        -------
        ax : matplotlib axes
        """
        # We map the variables to how we stored them
        loc = dict(s=0, h=1, z=2)
        pos = tuple([loc[j] for j in pos])

        # Arrow arguments
        a_arg = {'headwidth': 3, 'width': 0.003, 'zorder': zorder}

        if self._crit_point_info is None:
            self._point_classification(self._critical_points())

        for x, info in self._crit_point_info.items():
            xs = [x[j] for j in pos]  # X-coordinates in n-dimensions

            for i in range(info['evec'].shape[1]):
                v = info['evec'][:, i] / np.linalg.norm(info['evec'][:, i]) / 3
                # Criteria for differentiating
                eig_real = np.isreal(info['eval'][i])
                eig_pos = np.real(info['eval'][i]) > 0

                if eig_real and eig_pos:
                    vs = [-v[j] for j in pos]
                    ax.quiver(*xs, *vs, pivot='tail', color='black', **a_arg)
                    vs = [v[j] for j in pos]
                    ax.quiver(*xs, *vs, pivot='tail', color='black', **a_arg)
                elif eig_real and not eig_pos:
                    vs = [-v[j] for j in pos]
                    ax.quiver(*xs, *vs, pivot='tip', color='black', **a_arg)
                    vs = [v[j] for j in pos]
                    ax.quiver(*xs, *vs, pivot='tip', color='black', **a_arg)
                elif not eig_real and eig_pos:
                    vs = [np.real(v[j]) / 1.5 for j in pos]
                    ax.quiver(*xs, *vs, pivot='tail', color='red')
                    vs = [-np.real(v[j]) / 1.5 for j in pos]
                    ax.quiver(*xs, *vs, pivot='tail', color='red')
                elif not eig_real and not eig_pos:
                    vs = [np.real(v[j]) / 1.5 for j in pos]
                    ax.quiver(*xs, *vs, pivot='tip', color='green', **a_arg)
                    vs = [-np.real(v[j]) / 1.5 for j in pos]
                    ax.quiver(*xs, *vs, pivot='tip', color='green', **a_arg)

            # Plot the solutions
            kw_arg = dict(s=8, label=info['kind'], zorder=zorder)
            if "unstable" in info['kind']:
                ax.scatter(*xs, c='gainsboro', **kw_arg)
            else:
                ax.scatter(*xs, c='black', **kw_arg)

        return ax

    def asymptotics(self) -> list:
        """ Calculate the asymptotic growth rates, and average sentiment levels
        for a given run of the simulation

        Returns
        -------
        asymptotics :   list
            order of params [psi_y, psi_kd, g]
        """
        assert self.path is not None, "Simulation run required first"
        x = self.path.loc[:, ['y', 'kd']].to_numpy()
        y0 = (x[-1, 0] - x[0, 0]) / self.t_end
        kd0 = (x[-1, 1] - x[0, 1]) / self.t_end
        #g = kd0 / (p['c2'] * p['beta2'] * p['gamma'] * y0)
        sbar = self.path.s.mean().values
        return [y0, kd0, sbar]

    @staticmethod
    def _velocity(t: float, x: np.ndarray, p: dict,
                  reduced: bool) -> np.ndarray:
        """ Calculate the velocity of the demand system in (s,h,z)-space.

        Parameters
        ----------
        t : float
            time of the simulation (not used but necessary for scipy)
        x : list
            values of (s,h,z) in that order
        p : dict
            parameters of the model
        reduced : bool
            whether to only output s, h, z velocities (True) or the velocity
            for all elements y, k, s, h, z (False)

        Returns
        -------
        v_x :   list
            velocities of s,h,z at that point in time
        """
        if reduced:
            s, h, z = x
        else:
            _, _, s, h, z = x
        v_y = (np.exp(z) - 1) / p['tau_y']
        v_s = (-s + np.tanh(p['beta1'] * s + p['beta2'] * h)) / p['tau_s']
        v_h = (-h + np.tanh(p['gamma'] * v_y)) / p['tau_h']
        v_k = p['rho'] * (p['c1'] * v_s + p['c2'] * s)
        v_z = v_k - v_y + p['epsilon']

        if reduced:
            return np.array([v_s, v_h, v_z])
        else:
            return np.array([v_y, v_k, v_s, v_h, v_z])

    def _point_classification(self, crit_points: list) -> dict:
        """ Classify the equilibria according to the eigenvalues of the Jacobian
        This uses numerical differentiation.

        Parameters
        ----------
        crit_points : list
            list of the locations of critical points. Should be made up of
            tuples of the form (s,h,z) in coordinate space (floats)

        Returns
        -------
        info : dict
            dictionary for each critical point. Key is the (s,h,z) coordinate
            tuple. Contains: Eigenvectors, eigenvalues, and kind of point.
        """

        result = {}

        def func(x):
            # Function to pass the arguments and t=0
            return self._velocity(0, x, self.params, reduced=True)

        # Iterate through points and categorize
        for point in crit_points:
            jacobian = Jacobian(func)(point)
            eig_val, eig_vec = np.linalg.eig(jacobian)
            result[point] = {'evec': eig_vec, 'eval': eig_val}

            if all(np.isreal(eig_val)):
                if all(eig_val < 0):
                    # 3x real and 3x negative -> stable node
                    result[point]['stability'] = 'stable'
                    result[point]['type'] = 'node'
                    result[point]['real'] = 'stable node'
                elif all(eig_val > 0):
                    # 3x real and 3x positive -> unstable node
                    result[point]['stability'] = 'unstable'
                    result[point]['type'] = 'node'
                    result[point]['real'] = 'unstable node'
                elif np.sum(eig_val > 0) == 2:
                    # 3x real and 2x negative -> unstable node
                    result[point]['stability'] = 'unstable'
                    result[point]['type'] = 'node'
                    result[point]['real'] = 'saddle index 2'
                else:
                    # 3x real and 1x negative -> unstable saddle
                    result[point]['stability'] = 'unstable'
                    result[point]['type'] = 'saddle'
                    result[point]['real'] = 'saddle index 1'

            elif np.sum(np.isreal(eig_val)) == 1:
                if all(np.real(eig_val) < 0):
                    # Imaginary and 2x negative -> stable focus
                    result[point]['stability'] = 'stable'
                    result[point]['type'] = 'focus'
                    result[point]['real'] = 'spiral node'
                elif all(np.real(eig_val) > 0):
                    # Imaginary and 2x positive -> unstable focus
                    result[point]['stability'] = 'unstable'
                    result[point]['type'] = 'focus'
                    result[point]['real'] = 'spiral saddle'
                else:
                    # Imaginary and mixed -> unstable focus
                    result[point]['stability'] = 'unstable'
                    result[point]['type'] = 'focus'
                    result[point]['real'] = 'unstable focus'

        self._crit_point_info = result

        return result

    def _critical_points(self) -> list:
        """ Determine the critical points in the system, where v_s, v_h and v_z
        are equal to 0. Do this by substituting in for s, solving s, and then
        solving the remaining points.
        Returns
        -------
        coordinates :   list
            list of tuples for critical coordinates in (s,h,z)-space
        """
        p = self.params

        def func(s):
            # minimisation function to find equilibria
            inner = p['gamma'] * (p['rho'] * p['c2'] * s + p['epsilon'])
            x = np.tanh(p['beta1'] * s + p['beta2'] * np.tanh(inner)) - s
            return np.abs(x)

        sols = []
        kwargs = dict(
            method='L-BFGS-B',
            bounds=[(-1, 1)],
            options={'eps': 1e-10, 'ftol': 1e-10, 'gtol': 1e-10}
        )

        def lhs(s):
            return np.arctanh(s) - p['beta1'] * s

        def rhs(s):
            inner = p['gamma'] * (p['rho'] * p['c2'] * s + p['epsilon'])
            return p['beta2'] * np.tanh(inner)

        # Use minimiser to determine where the function crosses 0 (s_dot=0)
        for x in np.linspace(-0.9, 0.9, 10):
            candidate = minimize(func, x0=x, **kwargs)
            if candidate.success:
                checks = [
                    # Check the true tolerance
                    candidate.fun < 1e-4,
                    # Check if same candidate already present
                    all([np.abs(sol - candidate.x[0]) >= 1e-4 for sol in sols])
                ]
                if all(checks):
                    sols.append(candidate.x[0])
        """
        # To check the solution correctness, one can plot this
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        for q in sols:
            ax.axvline(q, color='red')
        s = np.linspace(-0.99, 0.99, 1000)
        ax.plot(s, lhs(s), label=r"$\textrm{arctanh}(s)-\beta_1s$")
        label = r"$\beta_2\tanh(\gamma\rho c_2 s + \gamma\varepsilon)$"
        ax.plot(s, rhs(s), label=label)
        #ax.plot(s, func(s), label='min')
        #print(func(s))
        ax.set_title(r"$\gamma={:.0f},~c_2={:.1e}$".format(p['gamma'], p['c2']))
        ax.legend()
        plt.savefig('figures/testing_solutions_g{:.0f}.pdf'.format(p['gamma']))
        plt.close(fig)
        """

        # Determine h and z for each critical point in s
        coordinates = []
        for i, s in enumerate(sols):
            inner = (p['rho'] * p["c2"] * s + p['epsilon'])
            h = np.tanh(p['gamma'] * inner)
            z = np.log((p['tau_y'] * inner + 1))
            coordinates.append((s, h, z))
        return coordinates
