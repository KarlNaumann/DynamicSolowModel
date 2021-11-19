"""Graphing File
--------------------------
File applying matplotlib to generate all the figures present in the paper by
Naumann-Woleske et al.
"""

__author__ = "Karl Naumann-Woleske"
__version__ = "0.0.1"
__license__ = "MIT"

import copy
import os
import pickle

import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import ticker as tkr
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.optimize import minimize

from demandSolow import DemandSolow
from solowModel import SolowModel

# LaTeX page width for accurate sizing in inches
PAGE_WIDTH = 5.95114
FIGSIZE = (PAGE_WIDTH, PAGE_WIDTH / 2)

# General Font settings
x = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
rc('text.latex', preamble=x)
rc('text', usetex=True)
rc('font', **{'family': 'serif'})

# Font sizes
base = 12
rc('axes', titlesize=base - 2)
rc('legend', fontsize=base - 2)
rc('axes', labelsize=base - 2)
rc('xtick', labelsize=base - 3)
rc('ytick', labelsize=base - 3)

# Axis styles
cycles = cycler('linestyle', ['-', '--', ':', '-.'])
cmap = get_cmap('gray')
cycles += cycler('color', cmap(list(np.linspace(0.1, 0.9, 4))))
rc('axes', prop_cycle=cycles)


# ----------- UTILITY FUNCTION ----------- #
YEARFMT = tkr.FuncFormatter(lambda s, _: '{:.0f}'.format(s / 250.0))


def sci_notation(x: float):
    """ Format scientifically as 10^

    Parameters
    ----------
    x : float

    Returns
    -------
    y : str
    """
    a, b = '{:.2e}'.format(x).split('e')
    return r'${}\times10^{{{}}}$'.format(a, int(b))


def read_filename(filename: str) -> dict:
    """ Extract the parameter values from the filename and return them in the
    form of a dictionary. Also extracts the simulation duration.

    Parameters
    ----------
    filename    :   str

    Returns
    -------
    parameters  :   dict
        dictionary of the parameters that are base for a given simulation file
    """
    # Skip the general_ at the start and filetype .df at the end
    filename = filename[8:-3]
    # Parameter ids and float lengths
    parameters = dict(
        t_end=('t', 5), gamma=('g', 5), epsilon=('e', 7), c1=('c1_', 3),
        c2=('c2_', 7), beta1=('b1_', 3), beta2=('b2_', 3), tau_y=('ty', 4),
        tau_s=('ts', 3), tau_h=('th', 2), lam=('lam', 4), dep=('dep', 7),
        tech0=('tech', 4), rho=('rho', 4)
    )

    for param, info in parameters.items():
        if filename.find(info[0]) == -1:
            parameters[param] = None
        else:
            start = filename.find(info[0]) + len(info[0])
            parameters[param] = np.float64(filename[start:start + info[1]])
            filename = filename[start + info[1]:]

    # Determine if there is a random seed (i.e. it is a path simulation)
    seed_ix = filename.find('seed')
    if seed_ix != -1:
        parameters['seed'] = int(filename[seed_ix + 5:])
    else:
        parameters['seed'] = None

    return parameters


def save_graph(save: str, fig, format: str = 'pdf',
               pad_inches: float = 0.05, bbox_extra_artists=None):
    """ Function to save a graph to pdf format

    Parameters
    ----------
    save : str
    fig : matplotlib figure object
    format : str (default '.pdf')
    pad_inches: float (default False)
    bbox_extra_artists: any extra artists for the bbox
    """

    kwargs = {
        'format': format, 'bbox_inches': 'tight',
        'pad_inches': pad_inches, 'bbox_extra_artists': bbox_extra_artists,
    }

    if save != '':
        if save[-4:] != '.' + format:
            save += '.' + format
        plt.savefig(save, **kwargs)
        plt.close(fig)
    else:
        plt.show()

# ----------- SECTION 3.1 - SUPPLY LIMIT CASE ----------- #


def boundary_layer_approximation(t_end: float, b: float, eps: float, rho: float,
                                 tau_y: float, lam: float,
                                 dep: float) -> np.ndarray:
    """ Calculate the path of production for the classic Solow case based on
    the approximate solution from the boundary layer technique

    Parameters
    ----------
    t_end   :   float
        duration of the simulation
    b   :   float
        constant of integration
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    solution    :    np.ndarray
        solution path of production
    """
    rho_inverse = 1 - rho
    constant = (lam / dep) ** (rho / rho_inverse)
    t = np.arange(int(t_end))
    temp = (b * np.exp(-rho_inverse * t / tau_y) + 1) ** (1 / rho_inverse)
    temp += np.exp(eps * t / rho_inverse)
    return constant * (temp - 1)


def classic_solow_growth_path(t_end: float, start: list, eps: float, rho: float,
                              tau_y: float, lam: float,
                              dep: float) -> np.ndarray:
    """ Function to integrate the path of capital and production in the classic
    Solow limiting case

    Parameters
    ----------
    t_end   :   float
        total time of the simulation
    start   :   list
        initial values y0, k0
    eps :   float
        technology growth rate
    rho :   float
        capital share in cobb-douglas production
    tau_y   :   float
        characteristic timescale of production
    lam :   float
        household saving rate
    dep :   float
        depreciation rate

    Returns
    -------
    path    :   np.ndarray
        path of capital and production
    """

    path = np.empty((int(t_end), 2))
    path[0, :] = start
    for t in range(1, path.shape[0]):
        y, k = path[t - 1, :]
        v_y = np.exp((rho * k) + (eps * t) - y) - 1
        v_k = lam * np.exp(y - k) - dep
        path[t, 0] = path[t - 1, 0] + v_y / tau_y
        path[t, 1] = path[t - 1, 1] + v_k
    return path


def figure_supply_limit(params: None = None, const: float = 1.5,
                        t_end: float = 1e5, save: str = ''):
    """ Function to plot the figure for the supply limiting case that compares
    the boundary layer approximation with the true Solow path.

    Parameters
    ----------
    params : dict (default None)
        parameters for the model, needs to include rho, epsilon, tau_y, lambda
        and delta
    const : float (default 1.5)
        constant of integration
    t_end : float (default 1e5)
        duration of the simulation
    save : str (default '')
        name of the file where to save the figure. If an empty string is
        provided, will show the figure instead
    """
    if params is None:
        params = dict(rho=1 / 3, eps=1e-5, tau_y=1e3, lam=0.15, dep=0.02)

    boundary_layer = boundary_layer_approximation(t_end, const, **params)

    # Starting values are in real terms and match the BLA
    ln_y0 = np.log(boundary_layer[0])
    ln_k0 = ln_y0 / params['rho']

    # Estimate true solow model and convert to real terms (from log) to compare
    solow = classic_solow_growth_path(t_end, [ln_y0, ln_k0], **params)
    solow = np.exp(solow)

    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_WIDTH / 2))
    ax = fig.add_subplot()

    # Main Plot to compare across the entire series
    ax.plot(boundary_layer, color='firebrick', label='Approximate Solution')
    ax.plot(solow[:, 0], color='navy', label='Numerical Solution')
    ax.set_xlabel(r'Time (Years)')
    ax.set_ylabel(r'$Y$', rotation=0)
    ax.set_xlim(0, t_end)
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(YEARFMT)
    ax.legend(ncol=1, loc=4, frameon=False)

    # Inset axis to highlight the adjustment period
    axis_inset = ax.inset_axes([0.1, 0.5, 0.47, 0.47])
    axis_inset.xaxis.set_major_formatter(YEARFMT)

    # Generate markings for the inset (like a magnifying glass to show location)
    start, end = int(2e3), int(2e4)
    t = np.arange(start, end)
    axis_inset.plot(t, boundary_layer[start:end], color='firebrick')
    axis_inset.plot(t, solow[start:end, 0], color='navy')
    axis_inset.set_xlim(start, end)
    axis_inset.set_yticklabels(axis_inset.get_yticks(), backgroundcolor='w')
    mark_inset(ax, axis_inset, loc1=2, loc2=4, fc="none", ec='0.5',
               linestyle='--')

    fig.tight_layout()

    if save != '':
        if save[-4:] != '.pdf':
            save += '.pdf'
        plt.savefig(save, bbox_inches='tight', format='pdf')
        plt.close(fig)
    else:
        plt.show()


# ----------- SECTION 3.2 - DEMAND LIMIT CASE ----------- #

def fourier_transformation(series: np.ndarray, minimum_period: int):
    """ Apply a fourier filter to the given series to filter out noise with a
    frequency below the minimum period.
    Source: https://www.youtube.com/watch?v=s2K1JfNR7Sc

    Parameters
    ----------
    series : np.ndarray
        Time-series for which to apply the fourier transformation
    minimum_period : int
        frequency above which to cut off the fourier transformation

    Returns
    -------
    filtered : np.ndarray
        time-series with the fourier filter applied
    """
    n = series.shape[0]
    fhat = np.fft.fft(series, n)
    r = np.arange(n)
    indices = (r < minimum_period) + (r > n - minimum_period)
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)
    print(f"Cutoff: {n / minimum_period} days")

    return ffilt


def approximate_separatrix(parameters: dict):
    """ Approximation of the separatrix starting from positive and negative
    stable sentiment equilibria

    Parameters
    ----------
    parameters : dict
        parameter set for the demand solow

    Returns
    -------
    separatrix : pd.DataFrame
        columns include (s,h,z)
    """

    ds = DemandSolow(parameters, dict(decay=0.0, diffusion=0.0))
    points = ds.get_critical_points()

    unstable = [p for p, i in points.items() if 'unstable' in i['stability']][0]
    unstable = pd.Series(unstable, index=['s', 'h', 'z'])

    def loss(x, s):
        start = [0, x[1], 1, s, x[0], 0]
        start[0] = parameters['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=int(2e4), xi=False)
        dist = (path.loc[:, ['s', 'h', 'z']].sub(unstable) ** 2).sum(axis=1)
        return dist.min()

    def loss_eval(x, s):
        start = [0, x[1], 1, s, x[0], 0]
        start[0] = parameters['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=int(2e4), xi=False)
        dist = (path.loc[:, ['s', 'h', 'z']].sub(unstable) ** 2).sum(axis=1)
        return dist

    kwarg = dict(bounds=((-1.0, 1.0), (-np.inf, np.inf)),
                 method='L-BFGS-B', options=dict(maxiter=150))

    # Separatrix starting from the negative attractor
    pos_list = [(-0.95, -0.85, -0.13)]
    pos_sep = []
    i = 0
    stop = 1
    while stop > 1e-4:
        res = minimize(loss, pos_list[i][1:], args=(pos_list[i][0]), **kwarg)

        start = [0, res.x[1], 1, pos_list[i][0], res.x[0], 0]
        start[0] = parameters['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=int(2e4), xi=False)

        dist = loss_eval(res.x, pos_list[i][0])
        shz = path.loc[dist.idxmin() - 1000, ['s', 'h', 'z']].to_list()
        pos_sep.append(path.loc[:dist.idxmin() - 1000, ['s', 'z']])
        pos_list.append(tuple(shz))

        i += 1
        if np.abs(stop - res.fun) < 1e-9:
            break
        else:
            stop = res.fun

    # Separatrix starting from the positive attractor
    neg_list = [(0.95, 0.8, 0.1)]
    neg_sep = []
    i = 0
    stop = 1
    while stop > 1e-4:
        res = minimize(loss, neg_list[i][1:], args=(neg_list[i][0]), **kwarg)

        start = [0, res.x[1], 1, neg_list[i][0], res.x[0], 0]
        start[0] = parameters['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=int(2e4), xi=False)

        dist = loss_eval(res.x, neg_list[i][0])
        shz = path.loc[dist.idxmin() - 1000, ['s', 'h', 'z']].to_list()
        neg_sep.append(path.loc[:dist.idxmin() - 1000, ['s', 'z']])
        neg_list.append(tuple(shz))

        i += 1
        if np.abs(stop - res.fun) < 1e-9:
            break
        else:
            stop = res.fun

    sep = pd.concat([pd.concat(pos_sep), pd.concat(neg_sep).iloc[::-1]], axis=0)
    return sep


def add_critical_points(points: dict, coord: tuple, ax,
                        stableonly: bool = False):
    """ Add the critical points to a graph

    Parameters
    ----------
    points : dict
        keys are (s,h,z) coordinates, contains information on points
    coord : tuple
        tuple of which coordinates e.g. ('s','z')
    ax : matplotlib axes object
        axes to plot on
    stableonly : bool
        if only stable points should be plotted

    Returns
    -------
    ax : matplotlib axes object
    """
    loc = dict(s=0, h=1, z=2)

    for x, info in points.items():
        xs = [x[loc[i]] for i in coord]
        c = {'stable': 'green', 'unstable': 'red'}[info['stability']]
        shape = {'node': '^', 'focus': 's', 'saddle': 'o'}[info['type']]
        label = info['stability'] + ' ' + info['type']
        ax.scatter(*xs, c=c, marker=shape, s=15, label=label, zorder=2)
    return ax


def figure_demand_series_3d(parameters: dict = None, xi_args: dict = None,
                            t_end: int = int(6e4), seed: int = 40,
                            minimum_period: float = 500, save: str = ''):
    """ 3-d plot in the (s,h,z)-space of the fourier-filtered time-series for
    the demand limit case of the Dynamic Solow model. Note that arrows in the
    Naumann-Woleske et al. paper were added manually.

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    minimum_period : float (default 500)
        minimum period length over which to apply the fourier filter
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(parameters, xi_args)
    points = ds.get_critical_points()  # (s,h,z) coordinates are keys
    stable = [p for p, i in points.items() if i['stability'] == 'stable']
    shz = [p for p in stable if p[0] > 0][0]
    start = [0, shz[2], 1, min(max(shz[0], -1), 1), shz[1], 0]
    start[0] = parameters['rho'] * start[2] - start[1]
    path = ds.simulate(start, interval=1e-1, t_end=t_end, seed=seed)

    # Apply the fourier filter to the time-series
    h = fourier_transformation(path.h, minimum_period)
    s = fourier_transformation(path.s, minimum_period)
    z = fourier_transformation(path.z, minimum_period)

    # Generate 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(s, h, z, linewidth=0.9, color='navy')

    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$h$', rotation=0)
    ax.set_zlabel(r'$z$')

    points = ds.get_critical_points()
    ax = add_critical_points(points, ('s', 'h', 'z'), ax, stableonly=True)

    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='z', nbins=3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.view_init(elev=10.0, azim=-130.0)

    save_graph(save, fig)


def figure_demand_series_sy(parameters: dict = None, xi_args: dict = None,
                            t_end: int = int(1e5), seed: int = 40,
                            save: str = '', figsize=FIGSIZE):
    """ Side-by-side plot of the time-series of sentiment (LHS) and the log
    production of the economy (RHS) for the demand limit case of the Dynamic
    Solow model.

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    minimum_period : float (default 500)
        minimum period length over which to apply the fourier filter
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(parameters, xi_args)
    points = ds.get_critical_points()  # (s,h,z) coordinates are keys
    stable = [p for p, i in points.items() if 'unstable' not in i['stability']]
    shz = [p for p in stable if p[0] > 0][0]  # Positive equilibrium
    start = [0, shz[2], 1, shz[0], shz[1], 0]
    start[0] = parameters['rho'] * start[2] - start[1]
    data = ds.simulate(start, interval=1e-1, t_end=t_end, seed=seed)

    # Generate 3D plot
    fig = plt.figure(figsize=figsize)
    ax_s = fig.add_subplot(1, 2, 1)
    ax_y = fig.add_subplot(1, 2, 2)

    # Sentiment Plot
    ax_s.plot(data.s, color='navy', linewidth=0.8)
    ax_s.set_ylim(-1, 1)
    ax_s.set_xlim(0, t_end)
    ax_s.set_ylabel(r'$s$', rotation=0)
    ax_s.set_xlabel(r'Time (Years)')
    ax_s.minorticks_on()
    ax_s.xaxis.set_major_formatter(YEARFMT)

    # Production Plot
    ax_y.plot(data.y, color='navy', linewidth=0.8)
    ax_y.set_xlim(0, t_end)
    ax_y.set_ylabel(r'$y$', rotation=0)
    ax_y.set_xlabel(r'Time (Years)')
    ax_y.minorticks_on()
    ax_y.xaxis.set_major_formatter(YEARFMT)

    fig.tight_layout()
    save_graph(save, fig)


def figure_demand_3d_phase(parameters: dict = None, t_end: int = int(2e4),
                           save: str = '',
                           figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 2)):
    """ Three dimensional phase diagram in the (s,h,z)-space to show the
    attracting regions and equilibria.

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    t_end : int
        total duration of the simulation
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """

    ds = DemandSolow(parameters, dict(decay=0.0, diffusion=0.0))

    # Load starting points
    shz = pd.read_csv('starting_shz_3d.csv')

    series = []
    for i in shz.index:
        start = [0, shz.loc[i, 'z'], 1, shz.loc[i, 's'], shz.loc[i, 'h'], 0]
        start[0] = parameters['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=1e4, xi=False)
        series.append(path)

    # Generate 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax = add_critical_points(ds.get_critical_points(), ('s', 'h', 'z'), ax)

    # Add chosen trajectories to the graph
    kw_args = dict(linewidth=0.3, linestyle='-', zorder=1)
    for s in series:
        if s.s.iloc[-1] < 0:
            color = 'firebrick'
        else:
            color = 'navy'
        ax.plot(s.s, s.h, s.z, color=color, **kw_args)

    fontsize = 8
    ax.set_ylabel(r'$h$', fontsize=fontsize, labelpad=-5, rotation=0)
    ax.set_xlabel(r'$s$', fontsize=fontsize, labelpad=-5)
    ax.set_zlabel(r'$z$', fontsize=fontsize, labelpad=-5)

    labels = [-1, 0, 1]
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_zticks(labels)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_zticklabels(labels, fontsize=fontsize)

    ax.tick_params(axis='both', pad=-3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1.4, 1)

    ax.view_init(elev=10.0, azim=-55.0)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_graph(save, fig)


def figure_demand_3d_phase_cycle(parameters: dict = None, t_end: int = int(2e4),
                                 save: str = '',
                                 figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 2)):
    """ Three dimensional phase diagram in the (s,h,z)-space to show the
    attracting regions and equilibria for the special case of the limit cycle

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    t_end : int
        total duration of the simulation
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(parameters, dict(decay=0.0, diffusion=0.0))
    start = [0, 0, 1, 0, 0, 0]
    start[0] = parameters['rho'] * start[2] - start[1]
    path = ds.simulate(start, interval=1e-1, t_end=3e4, xi=False)
    path = path.loc[int(1e4):, :]

    # Generate 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    plt.margins(0, 0, 0)

    # Add the equilibrium points to the graph
    ax = add_critical_points(ds.get_critical_points(), ('s', 'h', 'z'), ax)

    kw_args = dict(linewidth=0.9, linestyle='-', zorder=1)
    ax.plot(path.s, path.h, path.z, color='navy', **kw_args)

    fontsize = 8
    ax.set_ylabel(r'$h$', fontsize=fontsize, labelpad=-5, rotation=0)
    ax.set_xlabel(r'$s$', fontsize=fontsize, labelpad=-5)
    ax.set_zlabel(r'$z$', fontsize=fontsize, labelpad=-5)

    labels = [-1, 0, 1]
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_zticks(labels)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_zticklabels(labels, fontsize=fontsize)

    ax.tick_params(axis='both', pad=-3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1.4, 1)

    ax.view_init(elev=10.0, azim=-55.0)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_graph(save, fig)


def figure_demand_sz_comparison(coherence_params: dict, cycle_params: dict,
                                t_end: int = int(1e5), save: str = '',
                                figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 2)):
    """ 2-d plot in the (s,z)-space of the phase diagram for the demand limit
    case of the Dynamic Solow model.

    Parameters
    ----------
    coherence_parameters : dict
        parameters to use for the coherence resonance case
    cycle_parameters : dict
        parameters to use for the cycle case
    t_end : int
        total duration of the simulation
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    figsize : tuple
        size of the figure
    """

    # Load coherence resonance
    ds = DemandSolow(coherence_params, dict(decay=0.0, diffusion=0.0))
    coherence_points = ds.get_critical_points()
    shz = pd.read_csv('starting_shz_3d.csv')
    coherence_series = []
    for i in shz.index:
        start = [0, shz.loc[i, 'z'], 1, shz.loc[i, 's'], shz.loc[i, 'h'], 0]
        start[0] = coherence_params['rho'] * start[2] - start[1]
        path = ds.simulate(start, interval=1e-1, t_end=1e4, xi=False)
        coherence_series.append(path)

    # Limit Cycle
    ds = DemandSolow(cycle_params, dict(decay=0.0, diffusion=0.0))
    cycle_points = ds.get_critical_points()
    start = np.array([1, 0, 1, 0, 0, 0], dtype=float)
    start[0] = cycle_params['epsilon'] + cycle_params['rho'] * start[2]
    cycle_path = ds.simulate(start, t_end=t_end, xi=False).iloc[int(2e4):, :]

    # Generate figure
    fig = plt.figure(figsize=figsize)
    ax_coh = fig.add_subplot(1, 2, 1)
    ax_coh = add_critical_points(coherence_points, ('s', 'z'), ax_coh)
    kw_args = dict(linewidth=0.5, linestyle='-', zorder=1, alpha=0.7)

    for s in coherence_series:
        c = 'navy' if s.s.iloc[-1] > 0 else 'firebrick'
        ax_coh.plot(s.s, s.z, color=c, **kw_args)

    sep = approximate_separatrix(coherence_params)
    ax_coh.plot(sep.s, sep.z, linestyle='--', color='black', linewidth=0.7,
                zorder=1)

    ax_cyc = fig.add_subplot(1, 2, 2)
    ax_cyc = add_critical_points(cycle_points, ('s', 'z'), ax_cyc)
    ax_cyc.plot(cycle_path.s, cycle_path.z, color='navy', linestyle='-',
                linewidth=1.0, zorder=1)

    lim_coh, lim_cyc = ax_coh.get_ylim(), ax_cyc.get_ylim()
    ylim = (min(lim_coh[0], lim_cyc[0]) - 0.1,
            max(lim_coh[1], lim_cyc[1]) + 0.1)

    for ax in [ax_coh, ax_cyc]:
        ax.set_ylim(ylim)
        ax.set_xlim(-1, 1.0)
        ax.set_xlabel(r'$s$')
        ax.set_ylabel(r'$z$', rotation=0)
        ax.minorticks_on()

    fig.tight_layout()
    save_graph(save, fig)


# ----------- SECTION 4.2 - ASYMPTOTIC CONVERGENCE ----------- #

def figure_asymp_supply(parameters: dict, xi_args: dict, t_end: int = int(5e7),
                        seed: int = 40, save: str = '',
                        figsize: tuple = (PAGE_WIDTH / 1.5, PAGE_WIDTH / 3)):
    """ Asymptotic behaviour of the enforced supply case i.e. k=ks at all points
    irrespective of capital demand

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    save : str (default '')
        name of the figure to save. If '' will show figure (slower).
        Figures are saved in pdf format
    """
    model = SolowModel(parameters, xi_args)
    start = np.array([3, 10, 400, 0, 0, 1, 0])
    start[0] = parameters['rho'] * start[1]
    path = model.simulate(start, t_end, seed=seed, case='general')

    growth = parameters['epsilon'] / (1 - parameters['rho'])
    growth = start[0] + growth * np.arange(t_end)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    kwargs = dict(linewidth=0.8)
    ax.plot(path.kd, color='firebrick', label=r'$k_d$', zorder=2, **kwargs)
    ax.plot(path.y, color='navy', label=r'$y$', zorder=2, **kwargs)
    ax.plot(path.ks, color='black', label=r'$k_s$', zorder=2, **kwargs)
    ax.plot(growth, color='gold', label=r'$R$', zorder=1,
            linewidth=1.0, linestyle='-.')

    ax.set_xlim(0, t_end)
    ax.set_xlabel(r'Time (years)')
    # ax.set_ylabel(r'$y,~k_s,~k_d$', rotation=0)
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(YEARFMT)
    ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(1.0, -0.05),
              frameon=False, handlelength=1)

    fig.tight_layout()
    save_graph(save, fig)


def figure_asymp_purecycle(parameters: dict, t_end: int = int(5e7),
                           save: str = '',
                           figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 3)):
    """ Asymptotic behaviour of the enforced supply case i.e. k=ks at all points
    irrespective of capital demand

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    t_end : int
        total duration of the simulation
    save : str (default '')
        name of the figure to save. If '' will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(parameters, dict(decay=0.0, diffusion=0.0))
    start = np.array([0, 0, 1, 0, 0, 0, 1])
    start[1] = parameters['rho'] * start[2] - start[0]
    path = ds.simulate(start, interval=1e-1, t_end=t_end, xi=False)

    growth = parameters['epsilon'] / (1 - parameters['rho'])
    growth = start[0] + growth * np.arange(t_end)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    kwargs = dict(linewidth=0.6)
    ax1.plot(path.kd, color='firebrick', label=r'$k_d$', zorder=2, **kwargs)
    ax1.plot(path.y, color='navy', label=r'$y$', zorder=2, **kwargs)
    # ax1.plot(path.ks, color='black', label=r'$k_s$', zorder=2, **kwargs)
    ax1.plot(growth, color='gold', label=r'$R$', zorder=1,
             linewidth=1.0, linestyle='-.')

    ax1.set_xlim(0, t_end)
    ax1.set_xlabel(r'Time (years)')
    # ax1.set_ylabel(r'$y,~k$')
    ax1.minorticks_on()
    ax1.xaxis.set_major_formatter(YEARFMT)
    ax1.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.01, 1.05),
               frameon=False, handlelength=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(path.s.loc[:int(2e5)], **kwargs)
    ax2.set_xlim(0, int(2e5))
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel(r'Time (years)')
    ax2.set_ylabel(r'$s$', rotation=0)
    ax2.minorticks_on()
    ax2.xaxis.set_major_formatter(YEARFMT)

    fig.tight_layout()
    save_graph(save, fig)


def figure_asymp_noisecycle(parameters: dict, xi_args: dict,
                            t_end: int = int(5e7), seed: int = 40,
                            save: str = '',
                            figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 3)):
    """ Asymptotic behaviour of the enforced supply case i.e. k=ks at all points
    irrespective of capital demand

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    save : str (default '')
        name of the figure to save. If '' will show figure (slower).
        Figures are saved in pdf format
    """

    ds = DemandSolow(parameters, xi_args)
    start = [0, 0, 1, 0, 0, 0]
    start[0] = parameters['rho'] * start[2] - start[1]
    path = ds.simulate(start, seed=seed, interval=1e-1, t_end=t_end)

    growth = parameters['epsilon'] / (1 - parameters['rho'])
    growth = start[0] + growth * np.arange(t_end)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    kwargs = dict(linewidth=0.6)
    ax1.plot(path.kd, color='firebrick', label=r'$k_d$', zorder=2, **kwargs)
    ax1.plot(path.y, color='navy', label=r'$y$', zorder=2, **kwargs)
    ax1.plot(growth, color='gold', label=r'$R$', zorder=1,
             linewidth=1.0, linestyle='-.')

    ax1.set_xlim(0, t_end)
    ax1.set_xlabel(r'Time (years)')
    # ax1.set_ylabel(r'$y,~k_s,~k_d$')
    ax1.minorticks_on()
    ax1.xaxis.set_major_formatter(YEARFMT)
    ax1.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.01, 1.05),
               frameon=False, handlelength=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(path.s.loc[:int(2e5)], **kwargs)
    ax2.set_xlim(0, int(2e5))
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel(r'Time (years)')
    ax2.set_ylabel(r'$s$', rotation=0)
    ax2.minorticks_on()
    ax2.xaxis.set_major_formatter(YEARFMT)

    fig.tight_layout()
    save_graph(save, fig)


def figure_asymp_demand(parameters: dict, xi_args: dict, t_end: int = int(5e7),
                        seed: int = 40, save: str = '',
                        figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 3)):
    """ Asymptotic behaviour of the coherence resonance demand case

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    save : str (default '')
        name of the figure to save. If '' will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(parameters, xi_args)
    points = ds.get_critical_points()  # (s,h,z) coordinates are keys
    stable = [p for p, i in points.items() if 'unstable' not in i['stability']]
    shz = [p for p in stable if p[0] > 0][0]
    start = [101, shz[2], 1, shz[0], shz[1], 0, 100]
    # start[0] = parameters['rho'] * start[2] - start[1]
    path = ds.simulate(start, interval=1e-1, t_end=t_end, seed=seed)

    r_star = start[0] + path.y.diff().mean() * np.arange(t_end)
    growth = parameters['epsilon'] / (1 - parameters['rho'])
    growth = start[0] + growth * np.arange(t_end)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    kwargs = dict(linewidth=0.8)
    ax.plot(path.kd, color='firebrick', label=r'$k_d$', zorder=2, **kwargs)
    ax.plot(path.ks, color='black', linestyle='-', label=r'$k_s$', zorder=2, **kwargs)
    ax.plot(path.y, color='navy', label=r'$y$', zorder=2, **kwargs)
    ax.plot(growth, color='gold', label=r'$R$', zorder=3,
            linewidth=1.0, linestyle='-.')
    ax.plot(r_star, color='gold', label=r'$R^\star$', zorder=3,
            linewidth=1.0, linestyle=':')

    ax.set_xlim(0, t_end)
    ax.set_xlabel(r'Time (years)')
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(YEARFMT)
    ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.01, 1.05),
              frameon=False, handlelength=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(path.s.loc[:int(2e5)], label=r'$s$', **kwargs)
    ax2.set_xlim(0, int(2e5))
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel(r'Time (years)')
    ax2.set_ylabel(r'$s$', rotation=0)
    ax2.minorticks_on()
    ax2.xaxis.set_major_formatter(YEARFMT)

    fig.tight_layout()
    save_graph(save, fig)


def figure_asymp_general(parameters: dict, xi_args: dict, t_end: int = int(5e7),
                         seed: int = 40, save: str = '',
                         figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH / 3)):
    """ Asymptotic behaviour of the coherence resonance general case

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        decay and diffusion of the Ornstein-Uhlenbeck process
    t_end : int
        total duration of the simulation
    seed : int
        numpy random seed for the simulation
    save : str (default '')
        name of the figure to save. If '' will show figure (slower).
        Figures are saved in pdf format
    """
    model = SolowModel(parameters, xi_args)
    start = np.array([1, 10, 9, 0.85, 0.5, 1, 0])
    start[0] = parameters['rho'] * min(start[1], start[2])
    path = model.simulate(start, interval=1e-1, t_end=t_end, seed=seed)

    growth = parameters['epsilon'] / (1 - parameters['rho'])
    growth = start[0] + growth * np.arange(t_end)

    fig = plt.figure(figsize=figsize)
    # Long-run growth
    ax1 = fig.add_subplot(1, 2, 1)
    kwargs = dict(zorder=2, linewidth=0.8)
    ax1.plot(path.kd, color='firebrick', label=r'$k_d$', **kwargs)
    ax1.plot(path.y, color='navy', label=r'$y$', **kwargs)
    ax1.plot(path.ks, color='black', label=r'$k_s$', **kwargs)
    ax1.plot(growth, color='gold', label=r'$R$', zorder=3,
             linewidth=1.0, linestyle='-.')

    ax1.set_xlim(0, t_end)
    ax1.set_xlabel(r'Time (years)')
    ax1.minorticks_on()
    ax1.xaxis.set_major_formatter(YEARFMT)
    ax1.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.01, 1.05),
               frameon=False, handlelength=1)

    # Interplay of demand and supply
    ax2 = fig.add_subplot(1, 2, 2)
    start, end = int(5e6), int(7.5e6)
    start, end = 0, int(2.5e6)
    ax2.plot(path.kd.loc[start:end], color='firebrick',
             label=r'$k_d$', linestyle='-', linewidth=0.8)
    ax2.plot(path.ks.loc[start:end], color='black',
             label=r'$k_s$', linestyle=':', linewidth=0.8)

    supply = path.loc[start:end].ks < path.loc[start:end].kd
    low, high = ax2.get_ylim()
    ax2.fill_between(path.loc[start:end, :].index, low, high, where=supply,
                     alpha=0.5, facecolor='lightgray', edgecolor="none")

    ax2.set_xlim(start, end)
    ax2.set_ylim(low, high)
    ax2.set_xlabel(r'Time (years)')
    ax2.minorticks_on()
    ax2.xaxis.set_major_formatter(YEARFMT)
    ax2.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.01, 1.05), frameon=False)

    fig.tight_layout()
    save_graph(save, fig)

# ----------- SECTION 4.3 - MEDIUM-TERM DYNAMICS ----------- #


def figure_medium_term_dynamics(parameters: dict, xi_args: dict,
                                t_end: int = int(3e5),
                                seed: int = 40, save: str = '',
                                figsize: tuple = (PAGE_WIDTH, PAGE_WIDTH)):
    """ Function to plot the dynamics of a single realisation of the general
    Dynamic Solow model in the convergent case

    Parameters
    ----------
    parameters : dict
        Parameter set for the Dynamic Solow model
    xi_args : dict
        Parameters of the Ornstein-Uhlenbeck process
    seed : int (default 40)
        numpy random seed
    t_end : int (default int(5e7))
        duration of the simulation. For asymptotic analysis should be >1e7
    save : str
        location of where to save the figure
    figsize : tuple
        (width, height) in inches
    """
    sm = SolowModel(parameters, xi_args)
    start = np.array([1, 10, 9, 0, 0, 1, 0])
    start[0] = 1e-5 + (min(start[1:3]) / 3)
    path = sm.simulate(start, t_end=t_end, seed=seed)

    fig = plt.figure(figsize=figsize)
    ax_s = fig.add_subplot(3, 1, 3)
    ax_y = fig.add_subplot(3, 1, 1, sharex=ax_s)
    ax_k = fig.add_subplot(3, 1, 2, sharex=ax_s)

    # Production
    ax_y.plot(path.y, color='navy', linewidth=0.8)
    ax_y.set_ylabel(r'$y$', rotation=0)

    # Capital Timeseries
    ax_k.plot(path.ks, label=r'Supply', color='black', linewidth=0.8)
    ax_k.plot(path.kd, label=r'Demand', color='firebrick', linewidth=0.8)
    ax_k.legend(frameon=False, loc='upper left', ncol=2, bbox_to_anchor=(0, 1.05))
    ax_k.set_ylabel(r'$k$', rotation=0)

    # Sentiment timeseries
    ax_s.plot(path.s, color='black', linewidth=0.8)
    ax_s.set_ylim(-1, 1)
    ax_s.set_ylabel(r'$s$', rotation=0)

    supply = path.ks < path.kd
    for ax in [ax_y, ax_k, ax_s]:
        low, high = ax.get_ylim()
        ax.fill_between(path.index, low, high, where=supply,
                        alpha=0.5, facecolor='lightgray', edgecolor="none")
        ax.set_ylim(low, high)
        ax.set_xlim(0, t_end)
        ax.xaxis.set_major_formatter(YEARFMT)
        ax.xaxis.major.formatter._useMathText = True
        ax.minorticks_on()
        ax.set_xlabel(r'Time (Years)')

    fig.align_ylabels()
    fig.tight_layout()
    save_graph(save, fig)


# ----------- SECTION 4.4 - BUSINESS CYCLE CHARACTERISTICS ----------- #


def winsorize(series: pd.Series, perc: tuple = (0.05, 0.95)) -> pd.Series:
    """ Winsorize a given timeseries

    Parameters
    ----------
    series : pd.Series
    perc : tuple
        lower and higher cutoff points, must be in [0,1]

    Returns
    -------
    series : pd.Series
        winsorized verision of the same series
    """
    quint = series.quantile(perc)
    new = series[series > quint.min()]
    return new[new < quint.max()]


def load_cycles(parameters: dict, simulation_folder: str) -> list:
    """ Load the simulation results for the business cycles. These can be found
    in the simulations.py files

    Parameters
    ----------
    parameters : dict
        parameters for which to load the simulations
    simulation_folder : str
        where to find the simulations

    Returns
    -------
    cycles : list
        list of loaded dataframes
    """
    # DataFrame indexed by filename, columns are parameters
    files = os.listdir(simulation_folder)
    files = pd.DataFrame.from_dict({f: read_filename(f) for f in files}).T

    # Keep only files where given parameters are used
    param = {k: v for k, v in parameters.items() if k in files.columns}
    files = files.loc[(files[list(param)] == pd.Series(param)).all(axis=1)]
    assert files.shape[0] > 0, "No simulations found for given criteria"

    # Extract information
    cycles = []
    for filename in files.index:
        file = open(simulation_folder + '/' + filename, 'rb')
        try:
            cycle = pickle.load(file)
            file.close()
            cycles.append(cycle)
            del cycle
        except EOFError:
            file.close()

    return cycles


def figure_cycle_duration(parameters: dict, simulation_folder: str,
                          measure: str = 's', binyears: int = 2,
                          save: str = ''):
    """ Generate histogram of the business cycle duration

    Parameters
    ----------
    parameters : dict
        set of parameters that are key to the simulation
    simulation_folder : str
        location where to find the .df files from the simulations. These are
        created using the simulation.py file
    binyears : int (default 2)
        number of years per bin
    save : str (default '')
        name of the figure. If empty will simply plot the figure
    """
    cycles = load_cycles(parameters, simulation_folder)
    series = 'duration_' + measure
    duration = pd.concat([c.loc[:, series] / 250 for c in cycles], ignore_index=True).dropna()
    print("Duration stats:\n", duration.describe())

    prop4070 = duration[40 < duration][duration < 70].shape[0]
    prop10150 = duration[10 < duration][duration < 150].shape[0]
    print(f'Proportion 40-70 / 10-150: {prop4070/prop10150}')

    start, end = 10, 151
    bins = np.arange(start, end, binyears)
    n_dur, bins_dur = np.histogram(duration, bins=bins)  # 2-year bins
    n_dur = n_dur / np.sum(n_dur)

    n_peak = 2
    ix = n_peak + np.argmax(n_dur[n_peak:])
    peak = bins_dur[ix] + 0.5 * (bins_dur[ix + 1] - bins_dur[ix])

    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_WIDTH / 3))
    kwargs = dict(color='navy', alpha=0.95, edgecolor='black', linewidth=0.5)

    # Duration Histogram
    ax_cycle = fig.add_subplot(1, 1, 1)
    ax_cycle.hist(bins_dur[:-1], bins_dur, weights=n_dur, **kwargs)
    ax_cycle.axvline(peak, color='firebrick', linestyle='--',
                     label=r'Peak: {:.1f} years'.format(peak))
    ax_cycle.set_xlabel('Duration (years)')
    # ax_cycle.set_ylabel('Proportion')
    ax_cycle.minorticks_on()
    ax_cycle.set_xlim(start, end)
    ax_cycle.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    save_graph(save, fig)


def figure_cycle_depth(parameters: dict, simulation_folder: str,
                       measure: str = 's', binsize: float = 0.1,
                       save: str = ''):
    """ Generate histogram of the percentage depth of recessions.

    Parameters
    ----------
    parameters : dict
        set of parameters that are key to the simulation
    simulation_folder : str
        location where to find the .df files from the simulations. These are
        created using the simulation.py file
    binsize : float (default 0.2)
        width of the bins in the histogram
    save : str (default '')
        name of the figure. If empty will simply plot the figure
    """
    cycles = load_cycles(parameters, simulation_folder)
    p2t = []
    for c in cycles:
        duration = c.loc[:, 'duration_' + measure] / 250
        peak = c.loc[:, 'peak_' + measure].loc[duration > 30].loc[duration < 100]
        trough = c.loc[:, 'trough_' + measure].loc[duration > 30].loc[duration < 100]
        p2t.append(100 * (peak - trough).div(peak))

    p2t = pd.concat(p2t)
    print("Peak-to-trough stats:\n", p2t.describe())

    bins = np.arange(0, 10, binsize)
    n_dur, bins_dur = np.histogram(p2t, bins=bins)  # 2-year bins
    n_dur = n_dur / np.sum(n_dur)

    ix = np.argmax(n_dur)
    peak = bins_dur[ix] + 0.5 * (bins_dur[ix + 1] - bins_dur[ix])

    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_WIDTH / 3))
    kwargs = dict(color='navy', alpha=0.95, edgecolor='black', linewidth=0.5)

    # Duration Histogram
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(bins_dur[:-1], bins_dur, weights=n_dur, **kwargs)
    ax.axvline(peak, color='firebrick', linestyle='--',
               label=r'Peak: {:.1f}\%'.format(peak))
    ax.set_xlabel(r'Peak-to-Trough Difference in $y$')
    ax.set_ylabel('Proportion')

    tick_loc = np.arange(0, 11, 1)
    tick_label = [r"{:.0f}\%".format(x) for x in tick_loc]
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(tick_label)

    ax.set_xlim(0, 10)
    ax.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    save_graph(save, fig)


def figure_hist_prevalence(parameters: dict, simulation_folder: str,
                           measure: str = 's', binsize: float = 0.1,
                           save: str = ''):
    """ Generate histogram of the percentage depth of recessions.

    Parameters
    ----------
    parameters : dict
        set of parameters that are key to the simulation
    simulation_folder : str
        location where to find the .df files from the simulations. These are
        created using the simulation.py file
    binsize : float (default 0.2)
        width of the bins in the histogram
    save : str (default '')
        name of the figure. If empty will simply plot the figure
    """
    proportions = load_cycles(parameters, simulation_folder)

    size = 0.025
    bins = np.arange(0.5, 1.0, size)
    n_dur, bins_dur = np.histogram(proportions, bins=bins)  # 2-year bins
    n_dur = n_dur / len(proportions)

    ix = np.argmax(n_dur)
    peak = bins_dur[ix] + 0.5 * (bins_dur[ix + 1] - bins_dur[ix])

    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_WIDTH / 3))
    kwargs = dict(color='navy', alpha=0.95, edgecolor='black', linewidth=0.5)

    # Duration Histogram
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(bins_dur[:-1], bins_dur, weights=n_dur, **kwargs)
    ax.axvline(peak, color='firebrick', linestyle='--',
               label=r'Peak: {:.1f}\%'.format(100 * peak))
    ax.set_xlabel(r'Proportion of time $k_s>k_d$')
    ax.set_ylabel('Proportion')

    tick_loc = bins[::int(0.05 / size)]  # np.arange(0, 11, 1)
    tick_label = [r"{:.0f}\%".format(100 * x) for x in tick_loc]
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(tick_label)

    ax.set_xlim(min(bins), max(bins))
    ax.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    save_graph(save, fig)


# ----------- APPENDIX C - PARAMETERISING THE DEMAND SYSTEM ----------- #


def add_s_dynamics(parameters: dict, xi_args: dict, ax, t_end: int = int(6e4),
                   seed: int = 42):
    """ Add sentiment dynamics to a given matplotlib axis

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        noise process arguments to use in demand case
    ax : matplotlib axes object
        where to plot the sentiment
    t_end : int
        total duration of the simulation
    seed : int
        random seed to initialise the simulation
    """
    ds = DemandSolow(params=parameters, xi_args=xi_args)

    # Add equilibria as horizontal lines & pick positive equilibrium to start
    shz = (0, 0, 0)
    for x, info in ds.get_critical_points().items():
        if info['stability'] == 'stable':
            if x[0] > shz[0]:
                shz = x

    # Time-series of sentiment, starting at positive equilibrium
    start = np.array([1, shz[2], 3, shz[0], shz[0], 0], dtype=float)
    start[0] = parameters['rho'] * start[2] - start[1]
    path = ds.simulate(start, t_end=t_end, seed=seed)
    ax.plot(path.s, color='navy', linewidth=0.7, zorder=2)

    # Format axis
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, t_end)
    ax.set_ylabel(r'$s$', rotation=0)
    ax.set_xlabel(r'Time (years)')
    ax.xaxis.set_major_formatter(YEARFMT)
    ax.minorticks_on()
    return ax


def figure_appxC_sz_s(parameters: dict, t_end: int = int(6e4),
                      limit_cycle: bool = False, sz_lines: list[tuple] = None,
                      s_lines: int = 5, figsize: tuple = FIGSIZE,
                      save: str = '', lim: tuple = None):
    """ 2-part diagram of the dynamics for small values of gamma.
    LHS: Phase diagram in the (s,z) space. RHS: time-series of the sentiment
    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    t_end : int
        total duration of the simulation
    limit_cycle : bool
        if true draws only one trajectory in the RHS
    figsize : tuple
        figure size, default (PAGE_WIDTH, PAGE_WIDTH / 2)
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(params=parameters, xi_args=dict(decay=0.0, diffusion=0.0))
    start = np.array([1, 0, 3, 0, 0, 0], dtype=float)
    start[0] = parameters['rho'] * start[2] - start[1]

    if sz_lines is None:
        sz_lines = []
        points = ds.get_critical_points()
        for p, i in points.items():
            if i['stability'] != 'stable':
                continue
            for increment in np.linspace(-0.25, 0.25, 6):
                sz_lines.append((p[0] + increment, 0.25))
                sz_lines.append((p[0] + increment, -0.25))

    if s_lines is None:
        s_lines = sz_lines

    fig = plt.figure(figsize=figsize)
    ax_sz = fig.add_subplot(1, 2, 1)
    kwargs = dict(linewidth=0.5, linestyle='-', zorder=1, alpha=0.75)

    if limit_cycle:
        t = t_end + int(2e4)  # Discard starting path into the limit cycle
        path = ds.simulate(start, t_end=t, xi=False)
        path = path.iloc[int(2e4):, :].reset_index()
        ax_sz.plot(path.s, path.z, color='navy', **kwargs)
    else:
        for s, z in sz_lines:
            start[1] = z
            start[3] = s
            path = ds.simulate(start, t_end=t_end, xi=False)
            c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
            ax_sz.plot(path.s, path.z, color=c, **kwargs)

    add_critical_points(ds.get_critical_points(), ('s', 'z'), ax_sz)

    if lim is not None:
        ax_sz.set_ylim(lim)
    ax_sz.set_xlim(-1, 1)
    ax_sz.set_xlabel(r'$s$')
    ax_sz.set_ylabel(r'$z$')
    ax_sz.minorticks_on()

    # Plot the convergence to the equilibria on the RHS
    ax_s = fig.add_subplot(1, 2, 2)
    if limit_cycle:
        ax_s.plot(path.s, color='navy', **kwargs)
    else:
        for s, z in s_lines:
            start[1] = z
            start[3] = s
            path = ds.simulate(start, t_end=t_end, xi=False)
            c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
            ax_s.plot(path.s, color=c, **kwargs)

    ax_s.set_ylim(-1, 1)
    ax_s.set_xlim(0, t_end)
    ax_s.set_ylabel(r'$s$')
    ax_s.set_xlabel(r'Time (years)')
    ax_s.xaxis.set_major_formatter(YEARFMT)
    ax_s.minorticks_on()

    fig.tight_layout()
    save_graph(save, fig, pad_inches=0.0)


def figure_appxC_limit_cycle(parameters: dict, gammas: dict,
                             t_end: int = int(6e4), figsize: tuple = FIGSIZE,
                             save: str = '', lim: tuple = None):
    """ 2-part diagram of the dynamics for different gammas
    LHS: Phase diagram in the (s,z) space. RHS: time-series of the sentiment

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    gammas : dict
        keys are the gamma variations, contains dict of ['limit _cycle': bool
        if limit cycle, 'sz_lines': starting coordinates in the sz-space (LHS),
        's_lines': starting coordinates for s (RHS)]
    t_end : int
        total duration of the simulation
    limit_cycle : bool
        if true draws only one trajectory in the RHS
    figsize : tuple
        figure size, default (PAGE_WIDTH, PAGE_WIDTH / 2)
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """

    fig, axs = plt.subplots(len(gammas.keys()), 2, figsize=figsize)
    kwargs = dict(linewidth=0.5, linestyle='-', zorder=1, alpha=0.75)

    ds = DemandSolow(params=parameters, xi_args=dict(decay=0.0, diffusion=0.0))
    start = np.array([1, 0, 3, 0, 0, 0], dtype=float)
    start[0] = parameters['rho'] * start[2] - start[1]

    for i, gamma in enumerate(gammas.keys()):

        ds.params['gamma'] = gamma

        # Plot the phase diagram in the (s,z) space
        if gammas[gamma]['limit_cycle']:
            t = t_end + int(2e4)  # Discard starting path into the limit cycle
            path = ds.simulate(start, t_end=t, xi=False)
            path = path.iloc[int(2e4):, :].reset_index()
            axs[i, 0].plot(path.s, path.z, color='navy', **kwargs)
        elif gammas[gamma]['sz_lines'] is None:
            gammas[gamma]['sz_lines'] = []
            points = ds.get_critical_points()
            for p, i in points.items():
                if i['stability'] != 'stable':
                    continue
                for increment in np.linspace(-0.25, 0.25, 6):
                    gammas[gamma]['sz_lines'].append((p[0] + increment, 0.25))
                    gammas[gamma]['sz_lines'].append((p[0] + increment, -0.25))
        else:
            for s, z in gammas[gamma]['sz_lines']:
                start[1] = z
                start[3] = s
                path = ds.simulate(start, t_end=t_end, xi=False)
                c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
                axs[i, 0].plot(path.s, path.z, color=c, **kwargs)

        add_critical_points(ds.get_critical_points(), ('s', 'z'), axs[i, 0])

        if lim is not None:
            axs[i, 0].set_ylim(lim)
        axs[i, 0].set_xlim(-1, 1)
        axs[i, 0].set_xlabel(r'$s$')
        axs[i, 0].set_ylabel(r'$z$', rotation=0)
        axs[i, 0].minorticks_on()
        axs[i, 0].set_title(r'$\gamma={}$'.format(parameters['gamma']))

        # Plot of sentiment over time
        if gammas[gamma]['limit_cycle']:
            axs[i, 1].plot(path.s, color='navy', **kwargs)
        else:
            if gammas[gamma]['s_lines'] is None:
                gammas[gamma]['s_lines'] = gammas[gamma]['sz_lines']

            for s, z in gammas[gamma]['s_lines']:
                start[1] = z
                start[3] = s
                path = ds.simulate(start, t_end=t_end, xi=False)
                c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
                axs[i, 1].plot(path.s, color=c, **kwargs)

        axs[i, 1].set_ylim(-1, 1)
        axs[i, 1].set_xlim(0, t_end)
        axs[i, 1].set_ylabel(r'$s$', rotation='horizontal')
        axs[i, 1].set_xlabel(r'Time (years)')
        axs[i, 1].xaxis.set_major_formatter(YEARFMT)
        axs[i, 1].minorticks_on()
        axs[i, 1].set_title(r'$\gamma={}$'.format(parameters['gamma']))

    fig.align_ylabels()
    fig.tight_layout()

    save_graph(save, fig, pad_inches=0.0)


def figure_appxC_c2_effect(parameters: dict, xi_args: dict, c2_alt: float,
                           t_end: int = int(3e4), seed: int = 42,
                           figsize=FIGSIZE, save: str = ''):
    """ 2-part diagram of the dynamics for different c2
    LHS: Base case. RHS: alternate c2

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        noise arguments
    c2_alt : float
        alternate value of c2 for the right-hand side
    t_end : int
        total duration of the simulation
    seed : int
        random seed for numpy
    figsize : tuple
        Dimensions of the figure
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """

    fig = plt.figure(figsize=figsize)
    ax_s1 = fig.add_subplot(1, 2, 1)
    ax_s2 = fig.add_subplot(1, 2, 2)

    ax_s1 = add_s_dynamics(parameters, xi_args, ax_s1, t_end, seed)
    ax_s1.set_title(r'$c_2=$' + sci_notation(parameters['c2']))

    parameters['c2'] = c2_alt
    ax_s2 = add_s_dynamics(parameters, xi_args, ax_s2, t_end, seed)
    ax_s2.set_title(r'$c_2=$' + sci_notation(parameters['c2']))

    fig.tight_layout()
    save_graph(save, fig)


def figure_appxC_gamma_effect(parameters: dict, xi_args: dict, gammas: list,
                              t_end: int = int(3e4), seed: int = 42,
                              figsize=FIGSIZE, save: str = ''):
    """ 2x2 diagram of the effect of gamma on the dynamics of the sentiment

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        noise arguments
    gammas : list
        four alternate values of gamma
    t_end : int
        total duration of the simulation
    seed : int
        random seed for numpy
    figsize : tuple
        Dimensions of the figure
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """

    assert len(gammas) == 4, "Only 4 values for 2x2 graph"

    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        parameters['gamma'] = gammas[i]
        ax = add_s_dynamics(parameters, xi_args, ax, t_end, seed)
        ax.set_title(r'$\gamma={:.0f}$'.format(gammas[i]))

    fig.tight_layout()
    save_graph(save, fig)


def figure_appxC_eps_effect(parameters: dict, xi_args: dict, eps_alt: float,
                            t_end: int = int(3e4), seed: int = 42,
                            figsize=FIGSIZE, save: str = ''):
    """ 2-part diagram of the dynamics for different epsilon
    LHS: Base case. RHS: alternate epsilon

    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    xi_args : dict
        noise arguments
    eps_alt : float
        alternate value of epsilon for the right-hand side
    t_end : int
        total duration of the simulation
    seed : int
        random seed for numpy
    figsize : tuple
        Dimensions of the figure
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """

    fig = plt.figure(figsize=figsize)
    ax_e1 = fig.add_subplot(1, 2, 1)
    ax_e2 = fig.add_subplot(1, 2, 2)

    ax_e1 = add_s_dynamics(parameters, xi_args, ax_e1, t_end, seed)
    ax_e1.set_title(r'$\varepsilon=$' + sci_notation(parameters['epsilon']))

    parameters['epsilon'] = eps_alt
    ax_e2 = add_s_dynamics(parameters, xi_args, ax_e2, t_end, seed)
    ax_e2.set_title(r'$\varepsilon=$' + sci_notation(parameters['epsilon']))

    fig.tight_layout()
    save_graph(save, fig)


def figure_sh_phase(parameters: dict, xi_args: dict, t_end: int = int(1e5),
                    minimum_period: int = 80, count: int = 7,
                    figsize: tuple = FIGSIZE, save: str = ''):
    """ (s,z) phase diagram
    Parameters
    ----------
    parameters : dict
        parameters to use for the demand limiting case
    t_end : int
        total duration of the simulation
    figsize : tuple
        figure size, default (PAGE_WIDTH, PAGE_WIDTH / 2)
    save : str (default '')
        name of the figure to save. If save='' it will show figure (slower).
        Figures are saved in pdf format
    """
    ds = DemandSolow(params=parameters, xi_args=xi_args)
    start = np.array([1, 0, 3, 0, 0, 0], dtype=float)
    start[0] = parameters['rho'] * start[2] - start[1]

    fig = plt.figure(figsize=figsize)
    kwargs = dict(linewidth=0.5, linestyle='-', zorder=1, alpha=0.75)
    ax_sh = fig.add_subplot(1, 2, 1)
    ax_stoch = fig.add_subplot(1, 2, 2)

    realisation = ds.simulate(start, seed=0, t_end=t_end)
    h = fourier_transformation(realisation.h, 100)
    s = fourier_transformation(realisation.s, 100)
    ax_stoch.plot(s, h, color='navy', linewidth=0.5, zorder=1)

    phase_series = []
    for i in np.linspace(-0.95, 0.95, count):
        for ii in [-1.0, 1.0]:
            start[3] = i
            start[4] = ii
            path = ds.simulate(start, t_end=t_end, xi=False)
            c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
            ax_sh.plot(path.s, path.h, color=c, **kwargs)
            start[3] = ii
            start[4] = i
            phase_series.append(ds.simulate(start, t_end=t_end, xi=False))
            c = 'navy' if path.s.iloc[-1] > 0 else 'firebrick'
            ax_sh.plot(path.s, path.h, color=c, **kwargs)

    for ax in [ax_sh, ax_stoch]:
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.set_ylabel(r'$s$', rotation=0)
        ax.set_xlabel(r'$h$')
        ax.set_title(' ')  # For size consistency
        ax.minorticks_on()
        add_critical_points(ds.get_critical_points(), ('s', 'h'), ax)

    fig.tight_layout()
    save_graph(save, fig)


if __name__ == '__main__':

    folder = 'figures/'
    if folder[:-1] not in os.listdir():
        os.mkdir(folder)

    noise = dict(decay=0.2, diffusion=1.0)
    base_case = dict(rho=0.33, epsilon=2.5e-5, tau_y=1e3, dep=2e-4,
                     tau_h=25, tau_s=250, c1=3, c2=7e-4, beta1=1.1,
                     beta2=1.0, gamma=2000, saving0=0.15, h_h=10)

    # Section 3.1 - Supply limit case
    print("Supply Limiting Case")
    name = folder + 'fig_supply_boundary_layer_approx.pdf'
    figure_supply_limit(params=None, const=1.5, t_end=1e5, save=name)

    # Section 3.2 - Demand limit case
    print("Demand Limiting Case")
    seed = 17
    cycle_params = copy.copy(base_case)
    cycle_params['gamma'] = 4000
    cycle_params['c2'] = 1e-4
    figsize_3d = (PAGE_WIDTH / 2, PAGE_WIDTH / 2)
    figsize_sz = (PAGE_WIDTH, PAGE_WIDTH / 3)

    name = folder + 'fig_demand_limit_base_phase_3d.pdf'
    figure_demand_3d_phase(base_case, int(5e4), name, figsize=figsize_3d)

    name = folder + 'fig_demand_limit_cycle_phase_3d.pdf'
    figure_demand_3d_phase_cycle(cycle_params, int(7e4), name,
                                 figsize=figsize_3d)

    name = folder + 'fig_demand_limit_sz_comparison.pdf'
    figure_demand_sz_comparison(base_case, cycle_params, int(5e4), name,
                                figsize=figsize_sz)

    name = folder + 'fig_demand_limit_base_series_3d.pdf'
    figure_demand_series_3d(base_case, noise, int(5e4), seed, 100, name)

    name = folder + 'fig_demand_limit_base_series_sy.pdf'
    figure_demand_series_sy(base_case, noise, int(1e5), seed, name, figsize_sz)

    # Section 4.2 - Asymptotic Analysis
    print("Asymptotic Analysis")
    name = folder + f'fig_results_asymp_supply.pdf'
    figure_asymp_supply(base_case, noise, t_end=int(1e7), seed=12, save=name)

    name = folder + f'fig_results_asymp_demand.pdf'
    figure_asymp_demand(base_case, noise, t_end=int(4e6), seed=12, save=name)

    name = folder + f'fig_results_asymp_general.pdf'
    figure_asymp_general(base_case, noise, t_end=int(4e6), seed=12, save=name)

    params = copy.copy(base_case)
    params['gamma'] = 1000
    params['c2'] = 2e-5
    name = folder + 'fig_results_asymp_purecycle.pdf'
    figure_asymp_purecycle(params, t_end=int(5e5), save=name)

    name = folder + 'fig_results_asymp_noisecycle.pdf'
    figure_asymp_noisecycle(params, noise, seed=12, t_end=int(5e5), save=name)

    # Section 4.3 - Medium-term Dynamics
    print("Medium-term dynamics")
    figsize = (PAGE_WIDTH, PAGE_WIDTH / 1.25)
    name = folder + f'fig_results_dynamics.pdf'
    figure_medium_term_dynamics(base_case, noise, t_end=int(5e5),
                                seed=12, save=name, figsize=figsize)

    # Section 4.4 - Business cycle characteristics
    # User should run simulations using simulations.py file first
    print("Business Cycles")
    simulation_folder = 'simulations_fluctuations_demand'
    name = folder + f'fig_results_cycle_duration_sentiment.pdf'
    binyears = 5
    figure_cycle_duration(base_case, simulation_folder, binyears=binyears,
                          save=name, measure='s')

    simulation_folder = 'simulations_fluctuations_general'
    name = folder + f'fig_results_cycle_duration_production.pdf'
    figure_cycle_duration(base_case, simulation_folder, binyears=binyears,
                          save=name, measure='ydc')

    simulation_folder = 'simulations_fluctuations_general'
    name = folder + f'fig_results_cycle_depth_production.pdf'
    figure_cycle_depth(base_case, simulation_folder, save=name, measure='ydc')

    simulation_folder = 'simulations_fluctuations_prevalence'
    name = folder + f'fig_results_prevalence.pdf'
    figure_hist_prevalence(base_case, simulation_folder, save=name, measure='ydc')

    # Appendix C - Setting up the parameters
    print("Appendix C")
    parameters = base_case.copy()
    parameters['c2'] = 1e-4
    figsize = (PAGE_WIDTH, PAGE_WIDTH / 3.5)
    gammas = {350: dict(limit_cycle=False,
                        sz_lines=[(-1.0, -0.6), (-1.0, -0.45), (-1.0, -0.3),
                                  (-1.0, 0.0), (-1.0, 0.15),
                                  (1.0, -0.45), (1.0, -0.3),
                                  (1.0, 0.0), (1.0, 0.15), (1.0, 0.3)],
                        s_lines=[(-1.0, -0.6), (-1.0, -0.45), (-1.0, -0.3),
                                 (-1.0, 0.0), (-1.0, 0.15),
                                 (-0.7, 0.0), (-0.7, 0.15),
                                 (0.7, -0.45), (0.7, -0.3),
                                 (1.0, -0.45), (1.0, -0.3),
                                 (1.0, 0.0), (1.0, 0.15), (1.0, 0.3)]),
              1000: dict(limit_cycle=True),
              4000: dict(limit_cycle=True),
              15000: dict(limit_cycle=False,
                          sz_lines=[(-1.0, 0.2), (1.0, -0.2)],
                          s_lines=[(-1.0, 0.25), (1.0, -0.25),
                                   (-0.6, 0.20), (0.6, -0.20),
                                   (-0.3, 0.15), (0.3, -0.15),
                                   (-0.0, 0.10), (0.0, -0.10)])}

    figure_appxC_limit_cycle(parameters, gammas, t_end=int(5e4),
                             lim=(-1.6, 1.6),
                             figsize=(PAGE_WIDTH, PAGE_WIDTH),
                             save=folder + f'fig_appxC_cycle_g.pdf')

    for g, kwargs in gammas.items():
        parameters['gamma'] = g
        name = folder + f'fig_appxC_cycle_g{g}.pdf'
        figure_appxC_sz_s(parameters, t_end=int(5e4), figsize=figsize,
                          save=name, lim=(-1.6, 1.6), **kwargs)

    figsize = (PAGE_WIDTH, PAGE_WIDTH / 3)
    name = folder + 'fig_appxC_base_c2_dynamics.pdf'
    figure_appxC_c2_effect(base_case.copy(), noise, c2_alt=9.5e-4,
                           t_end=int(2e5), seed=0, save=name, figsize=figsize)

    name = folder + 'fig_appxC_base_epsilon_asymmetry.pdf'
    figure_appxC_eps_effect(base_case.copy(), noise, eps_alt=7.5e-5,
                            t_end=int(2e5), seed=0, save=name, figsize=figsize)

    name = folder + 'fig_appxC_base_sh_phase.pdf'
    figure_sh_phase(base_case, noise, t_end=int(5e4), save=name, figsize=figsize)

    gammas = [300, 1500, 2500, 4500]
    figsize = (PAGE_WIDTH, PAGE_WIDTH / 1.5)
    name = folder + 'fig_appxC_base_gamma_dynamics.pdf'
    figure_appxC_gamma_effect(base_case.copy(), noise, gammas, t_end=int(2e5),
                              seed=0, save=name, figsize=figsize)
