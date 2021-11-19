"""
Simulation management file
--------------------------
Uses the python multiprocessing package to run all the simulations used in the
paper "Capital Demand Driven Business Cycles: Mechanism and Effects" by
Naumann-Woleske et al. 2021. There is a function for each set of simulations
that was run and used in the paper. The function strings specify which paper
outputs are related to these simulations.
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import os
import pickle
import sys
import time
from itertools import product
from multiprocessing import cpu_count, get_context
from typing import Callable

import numpy as np
import pandas as pd
# May be used when controlling simulations
from matplotlib import pyplot as plt

from demandSolow import DemandSolow
from solowModel import SolowModel


# ----------- UTILITY FUNCTIONS ----------- #

def generate_filename(p: dict, t_end: float, folder: str = 'computations/',
                      seed: int = None) -> str:
    """ Generate a name

    Parameters
    ----------
    p : dict
        Dictionary of all the parameters used in the model
    t_end : float
        Duration for which the simulation was run
    folder : str
        Folder in which to save the outputs
    seed : int (default = None)
        If there is a seed attached to the simulation output, specify in name

    Returns
    -------
    name : str
        Name of the file that is to be saved
    """
    parts = [
        'general',
        't{:05.0e}'.format(t_end),
        'g{:05.0f}'.format(p['gamma']),
        'e{:07.1e}'.format(p['epsilon']),
        'c1_{:03.1f}'.format(p['c1']),
        'c2_{:07.1e}'.format(p['c2']),
        'b1_{:03.1f}'.format(p['beta1']),
        'b2_{:03.1f}'.format(p['beta2']),
        'ty{:03.0f}'.format(p['tau_y']),
        'ts{:03.0f}'.format(p['tau_s']),
        'th{:02.0f}'.format(p['tau_h']),
        'lam{:01.2f}'.format(p['saving0']),
        'dep{:07.1e}'.format(p['dep']),
        'rho{:04.2f}'.format(p['rho']),
    ]

    name = '_'.join(parts)
    if seed is not None:
        return folder + name + '_seed_{:d}'.format(seed) + '.df'
    else:
        return folder + name + '.df'


def extract_info(filename: str, kind: str) -> dict:
    """ Extract the parameters from a filename. Used to confirm whether a file
    already exists, and thus does not need to be simulated again

    Parameters
    ----------
    filename : str
        Name of the file that is being analyzed
    kind : str
        Type of simulation that was run, can be either path (i.e one realisation
        of the model) or else asymptotics

    Returns
    -------
    info : dict
        Dictionary containing the key parameters of interest in the model
    """
    parts = filename.split('_')
    t = np.float(parts[1][1:])
    seed, gamma, c2 = None, None, None
    for i, part in enumerate(parts):
        if part[0] == 'g' and part[1] != 'e':
            gamma = int(part[1:])
        if part == 'c2':
            c2 = float(parts[i + 1])
        if 'seed' in part:
            seed = int(parts[i + 1][:-3])
    if kind == 'path':
        return t, gamma, c2, seed
    else:
        return t, gamma, c2


def cycle_timing(timeseries: pd.Series, timescale: int = 63,
                 kind: str = '2-period',
                 threshold: tuple = (-0.5, 0.5)) -> pd.DataFrame:
    """ Function to calculate the start and end-dates of business cycles. The
    rule that is applied is: expansion starts after two consecutive periods of
    expansion, and recession starts after two consecutive periods of contraction
    The period is defined as 63 days, i.e. a quarter

    Parameters
    ----------
    timeseries : pd.Series
        Time-series of data to analyse, typically economic production
    timescale : int
        Number of periods that correspond to "one-quarter"
    kind : str
        Method to apply (crossover, threshold, or 2-periods of consecutive)
    threshold : tuple
        Optionally the threshold to consider when looking at the crossover
        analysis

    Returns
    -------
    recessions : pd.DataFrame
        Dataframe with the periods of expansion and recession for each cycle
    """
    t = timeseries.shape[0]

    if kind == 'crossover':
        timescale = 10
        timeseries = timeseries.iloc[::timescale].copy(deep=True)
        growth = timeseries > 0

        expansions, recessions = [0], [0]
        for i in growth.index[1:]:
            # Switch negative to positive
            if all([~growth[i - timescale], growth[i]]):
                expansions.append(i)
            # Switch positive to negative
            elif all([growth[i - timescale], ~growth[i]]):
                recessions.append(i)

    elif kind == 'threshold':
        timescale = 10
        timeseries = timeseries.iloc[::timescale].copy(deep=True)

        expansions, recessions = [0], [0]
        upper = timeseries > max(threshold)
        lower = timeseries < min(threshold)

        for i in timeseries.index[1:]:
            # Upper threshold hit from below
            if all([~upper[i - timescale], upper[i]]):
                expansions.append(i)
            # Lower threshold from above
            if all([~lower[i - timescale], lower[i]]):
                recessions.append(i)

    else:
        timeseries = timeseries.iloc[::timescale].copy(deep=True)
        growth = timeseries.pct_change() > 0

        expansions, recessions = [0], [0]
        for i in growth.index[1:]:
            # Two quarters of positive growth
            if all([growth[i - timescale], growth[i]]):
                expansions.append(i)
            # Two quarters of negative growth
            elif all([~growth[i - timescale], ~growth[i]]):
                recessions.append(i)

    expansions = sorted(expansions)
    recessions = sorted(recessions)
    se = [(0, recessions[0])]
    exp, rec = se[0]

    while all([exp < t, rec < t]):
        expansions = expansions[expansions.index(exp):]
        recessions = recessions[recessions.index(rec):]
        exp = min([ix for ix in expansions if ix > rec] + [t])
        rec = min([ix for ix in recessions if ix > exp] + [t])
        se.append((exp, rec))

    return pd.DataFrame(se[1:], columns=['expansion', 'recession'])


def cycle_statistics(cycles: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    """ Determine relevant statistics for each of the cycles in the sample,
    including: peak index and value, trough index and value, growth in recession
    growth in expansion, and whether the utilisation rate was ever 100% in a
    cycle

    Parameters
    ----------
    cycles : pd.DataFrame
        dataframe with columns ['expansion', 'recession'] with indices for the
        timeseries
    path : pd.DataFrame

    Returns
    -------
    cycles : pd.DataFrame of the statistics
    """
    # Temporary column of the next expansion index
    cycles.loc[:, 'next_exp'] = cycles.expansion.shift(-1)
    cycles.loc[:, 'next_rec'] = cycles.recession.shift(-1)

    # Drop non-useful
    cycles.dropna(inplace=True)

    # Determine the duration of the cycles
    cycles.loc[:, 'duration'] = cycles.next_exp - cycles.expansion

    y = path.y

    # Peak to trough analysis
    def peak_index(row):
        return y.loc[row.loc['expansion']:row.loc['next_exp']].idxmax()

    cycles['pix'] = cycles.apply(peak_index, axis=1)
    cycles['peak'] = cycles.apply(lambda r: path.y.loc[r.loc['pix']], axis=1)

    def trough_index(row):
        return y.loc[row.loc['expansion']:row.loc['next_exp']].idxmin()

    cycles['tix'] = cycles.apply(trough_index, axis=1)
    cycles['trough'] = cycles.apply(lambda r: path.y.loc[r.loc['tix']], axis=1)

    # Growth rate analysis
    def rec_growth(row):
        return y.loc[row.loc['recession']:row.loc['next_rec']].diff().mean()

    def exp_growth(row):
        return y.loc[row.loc['expansion']:row.loc['next_exp']].diff().mean()

    cycles['rec_g'] = cycles.apply(rec_growth, axis=1)
    cycles['exp_g'] = cycles.apply(exp_growth, axis=1)

    # Whether utilisation reached 100% (supply limit reached)
    util = path.kd.div(path.ks)

    def ks_breach(row):
        if any(util.loc[row.loc['expansion']:row.loc['next_exp']] > 1.0):
            return 1.0
        else:
            return 0.0

    cycles['ks'] = cycles.apply(ks_breach, axis=1)

    return cycles


# ----------- MULTIPROCESSING FUNCTIONS ----------- #

def worker_general_path(args: list):
    """ Worker function for the multi-processing system that will run the
        Dynamic Solow model for a set of random seeds. Each run will be saved as
        a pandas DataFrame using the pickle module

        Parameters
        ----------
        args: list
            list of function arguments: the parameters (dict), the noise process
            arguments (dict), the seeds (list), the run-time (int), the starting
            values (np.ndarray of shape (7,1)), and the folder to save (str)
    """
    print(args)

    sm = SolowModel(params=args[0], xi_args=args[1])
    seeds, t_end, start, folder = args[2:]

    for i, seed in enumerate(seeds):
        path = sm.simulate(start, t_end=t_end, seed=seed)
        filename = generate_filename(args[0], t_end, folder=folder, seed=seed)
        file = open(filename, 'wb')
        pickle.dump(path, file)
        file.close()
    del sm


def worker_general_cycle(args: list):
    """ Worker function for the multi-processing system that will run the
        Dynamic Solow model for a set of random seeds. Each run will be saved as
        a pandas DataFrame using the pickle module

        Parameters
        ----------
        args: list
            list of function arguments: the parameters (dict), the noise process
            arguments (dict), the seeds (list), the run-time (int), the starting
            values (np.ndarray of shape (7,1)), and the folder to save (str)
    """
    sm = SolowModel(params=args[0], xi_args=args[1])
    seed, t_end, start, folder, smoothing = args[2:]
    path = sm.simulate(start, t_end=t_end, seed=seed)

    cycles = {}
    cycles['s'] = cycle_timing(path.s, timescale=10, kind='crossover')

    trend = path.y.iloc[0] + np.arange(t_end) * path.y.diff().mean()
    cycles['ydc'] = cycle_timing(path.y.sub(trend).dropna(), timescale=10, kind='crossover')

    for k, v in cycles.items():
        cycles[k] = cycle_statistics(v, path)
        cycles[k].columns = [c + '_' + k for c in cycles[k].columns]

    cycles = pd.concat([v for _, v in cycles.items()], axis=1)

    filename = generate_filename(args[0], t_end, folder=folder, seed=seed)
    file = open(filename, 'wb')
    pickle.dump(cycles, file)
    file.close()

    del sm


def worker_general_ratio(args: list):
    """ Worker function for the multi-processing system that will run the
        Dynamic Solow model for a set of random seeds. Each run will be saved as
        a pandas DataFrame using the pickle module

        Parameters
        ----------
        args: list
            list of function arguments: the parameters (dict), the noise process
            arguments (dict), the seeds (list), the run-time (int), the starting
            values (np.ndarray of shape (7,1)), and the folder to save (str)
    """
    sm = SolowModel(params=args[0], xi_args=args[1])
    seed, t_end, start, folder, smoothing = args[2:]
    path = sm.simulate(start, t_end=t_end, seed=seed)

    proportion = (path.ks > path.kd).sum() / t_end
    filename = generate_filename(args[0], t_end, folder=folder, seed=seed)
    file = open(filename, 'wb')
    pickle.dump(proportion, file)
    file.close()

    del sm


def worker_demand_cycle(args: list):
    """ Worker function for the multi-processing system that will run the
        Dynamic Solow model for a set of random seeds. Each run will be saved as
        a pandas DataFrame using the pickle module

        Parameters
        ----------
        args: list
            list of function arguments: the parameters (dict), the noise process
            arguments (dict), the seeds (list), the run-time (int), the starting
            values (np.ndarray of shape (7,1)), and the folder to save (str)
    """
    sm = DemandSolow(params=args[0], xi_args=args[1])
    seed, t_end, start, folder, smoothing = args[2:]
    path = sm.simulate(start, t_end=t_end, seed=seed)

    cycles = {}
    cycles['s'] = cycle_timing(path.s, timescale=10, kind='crossover')
    cycles['st04'] = cycle_timing(path.s, timescale=10, kind='threshold',
                                  threshold=(-0.4, 0.4))

    s = path.s.rolling(250).mean().dropna()
    cycles['ss1'] = cycle_timing(s, timescale=10, kind='crossover')
    cycles['ss1t04'] = cycle_timing(s, timescale=10, kind='threshold',
                                    threshold=(-0.4, 0.4))

    trend = path.y.iloc[0] + np.arange(t_end) * path.y.diff().mean()
    cycles['yd'] = cycle_timing(path.y.sub(trend).dropna(), timescale=10, kind='default')
    cycles['yd1'] = cycle_timing(path.y.rolling(250).mean().sub(trend).dropna(),
                                 timescale=63, kind='default')
    cycles['yd5'] = cycle_timing(path.y.rolling(1250).mean().sub(trend).dropna(),
                                 timescale=63, kind='default')
    cycles['yd10'] = cycle_timing(path.y.rolling(2500).mean().sub(trend).dropna(),
                                  timescale=63, kind='default')

    cycles['ydc'] = cycle_timing(path.y.sub(trend).dropna(), timescale=10, kind='crossover')
    cycles['ydc1'] = cycle_timing(path.y.rolling(250).mean().sub(trend).dropna(),
                                  timescale=63, kind='crossover')
    cycles['ydc5'] = cycle_timing(path.y.rolling(1250).mean().sub(trend).dropna(),
                                  timescale=63, kind='crossover')
    cycles['ydc10'] = cycle_timing(path.y.rolling(2500).mean().sub(trend).dropna(),
                                   timescale=63, kind='crossover')

    trend = path.y.rolling(300 * 250).mean().dropna()
    cycles['ymm'] = cycle_timing(path.y.sub(trend).dropna(), timescale=10, kind='default')
    cycles['ymm1'] = cycle_timing(path.y.rolling(250).mean().sub(trend).dropna(),
                                  timescale=63, kind='default')
    cycles['ymm5'] = cycle_timing(path.y.rolling(1250).mean().sub(trend).dropna(),
                                  timescale=63, kind='default')
    cycles['ymm10'] = cycle_timing(path.y.rolling(2500).mean().sub(trend).dropna(),
                                   timescale=63, kind='default')

    trend = path.y.rolling(300 * 250).mean().dropna()
    cycles['ymmc'] = cycle_timing(path.y.sub(trend).dropna(), timescale=10, kind='crossover')
    cycles['ymmc1'] = cycle_timing(path.y.rolling(250).mean().sub(trend).dropna(),
                                   timescale=63, kind='crossover')
    cycles['ymmc5'] = cycle_timing(path.y.rolling(1250).mean().sub(trend).dropna(),
                                   timescale=63, kind='crossover')
    cycles['ymmc10'] = cycle_timing(path.y.rolling(2500).mean().sub(trend).dropna(),
                                    timescale=63, kind='crossover')

    for k, v in cycles.items():
        cycles[k] = cycle_statistics(v, path)
        cycles[k].columns = [c + '_' + k for c in cycles[k].columns]

    cycles = pd.concat([v for _, v in cycles.items()], axis=1)

    filename = generate_filename(args[0], t_end, folder=folder, seed=seed)
    file = open(filename, 'wb')
    pickle.dump(cycles, file)
    file.close()

    del sm


def worker_demand_path(args: list):
    """ Worker function for the multi-processing system that will run the demand
        limiting case of the Dynamic Solow model (k = kd) for a set of random
        seeds. Each run will be saved as a pandas DataFrame using the pickle
        module

        Parameters
        ----------
        args: list
            list of function arguments: the parameters (dict), the noise process
            arguments (dict), the seeds (list), the run-time (int), the starting
            values (np.ndarray of shape (7,1)), and the folder to save (str)
        """
    # Initialise
    sm = DemandSolow(params=args[0], xi_args=args[1])

    seeds, t_end, start, folder = args[2:]

    for i, seed in enumerate(seeds):
        path = sm.simulate(start, t_end=t_end, seed=seed)

        p = dict(args[0])
        p['saving0'], p['dep'], p['h_h'] = 0, 0, 0

        filename = generate_filename(p, t_end, folder=folder, seed=seed)
        file = open(filename, 'wb')
        pickle.dump(path, file)
        file.close()
    del sm


def pool_mgmt(worker: Callable[[list], None], tasks: list, cpus: int = 20):
    """ Function that will execute all of the different workers submitted to it

    Parameters
    ----------
    worker : Callable[[list],None]
        the worker function that is to be applied, e.g. worker_path,
        worker_asymp, worker_demand_limit
    tasks : list
        the list for each worker to execute. This is a list of lists that have
        the relevant arguments for each of the workers. See their docstrings
    """
    n_tasks, t = len(tasks), time.time()
    cpus = min([cpu_count(), cpus])
    arg = (n_tasks, cpus, time.strftime("%H:%M:%S", time.gmtime(t)))
    print("Starting\t{} processes on {} CPUs at {}".format(*arg))

    # Launch multiprocessing, use tqdm to display a progress bar of tasks
    with get_context("spawn").Pool(processes=cpus) as pool:
        for _ in pool.map(worker, tasks):
            pass


def task_creator(variations: dict, folder: str, seeds: list, t_end: float,
                 start: np.ndarray, kind: str = 'asymptotic',
                 xi_args: None = None, smoothing: int = 2000):
    """ Function to set up the argument list to pass to the multiprocessing
    function.

    Parameters
    ----------
    variations : dict
        dictionary of parameter:list pairs that specify the variations of each
        parameter to use in simulations. e.g. gamma:[1000,2000,3000]
    folder : str
        location where the output will be saved
    seeds: list
        list of random seeds for which to run the simulation
    t_end : float
        duration of the simulations
    start : np.ndarray
        array of initial values y, ks, kd, s, h, g, xi
    kind : str
        type of simulation to run: path, asymptotic or demand limit
    xi_args : dict
        arguments for the ornstein-uhlenbeck process
    smoothing : int
        the number of periods that should be used when smoothing the output with
        a rolling average

    Returns
    -------
    tasks : list[list]
        list of lists with the tasks to pass to a worker function in the
        multiprocessing.
    """
    if xi_args is None:
        xi_args = dict(decay=0.2, diffusion=1.0)

    # Default parameters are set up
    params = dict(rho=0.33, epsilon=2.5e-5, tau_y=1e3, dep=2e-4,
                  tau_h=25, tau_s=250, c1=3, c2=7e-4, beta1=1.1,
                  beta2=1.0, gamma=2000, saving0=0.15, h_h=10)

    demand_params = dict(rho=0.33, epsilon=2.5e-5, tau_y=1e3, dep=2e-4,
                         tau_h=25, tau_s=250, c1=3, c2=7e-4, beta1=1.1,
                         beta2=1.0, gamma=2000, saving0=0.15, h_h=10)

    # Differentiated start for the pure demand limiting case
    demand_start = np.array([1, 0, 9, 0, 0, 0])
    demand_start[0] = np.exp(demand_start[1])

    # List of files that exist to avoid repeating simulations
    existing_files = [f for f in os.listdir(folder)
                      if '.df' in f
                      if not f.startswith('.')]
    tasks = []
    varied_parameters = list(variations.keys())

    # Iterate through all possible combinations of given parameters
    for tup in product(*list(variations.values())):
        # New dictionary object of simulation parameters
        p = dict(params)
        p_demand = dict(demand_params)
        for combo in zip(varied_parameters, tup):
            p[combo[0]] = combo[1]
            p_demand[combo[0]] = combo[1]

        # Generate a new task if there is no existing simulation yet
        if kind == 'path':
            for seed in seeds:
                filename = generate_filename(p, t_end, folder='', seed=seed)
                if filename not in existing_files:
                    arg = (p, xi_args, seeds, t_end, start, folder)
                    tasks.append(arg)
        elif kind == 'cycle':
            for seed in seeds:
                filename = generate_filename(p, t_end, folder='', seed=seed)
                if filename not in existing_files:
                    arg = (p, xi_args, seed, t_end, start, folder, smoothing)
                    tasks.append(arg)
        elif kind == 'asymptotic':
            filename = generate_filename(p, t_end, folder='')
            if filename not in existing_files:
                arg = (p, xi_args, seeds, t_end, start, folder)
                tasks.append(arg)
        elif kind == 'demand_limit':
            filename = generate_filename(p, t_end, folder='')
            if filename not in existing_files:
                arg = (p_demand, xi_args, seeds, t_end, demand_start, folder)
                tasks.append(arg)
        elif kind == 'demand_cycle':
            for seed in seeds:
                filename = generate_filename(p, t_end, folder='', seed=seed)
                if filename not in existing_files:
                    arg = (p_demand, xi_args, seed, t_end, demand_start, folder,
                           smoothing)
                    tasks.append(arg)
    if len(tasks) == 0:
        print("There are no tasks, it is likely all these files exist already")

    return tasks


# ----------- SIMULATIONS IMPLEMENTED ----------- #

def business_cycles(cpus: int = 20,
                    folder: str = 'simulations_fluctuations_general/'):
    """ Simulations used in the main part of the paper to demonstrate the
    business cycles observed (Section 4)

    Parameters
    ----------
    cpus : int      Number of cores to use for the parallel processing
    folder: str     Location to save the outputs to, must end with '/'
    """
    variations = dict(c1=[3], epsilon=[2.5e-5], gamma=[2000], c2=[7e-4])
    xi_args = dict(decay=0.2, diffusion=1.0)

    if folder[:-1] not in os.listdir():
        os.mkdir(folder)

    seeds = list(range(200))
    t_end = 1e7
    # Order: y, ks, kd, s, h, switch, xi
    start = np.array([1, 10, 9, 0.85, 0.5, 1, 0])
    start[0] = min(start[1], start[2]) / 3
    kind = 'cycle'

    tasks = task_creator(variations, folder, seeds, t_end, start, kind, xi_args)
    cpus = min([cpus, len(tasks), cpu_count()])
    pool_mgmt(worker_general_cycle, tasks, cpus=cpus)


def kd_prevalence(cpus: int = 20,
                  folder: str = 'simulations_fluctuations_prevalence/'):
    """ Simulations used in the main part of the paper to demonstrate the
    business cycles observed (Section 4)

    Parameters
    ----------
    cpus : int      Number of cores to use for the parallel processing
    folder: str     Location to save the outputs to, must end with '/'
    """
    variations = dict(c1=[3], epsilon=[2.5e-5], gamma=[2000], c2=[7e-4])
    xi_args = dict(decay=0.2, diffusion=1.0)

    if folder[:-1] not in os.listdir():
        os.mkdir(folder)

    seeds = list(range(30))
    t_end = 5e7
    # Order: y, ks, kd, s, h, switch, xi
    start = np.array([1, 10, 9, 0.85, 0.5, 1, 0])
    start[0] = min(start[1], start[2]) / 3
    kind = 'cycle'

    tasks = task_creator(variations, folder, seeds, t_end, start, kind, xi_args)
    cpus = min([cpus, len(tasks), cpu_count()])
    pool_mgmt(worker_general_ratio, tasks, cpus=cpus)


def demand_analysis(cpus: int = 20,
                    folder: str = 'simulations_fluctuations_demand/'):
    """ Simulations used in the assessment of the fluctuations that occur in the
    demand limiting case

    Parameters
    ----------
    cpus : int      Number of cores to use for the parallel processing
    folder: str     Location to save the outputs to, must end with '/'
    """
    variations = dict(c1=[3], epsilon=[2.5e-5], gamma=[2000], c2=[7e-4])
    xi_args = dict(decay=0.2, diffusion=1.0)

    if folder[:-1] not in os.listdir():
        os.mkdir(folder)

    seeds = list(range(50))
    t_end = 1e7
    # Order: y, z, kd, s, h, xi
    start = np.array([3, 0.01, 9, 0.85, 0.5, 0])
    kind = 'demand_cycle'

    tasks = task_creator(variations, folder, seeds, t_end, start, kind, xi_args)
    cpus = min([cpus, len(tasks), cpu_count()])
    pool_mgmt(worker_demand_cycle, tasks, cpus=cpus)


def test_analysis(cpus: int = 20,
                  folder: str = 'simulations_fluctuations_demand/'):
    """ Function for testing different combinations of parameters experimentally

    Parameters
    ----------
    cpus : int      Number of cores to use for the parallel processing
    folder: str     Location to save the outputs to, must end with '/'
    """

    print("Running")
    variations = dict(
        gamma=[2000],
        c2=np.arange(1e-4, 5e-4, 1e-4),
        c1=[3],
    )

    if folder[:-1] not in os.listdir():
        os.mkdir(folder)

    seeds = [1]
    t_end = 1e6
    # Order: y, ks, kd, s, h, switch, xi
    start = np.array([3, 10, 9, 0, 0, 1, 0])
    kind = 'path'

    tasks = task_creator(variations, folder, seeds, t_end, start, kind)
    cpus = min([cpus, len(tasks), cpu_count()])
    pool_mgmt(worker_general_path, tasks, cpus=cpus)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        globals()[sys.argv[1]](int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 2:
        globals()[sys.argv[1]](int(sys.argv[2]), sys.argv[3])
    else:
        globals()[sys.argv[1]]()
