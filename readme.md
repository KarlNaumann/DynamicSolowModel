# The Dynamic Solow Model

This is the accompanying code for the 2021 paper **Capital Demand Driven
Business Cycles: Mechanism and Effects** by Naumann-Woleske et al.

Links to the paper: [karlnaumann.com](https://karlnaumann.com/research/),
[arxiv](https://arxiv.org/abs/2110.00360),
[ssrn](https://papers.ssrn.com/abstract=3933586)

Abstract:
We develop a tractable macroeconomic model that captures dynamic behaviors
across multiple timescales, including business cycles. The model is anchored in
a dynamic capital demand framework reflecting an interactions-based process
whereby firms determine capital needs and make investment decisions on a micro
level. We derive equations for aggregate demand from this micro setting and
embed them in the Solow growth economy. As a result, we obtain a closed-form
dynamical system with which we study economic fluctuations and their impact on
long-term growth. For realistic parameters, the model has two attracting
equilibria: one at which the economy contracts and one at which it expands.
This bi-stable configuration gives rise to quasiperiodic fluctuations,
characterized by the economy’s prolonged entrapment in either a contraction or
expansion mode punctuated by rapid alternations between them. We identify the
underlying endogenous mechanism as a coherence resonance phenomenon. In
addition, the model admits a stochastic limit cycle likewise capable of
generating quasiperiodic fluctuations; however, we show that these fluctuations
cannot be realized as they induce unrealistic growth dynamics. We further find
that while the fluctuations powered by coherence resonance can cause substantial
excursions from the equilibrium growth path, such deviations vanish in the long
run as supply and demand converge.

## Installation
To use the Dynamic Solow model repository, please either clone or fork
the repository. The models were built with Python 3.9 utilizing Cython for
faster execution of the models

To install all required packages use the commands
````
# using pip
pip install -r pip_requirements.txt

# using Conda
conda create --name <env_name> --file conda_requirements.txt
````

Note that this makes use of numdifftools, which is not natively available in the
conda environment. Thus when using conda, one also needs to run
````
pip install numdifftools
````

Once these packages have been installed, it will be necessary to compile the
Cython code that is used to execute the simulations for the dynamic system. To
do so, execute the following commands:
````
cd cython_base
python setup.py build_ext --inplace
````

## Usage
There is an example jupyter notebook that details how to use both the demand
and general case of the Dynamic Solow Model to generate simulations and phase
diagrams

The simulations.py file provides the basic setup to run large scale simulations
in parallel. There are several functions available to be called:
1. **business_cycles**: this function simulates the general case and saves a set
of pickled dataframes that contain statistics on the individual cycles in the
sentiment and the production. Specifically: duration, peak index, peak value,
trough index, trough value, growth rate during recession, growth rate during
expansion, boolean whether a utilisation rate kd/ks of 100% was achieved.
2. **kd_prevalence**: simulates the general case and saves the proportion of
time during which capital supply was larger than demand
3. **demand_analysis**: same as the business_cycles function but focused on the
demand case of the Dynamic Solow Model. This also includes the timing of the
cycle under different methods, and with different detrending and smoothing
proceduress
4. **test_analysis**: testing function that simulates and saves a series of
paths taken by the model


## Replication
To replicate the figures presented in the paper. The user should run first the
simulations.py file with the business_cycles, kd_prevalence, and demand_analysis
functions to generate the simulation files. The code is as follows:

````
python simulations.py business_cycles 20 simulations_fluctuations_general/
python simulations.py kd_prevalence 20 simulations_fluctuations_prevalence/
python simulations.py demand_analysis 20 simulations_fluctuations_demand/
````
where the number 20 is an optional argument controlling how many CPUs to use in
the parallel processing and the second argument is the folder where these should
be saved (it must include '/').

Once this is completed, the user can execute graphs.py by running
```python graphs.py```, which will generate all of the graphs into a ```figures/```
folder. If you wish to specify a different folder, this can be done within the code.

## Performance
On an AMD Ryzen 5950X with 32GB of RAM, 200 simulations of T=1,000,000 timesteps
sampled at 0.1 timesteps, with 20 processors takes approximately 9 minutes
