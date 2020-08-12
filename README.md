# pyLeptophlicDM
Python codes to explore Leptophilic DM

# Requirements
pyLeptophilicDM uses:

- emcee (3.0.2)

and other common python packages (numpy, pandas, etc.)

# Class and function
## `model.py`
Define `Model` class to descibe Leptophilic DM model.
## `pyleptophilicdm.py`
Implementation of Leptophilic DM model (preliminary)
## `sampler.py`
Define `Sampler` class, a wrapper class of `emcee.EnsembleSampler`, and `Analyzer` class to plot the result of `Sampler` class.
## `run.py`
Example code to run MCMC sampling.
