# pyLeptophlicDM
Python codes to explore Leptophilic DM

# Requirements
pyLeptophilicDM uses:

- emcee (3.0.2)

and other common python packages (numpy, pandas, etc.)

# How to install pyLeptophilicDM
- install `git` on your computer.
- 'git clone' this project.

# Modules
## `model.py`
Define `Model` class to descibe Leptophilic DM model.
## `pyleptophilicdm.py`
Implementation of Leptophilic DM model (preliminary)
## `sampler.py`
Define `Sampler` class, a wrapper class of `emcee.EnsembleSampler`, and `Analyzer` class to plot the result of `Sampler` class.
## `run.py`
Example code to run MCMC sampling.
## `graphical_prior.py`
Define `Polygon` class, utilized to define a hard-cut prior by using a given csv file.

# Other Files
## `hepdata...csv`
LHC constraints downloaded from [https://www.hepdata.net/record/ins1750597?version=1&table=Exclusion%20contour%20(obs)%203](https://www.hepdata.net/record/ins1750597?version=1&table=Exclusion%20contour%20(obs)%203)
## `config.csv`
An example file to define the configuration of Leptophilic DM model such as 
  - parameter names (on-code or TeX)
  - parameter prior ranges
  - prior types
