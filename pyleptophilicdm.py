from numpy import inf
from emcee import EnsembleSampler
import numpy as np
from pandas import read_csv

import time  # for debugging

from .model import Model

from scipy.stats import norm  # for sample


class LeptophilicDM(Model):
    
    
    def __init__(self,fname):
        self.config = read_csv(fname)
    
    
    @property
    def param_names(self):
        """
        parameter names.
        
        Note: Paramters with mass-dimension should defined in GeV unit.
        """
        return self.config.name
    
    

    def lnlikelihood(self,array):
        """
        return log_likelihood value for given input array.
        
        input:
            array: numpy.ndarray, shape = (n_params,)
        """
        #### example ####
        lower = self.config.lo.values
        upper = self.config.hi.values
        lnls = norm.logpdf(array,loc=(lower+upper)/2,scale=(upper-lower)/2)
        #lnls = 0
        ################
        #time.sleep(1e-1)
        return np.sum(lnls)
    
    
    
    def lnprior(self,array):
        """
        return log_prior value for given input array.
        
        input:
            array: numpy.ndarray, shape = (n_params,)
        """
        
        lower = self.config.lo.values
        upper = self.config.hi.values
        prior_type = self.config.prior.values
        
        if not np.all((lower < array) & (array < upper)):
            return -inf
        else:
            lnps = -np.log(array[prior_type=="log"])
            return np.sum(lnps)
        
        



