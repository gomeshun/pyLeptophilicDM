from numpy import inf
from emcee import EnsembleSampler
import numpy as np
from pandas import read_csv, Series
import os

import time  # for debugging

from .model import Model
from .graphical_prior import Polygon

from scipy.stats import norm  # for sample


class LeptophilicDM(Model):
    
    
    def __init__(self,fname):
        self.config = read_csv(fname)
        
        points = np.loadtxt(os.path.dirname(__file__)+"/hepdata.89413.v1_t23.csv",delimiter=",")
        self.lhc_constraints = Polygon(points)  # (x,y) = (slepton mass, neutralino maxx) [GeV]
    
    
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
        #lower = self.config.lo.values
        #upper = self.config.hi.values
        #lnls = norm.logpdf(array,loc=(lower+upper)/2,scale=(upper-lower)/2)
        lnls = 0
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
        
        
        params = Series(array,index=self.param_names)
        if self.lhc_constraints.includes(params[["m_phi_L","m_chi"]].values.reshape(1,-1)):
            return -inf
        if self.lhc_constraints.includes(params[["m_phi_R","m_chi"]].values.reshape(1,-1)):
            return -inf
        
        lnps = -np.log(array[prior_type=="log"])
        return np.sum(lnps)
        
        



