from abc import ABCMeta, abstractmethod
from pandas import DataFrame,Series,read_csv
from numpy import inf, nan
import numpy as np

import os

from .polygon import Polygon

from scipy.stats import norm



class Model(metaclass=ABCMeta):
    '''
	Base class to define models.
	'''
    
    
    @abstractmethod
    def __init__(self):
        pass
    
    
    @abstractmethod
    def lnlikelihood(self,array):
        """
        return log_likelihood value for given input array.
        
        input:
            array: numpy.ndarray, shape = (n_params,)
        """
        pass
    
    
    @abstractmethod
    def lnprior(self,array):
        """
        return log_prior value for given input array.
        
        input:
            array: numpy.ndarray, shape = (n_params,)
        """
        pass
    
    
    def lnposterior(self,array):
        """
        return unnormalized log_posterior (=log_likelihood + log_prior) value for given input array.
        
        input:
            array: numpy.ndarray, shape = (n_params,)
        """
        lnp = self.lnprior(array)
        if lnp == -inf:  
            return -inf,-inf  # Note: avoid likelihood calcullation (usually time-consuming)
        
        else:
            lnl = self.lnlikelihood(array)
            return lnl+lnp, lnl
			
	
	 
class LeptophilicDM(Model):
    '''
	Implementation of Leptophilic DM model.
	'''
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
        
		# Hard-cut prior
        if not np.all((lower < array) & (array < upper)):
            return -inf
        
        # LHC constraints
		# NOTE: This is just an example. It must be updated.
        params = Series(array,index=self.param_names)
        if self.lhc_constraints.includes(params[["m_phi_L","m_chi"]].values.reshape(1,-1)):
            return -inf
        if self.lhc_constraints.includes(params[["m_phi_R","m_chi"]].values.reshape(1,-1)):
            return -inf
        
		# log-prior
        lnps = -np.log(array[prior_type=="log"]) # d(log x) = x^-1 dx = exp(-log x) dx
		#lnps += np.zeros(array[prior_type=="flat"].shape)
        return np.sum(lnps)
        
        



