from abc import ABCMeta, abstractmethod
from pandas import DataFrame
from numpy import inf, nan



class Model(metaclass=ABCMeta):
    
    
    
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
        
