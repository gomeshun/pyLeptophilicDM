from abc import ABCMeta, abstractmethod
from pandas import DataFrame,Series,read_csv
from numpy import inf, nan
import numpy as np
from .from_taisuke.py_src.g_minus_2 import log_likelihood_g_minus_2
from .from_taisuke.py_src.vacuum_stability import stability
from .from_taisuke.py_src.par_to_phys import par_to_phys
from .pymicromegas import PyMicrOmegas, Project

import os
import subprocess
from glob import glob

from .polygon import Polygon

from scipy.stats import norm



############ config ############
__filedir__ = os.path.dirname(__file__) + "/"

#### for collider constraints ####
hepdata_fname  = __filedir__ + "HEPData-ins1750597-v1-csv.tar.gz"
hepdata_dir    = __filedir__ + os.path.basename(hepdata_fname).split(".")[0]

constraint_l_selectron = __filedir__ + "Exclusioncontour(obs)7.csv"
constraint_r_selectron = __filedir__ + "Exclusioncontour(obs)8.csv"
constraint_l_smuon = __filedir__ + "Exclusioncontour(obs)10.csv"
constraint_r_smuon = __filedir__ + "Exclusioncontour(obs)11.csv"

hepdict = {
    "slepton" : r"m($\tilde{l}$) [GeV]",
    "neutralino" : r"m($\tilde{\chi}_{1}^{0}$) [GeV]",
}

#### for micromegas ####
mdl_file_paths = glob(__filedir__+"from_taisuke/models/*.mdl")

################################


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
			

            
class Collider:
    def __init__(self,fname):
        #### extract HEPData if called for the first time ####
        if not os.path.exists(hepdata_dir):
            print(f"{hepdata_dir} does not exist yet. Start unzip {hepdata_fname} ....")
            subprocess.run(f"tar -xzf {hepdata_fname}",shell=True,cwd=__filedir__,encoding="UTF-8",check=True)
        
        #points = np.loadtxt(os.path.dirname(__file__)+"/hepdata.89413.v1_t23.csv",delimiter=",")
        #self.lhc_constraints = Polygon(points)  # (x,y) = (slepton mass, neutralino maxx) [GeV]
        self.points = read_csv(fname,comment="#")
        self.polygon = Polygon(self.points.values)
    
    
    def excludes(self,x,y):
        """
        x: float
        y: float
        """
        point = np.array([x,y])
        return self.polygon.includes(point)[0]  # parameters are excluded when the polygon includes points
        

            
	 
class LeptophilicDM(Model):
    '''
	Implementation of Leptophilic DM model.
	'''
    def __init__(self,fname):
        self.config = read_csv(fname)
        
        
        #### initilalize project if called for the first time ####
        mo = PyMicrOmegas()
        if not mo.project_exists("LeptophilicDM"):
            micromegas = mo.load_project("LeptophilicDM")  
            mdl_file_paths = glob(os.path.dirname(__file__) + "/from_taisuke/models/*.mdl")
            micromegas.load_mdl_files(mdl_file_paths)
            micromegas.compile()
            
            # initialize project
            print(micromegas({},["OMEGA"]))
    
        self.micromegas = Project("LeptophilicDM")
        
        #### collider constraints ####
        self.coll_l_selectron = Collider(constraint_l_selectron)
        self.coll_r_selectron = Collider(constraint_r_selectron)
        self.coll_l_smuon = Collider(constraint_l_smuon)
        self.coll_r_smuon = Collider(constraint_r_smuon)
      
    
    @property
    def param_names(self):
        """
        parameter names.
        
        Note: Paramters with mass-dimension should defined in GeV unit.
        """
        return self.config.name
    
    
    def to_par(self,array):
        """
        make parameter dictionary (Series) from input array.
        """
        return Series(array,index=self.param_names)
    
    def to_par_physical(self,array):
        """
        ['Mx','MSLE','MSRE','MSNE','MSLM','MSRM','MSNM','MSLT','MSRT','MSNT',
                    'lamHSLE','lamHSRE','lamHSNE','lamHHSLE','lamHHSRE','lamHHSNE',
                    'lamHSLM','lamHSRM','lamHSNM','lamHHSLM','lamHHSRM','lamHHSNM',
                    'lamHSLT','lamHSLRT','lamHSRLT','lamHSRT','lamHSNT',
                    'lamHHSLT','lamHHSLRT','lamHHSRLT','lamHHSRT','lamHHSRT','lamHHSNT',
                    'yL','yR','yLT','yLRT','yRLT','yRT']
        """
        ret_dict = par_to_phys(self.to_par(array))
        if ret_dict == "unstable": 
            return "unstable"
        else: 
            return Series(ret_dict)
    
    

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
        #lnls = 0
        ################
        #time.sleep(1e-1)
        
        par = self.to_dict(array)
        par_physical = par_to_phys(par)
        if par_physical == "unstable": return -inf
        
        lnl = 0
        # vacuum stability
        #lnl = stability(???)
        
        #other constraints....
        #lnl = ...
        
        return lnl
    
    
    
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
        
        par_physical = self.to_par_physical(array)
        if par_physical == "unstable": return -inf
        
        # LHC constraints
		# NOTE: This is just an example. It must be updated.
        #if self.lhc_constraints.includes(params[["m_phi_L","m_chi"]].values.reshape(1,-1)):
        #    return -inf
        #if self.lhc_constraints.includes(params[["m_phi_R","m_chi"]].values.reshape(1,-1)):
        #    return -inf
        
		# log-prior
        lnps = -np.log(array[prior_type=="log"]) # d(log x) = x^-1 dx = exp(-log x) dx
		#lnps += np.zeros(array[prior_type=="flat"].shape)
        return np.sum(lnps)
        
        


        
        


        
        