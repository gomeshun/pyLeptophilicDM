import os
import subprocess
from glob import glob
from abc import ABCMeta, abstractmethod

import numpy as np
from pandas import DataFrame,Series,read_csv
from numpy import inf, nan, log, log10, exp, abs, min, max
from scipy.stats import norm


from .from_taisuke.py_src.g_minus_2 import log_likelihood_g_minus_2
from .from_taisuke.py_src.vacuum_stability import stability
from .from_taisuke.py_src.par_to_phys import par_to_phys
from .pymicromegas import PyMicrOmegas, Project
from .polygon import Polygon




############ config ############
__filedir__ = os.path.dirname(__file__) + "/"

#### for collider constraints ####
hepdata_fname      = __filedir__ + "HEPData-ins1750597-v1-csv.tar.gz"  # 13 TeV
hepdata_fname_8tev = __filedir__ + "EHPData-ins1286761-v1-csv.tar.gz"  #  8 TeV
hepdata_dir        = __filedir__ + os.path.basename(hepdata_fname).split(".")[0] + "/"
hepdata_dir_8tev   = __filedir__ + os.path.basename(hepdata_fname_8tev).split(".")[0] + "/"
hepdata_degen_dir  = __filedir__ + "coll_degenerated/"

constraint_se_l  = hepdata_dir + "Exclusioncontour(obs)7.csv"
constraint_se_r  = hepdata_dir + "Exclusioncontour(obs)8.csv"
constraint_smu_l = hepdata_dir + "Exclusioncontour(obs)10.csv"
constraint_smu_r = hepdata_dir + "Exclusioncontour(obs)11.csv"

constraint_se_l_8tev  = hepdata_dir_8tev + "Table41.csv"
constraint_se_r_8tev  = hepdata_dir_8tev + "Table39.csv"
constraint_smu_l_8tev = hepdata_dir_8tev + "Table47.csv"
constraint_smu_r_8tev = hepdata_dir_8tev + "Table45.csv"

constraint_se_l_degen      = hepdata_degen_dir + "eL_limit_degen.dat"
constraint_se_r_degen      = hepdata_degen_dir + "eR_limit_degen.dat"
constraint_smu_l_degen     = hepdata_degen_dir + "muL_limit_degen.dat"
constraint_smu_r_degen     = hepdata_degen_dir + "muR_limit_degen.dat"
#constraint_lep_se_r_degen  = hepdata_degen_dir + "LEP_eR.dat"
#constraint_lep_smu_r_degen = hepdata_degen_dir + "LEP_muR.dat"

constraint_lep_se  = hepdata_degen_dir + "LEP_data_eR.dat"
constraint_lep_smu = hepdata_degen_dir + "LEP_data_muR.dat"


hepdict = {
    "slepton" : r"m($\tilde{l}$) [GeV]",
    "neutralino" : r"m($\tilde{\chi}_{1}^{0}$) [GeV]",
}

#### for micromegas ####
mdl_file_paths = glob(__filedir__+"from_taisuke/models/*.mdl")
dof_fname   = __filedir__ + "pymicromegas/eos2020.dat"
dof_fname_1 = __filedir__ + "pymicromegas/eos2020_err1.dat"
dof_fname_2 = __filedir__ + "pymicromegas/eos2020_err2.dat"
dof_fname_11 = __filedir__ + "pymicromegas/eos2020_err11.dat"
dof_fname_22 = __filedir__ + "pymicromegas/eos2020_err22.dat"

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
    def __init__(self,fname,delim_whitespace=False):
        #### extract HEPData if called for the first time ####
        # --> disabled (extracted file uploaded on the repository)
        
        #if not os.path.exists(hepdata_dir):
        #    print(f"{hepdata_dir} does not exist yet. Start unzip {hepdata_fname} ....")
        #    subprocess.run(f"tar -xzf {hepdata_fname}",shell=True,cwd=__filedir__,encoding="UTF-8",check=True)
        
        #points = np.loadtxt(os.path.dirname(__file__)+"/hepdata.89413.v1_t23.csv",delimiter=",")
        #self.lhc_constraints = Polygon(points)  # (x,y) = (slepton mass, neutralino maxx) [GeV]
        points = read_csv(fname,comment="#",delim_whitespace=delim_whitespace)
        self.polygon = Polygon(points.values)
    
    @property
    def points(self):
        return self.polygon.points
    
    @property
    def x(self):
        return self.polygon.x
    
    @property
    def y(self):
        return self.polygon.y
    
    def reset_points(self,new_points):
        self.polygon.reset_points(new_points)
        
    
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
    def __init__(self,config_fname,
                 enable_vacuum_stability  = False,
                 enable_collider_const    = False,
                 enable_micromegas_likeli = False,
                 enable_micromegas_prior  = False,
                 enable_gm2               = False
                ):
        """
        initialize Leptophilic DM model.
        
        Note: stablility condition is always enabled. Other conditions (collider, micromegas, g-2) is optional.
        """
        self.config = read_csv(config_fname)
        
        self.enable_vacuum_stability  = enable_vacuum_stability
        self.enable_collider_const    = enable_collider_const
        self.enable_micromegas_likeli = enable_micromegas_likeli
        self.enable_micromegas_prior  = enable_micromegas_prior
        self.enable_gm2               = enable_gm2
        
        
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
        self.coll_se_l = Collider(constraint_se_l)
        self.coll_se_r = Collider(constraint_se_r)
        self.coll_smu_l = Collider(constraint_smu_l)
        self.coll_smu_r = Collider(constraint_smu_r)
        
        self.coll_se_l_8tev = Collider(constraint_se_l_8tev)
        self.coll_se_r_8tev = Collider(constraint_se_r_8tev)
        self.coll_smu_l_8tev = Collider(constraint_smu_l_8tev)
        self.coll_smu_r_8tev = Collider(constraint_smu_r_8tev)
        
        self.coll_se_l_degen      = Collider(constraint_se_l_degen,delim_whitespace=True)
        self.coll_se_r_degen      = Collider(constraint_se_r_degen,delim_whitespace=True)
        self.coll_smu_l_degen     = Collider(constraint_smu_l_degen,delim_whitespace=True)
        self.coll_smu_r_degen     = Collider(constraint_smu_r_degen,delim_whitespace=True)
        
        self.coll_lep_se  = Collider(constraint_lep_se,delim_whitespace=True)
        self.coll_lep_smu = Collider(constraint_lep_smu,delim_whitespace=True)
        
        # extend collider constraints to mx = 0 (y axis),
        # otherwise some narrow regions are remained to be un-excluded
        for coll in [self.coll_se_l,self.coll_se_r,self.coll_smu_l,self.coll_smu_r,
                     self.coll_se_l_8tev,self.coll_se_r_8tev,self.coll_smu_l_8tev,self.coll_smu_r_8tev
                    ]:
            new_points = np.array([
                [coll.x[0],0],
                *coll.points,
                [coll.x[-1],0]
            ])
            coll.reset_points(new_points)
            
        
        for coll in [self.coll_lep_se,self.coll_lep_smu]:
            new_points = np.array([
                [0,0],
                *coll.points,
                [coll.x[-1],0]
            ])
            coll.reset_points(new_points)
            
            
        for coll_degen in [self.coll_se_l_degen,self.coll_se_r_degen,
                           self.coll_smu_l_degen,self.coll_smu_r_degen]:
            new_points = np.array([
                [0,coll_degen.y[0]],
                *coll_degen.points,
                [0,coll_degen.y[-1]]
            ])
            coll_degen.reset_points(new_points)
            
    
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
    
    
    def collider_excludes(self,par_physical):
        m_sle = par_physical["MSLE"]  # left-handed slepton mass
        m_sre = par_physical["MSRE"]  # right-handed slepton mass
        m_slm = par_physical["MSLM"]  # left-handed smuon mass
        m_srm = par_physical["MSRM"]  # right-handed smuon mass
        m_x   = par_physical["Mx"]    # dark matter mass
        
        if self.coll_se_l.excludes(m_sle,m_x): return True
        if self.coll_se_r.excludes(m_sre,m_x): return True
        if self.coll_smu_l.excludes(m_slm,m_x): return True
        if self.coll_smu_r.excludes(m_srm,m_x): return True
        
        if self.coll_se_l_8tev.excludes(m_sle,m_x): return True
        if self.coll_se_r_8tev.excludes(m_sre,m_x): return True
        if self.coll_smu_l_8tev.excludes(m_slm,m_x): return True
        if self.coll_smu_r_8tev.excludes(m_srm,m_x): return True
        
        #if self.coll_lep_se_r_degen.excludes(m_sre,m_sre-m_x): return True
        #if self.coll_lep_smu_r_degen.excludes(m_srm,m_srm-m_x): return True
        if self.coll_lep_se.excludes(m_sle,m_x): return True
        if self.coll_lep_smu.excludes(m_slm,m_x): return True
        if self.coll_lep_se.excludes(m_sre,m_x): return True
        if self.coll_lep_smu.excludes(m_srm,m_x): return True
        
        if self.coll_se_l_degen.excludes(m_sle,m_sle-m_x): return True
        if self.coll_se_r_degen.excludes(m_sre,m_sre-m_x): return True
        if self.coll_smu_l_degen.excludes(m_slm,m_slm-m_x): return True
        if self.coll_smu_r_degen.excludes(m_srm,m_srm-m_x): return True
        
        
        return False
    
    
    def consistents_with_relic(self,par_physical):
        omega_obs = 0.120  # PLANCK(2018) 0.120 += 0.001
        
        par_physical = par_physical.to_dict()
        omega_11 = self.micromegas(par_physical,flags=["OMEGA"],dof_fname=dof_fname_11)["Omega"]
        omega_22 = self.micromegas(par_physical,flags=["OMEGA"],dof_fname=dof_fname_22)["Omega"]
        omegas = [omega_11,omega_22]
        
        if min(omegas) < 0 : return False 
        
        if min(omegas) <= omega_obs <= max(omegas): 
            return True
        else:
            return False
        
    
    def lnl_relic_abundance(self,par_physical):
        omega_obs = 0.120  # PLANCK(2018) 0.120 += 0.001
        ln_omega_obs = log(omega_obs)
        par_physical = par_physical.to_dict()
        
        omega   = self.micromegas(par_physical,flags=["OMEGA"],dof_fname=dof_fname)["Omega"]
        if omega < 0: return -inf
        ln_omega = log(omega)
        
        ln_omega_1 = log(self.micromegas(par_physical,flags=["OMEGA"],dof_fname=dof_fname_1)["Omega"])
        ln_omega_2 = log(self.micromegas(par_physical,flags=["OMEGA"],dof_fname=dof_fname_2)["Omega"])
        ln_omega_12 = [ln_omega_1,ln_omega_2]
        
        ln_omega_lo = min(ln_omega_12)
        ln_omega_hi = max(ln_omega_12)
        
        #if not (ln_omega_lo < ln_omega < ln_omega_hi): return -inf
        d_ln_omega = max(abs(ln_omega_12 - ln_omega))
        
        if d_ln_omega == 0: return -inf
        #print(dict(loc=ln_omega,scale=d_ln_omega))
        return norm.logpdf(ln_omega_obs,loc=ln_omega,scale=d_ln_omega)
        
        
        
    def lnl_gm2(self,par_physical):
        pass
    

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
        
        
        par_physical = self.to_par_physical(array)
        if type(par_physical) == str:
            raise RuntimeError(f"\"par_physical\" is string \"{par_physical}\"" + "\n" + f"\array: {array}")
        
        lnl = 0
        
        if self.enable_micromegas_likeli: lnl += self.lnl_relic_abundance(par_physical)
        
        # g-2 constraint
        if self.enable_gm2: lnl += self.lnl_gm2(par_physical)
        
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
        
        # vacuum stability
        if self.enable_vacuum_stability and isinstance(par_physical,str): return -inf
        
        # collider constraints
		# NOTE: This is just an example. It must be updated.
        #if self.lhc_constraints.includes(params[["m_phi_L","m_chi"]].values.reshape(1,-1)):
        #    return -inf
        #if self.lhc_constraints.includes(params[["m_phi_R","m_chi"]].values.reshape(1,-1)):
        #    return -inf
        
        if self.enable_collider_const and self.collider_excludes(par_physical): return -inf
        
        # relic density
        if self.enable_micromegas_prior and not self.consistents_with_relic(par_physical): return -inf
        
		# log-prior
        lnps = -np.log(array[prior_type=="log"]) # d(log x) = x^-1 dx = exp(-log x) dx
		#lnps += np.zeros(array[prior_type=="flat"].shape)
        return np.sum(lnps)
        
        


        
        


        
        