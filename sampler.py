import emcee
from emcee import EnsembleSampler
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
from .scatter_matrix import scatter_matrix
from multiprocessing import Pool


uses_old_emcee = int(emcee.__version__.split(".")[0]) <= 2
if uses_old_emcee:
    raise RuntimeError(print("Your emcee version is {}. Use emcee3.".format(emcee.__version__)))

class Sampler:
    """
    wrapper of emcee.EnsembleSampler. 
    """
    def __init__(self,lnpost,p0,keys,nwalkers=120):
        """
        init
        """
        
        self.lnpost = lnpost
        blobs_dtype = float  # Note: Here dtype must be specified, otherwise an error happens. #[("lnlike",float),]
        self.sampler = EnsembleSampler(nwalkers,p0.shape[1],lnpost,blobs_dtype=blobs_dtype)  # NOTE: dtype must be list of tuple (not tuple of tuple)
        self.p0 = p0
        self.p_last = p0
        self.keys = keys
        self.ndim = len(keys)
        
        
    def reset_sampler(self):
        self.sampler.reset()
        
        
    def sample(self,n_sample,burnin=False,use_pool=False):
        """
        execute mcmc for given iteration steps.
        """
        desc = "burnin" if burnin else "sample"
        
        with Pool() as pool:
            self.sampler.pool = pool if use_pool else None
            iteration = tqdm(self.sampler.sample(self.p_last,iterations=n_sample),total=n_sample,desc=desc)
            for _ret in iteration:
                self.p_last = _ret.coords   # if uses_emcee3 else _ret[0]  # for emcee2
                lnposts     = _ret.log_prob # if uses_emcee3 else _ret[1]  # for emcee2
                iteration.set_postfix(lnpost_min=np.min(lnposts),lnpost_max=np.max(lnposts),lnpost_mean=np.mean(lnposts))
            if burnin:
                self.reset_sampler()
        
        
    def _save(self,fname_base):
        np.save(fname_base+"_chain.npy",self.sampler.get_chain())
        np.save(fname_base+"_lnprob.npy",self.sampler.get_log_prob())
        np.save(fname_base+"_lnlike.npy",self.sampler.get_blobs())

    
    def save(self,fname_base):
        '''
        Save MCMC results into "<fname_base>_chain/lnprob/lnlike.npy".
        If fname_base is like "your_directory/your_prefix", create "your_directory" before saving.
        '''
        dirname = os.path.dirname(fname_base)
        if dirname == "":
            self._save(fname_base)
        else:
            if not os.path.isdir(dirname): os.mkdir(dirname)
            self._save(fname_base)   
        
    
    

class Analyzer:
    
    def __init__(self,fname_base,keys,n_skipinit=0,n_sep=1):
        self.fname_base = fname_base
        self.keys = keys
        self._chain         = np.load(fname_base+"_chain.npy")
        self._lnprobability = np.load(fname_base+"_lnprob.npy")
        self._lnlike = np.load(fname_base+"_lnlike.npy")
        self.n_skipinit = n_skipinit
        self.n_sep = n_sep
    
    @property
    def ndim(self):
        return len(self.keys)
    
    @property
    def chain(self):
        return self._chain[self.n_skipinit::self.n_sep,:,:]
    
    @property
    def lnprobability(self):
        return self._lnprobability[self.n_skipinit::self.n_sep,:]
    
    @property
    def lnlike(self):
        return self._lnlike[self.n_skipinit::self.n_sep,:]
    
    @property
    def flatchain(self):
        return self.chain.reshape(-1,self.ndim)
    
    @property
    def flatlnprobability(self):
        return self.lnprobability.reshape(-1)
    
    @property
    def flatlnlike(self):
        return self.lnlike.reshape(-1)
    
    @property
    def df(self):
        _df = DataFrame(self.flatchain,columns=self.keys)
        _df["lnprob"] = self.flatlnprobability
        _df["lnlike"] = self.flatlnlike
        return _df
    
    def plot_chain(self,kwargs_subplots={},**kwargs):
        fig,ax = plt.subplots(self.ndim+2,**kwargs_subplots)
        for i in range(self.ndim):
            ax[i].plot(self.chain[:,:,i],**kwargs) # [nwalkers,nsample,ndim]
            ax[i].set_ylabel(self.keys[i])
        ax[self.ndim].plot(self.lnprobability,**kwargs) # [nwalkers,nsample,ndim]
        ax[self.ndim].set_ylabel("lnprob")
        ax[self.ndim+1].plot(self.lnlike,**kwargs) # [nwalkers,nsample,ndim]
        ax[self.ndim+1].set_ylabel("lnlike")
        
        
    def plot_hist(self,skip=0,n_sep=1,**kwargs):
        self.df.hist(**kwargs)
        
    
    def map_estimater(self):
        _i = self.df.lnprob.idxmax()
        return self.df.iloc[_i]
    

    def scatter_matrix(self,c,plot_axes="lower",hist_kwds=dict(bins=64,histtype="step",color="gray"),**kwargs):
        df = self.df.sort_values(c).reset_index(drop=True)
        #print(df)
        return scatter_matrix(df,plot_axes=plot_axes,
                              c=df[c].values,
                              hist_kwds = hist_kwds,
                              **kwargs)
                                
    
    
Analyser = Analyzer
