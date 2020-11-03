from scipy.stats import norm, uniform, loguniform
import matplotlib.pyplot as plt
import os
from numpy import arange, log10, empty
import yaml
import datetime

from .model import LeptophilicDM
from .sampler import Sampler


def generate_p0(config_fname):
    file_dir = os.path.dirname(__file__) + "/"
    path_config_fname = file_dir + config_fname

    config = LeptophilicDM(path_config_fname).config
    adopts_logprior = config.prior=="log"
    _a = config.lo.values
    _b = config.hi.values
    a = _a[adopts_logprior]
    b = _b[adopts_logprior]
    loc = _a[~adopts_logprior]
    scale = (_b-_a)[~adopts_logprior]

    #print(loc,scale)

    p0 = empty((nwalkers,len(config)))
    #print(p0.shape)
    p0[:,~adopts_logprior]  = uniform.rvs(size=(nwalkers,(~adopts_logprior).sum()),loc=loc,scale=scale) 
    p0[:,adopts_logprior] = loguniform.rvs(size=(nwalkers,adopts_logprior.sum()),a=a,b=b) 
    
    return p0


def run(fname_prefix,
        p0=None,
        nwalkers = 20,
        nsample = 1000,
        nburnin = 100,
        enable_vacuum_stability=False,
        enable_collider_const=False,
        enable_micromegas_likeli=False,
        enable_micromegas_prior=False,
        enable_gm2=False,
        use_pool=False,
        config_fname = "config.csv",
        project_name = "LeptophilicDM",
        dir_models = "/from_taisuke/models",
        overwrite=False
       ):
    """
    fname_prefix: 
        if like "test", save chains into
            test_chain.npy
        etc. if like "test/", save
            test/_chain.npy
        etc.
    """
    executed_time = datetime.datetime.now()
    
    file_dir = os.path.dirname(__file__) + "/"
    path_config_fname = file_dir + config_fname
    
    configs = [enable_vacuum_stability,
               enable_collider_const,
               enable_micromegas_likeli,
               enable_micromegas_prior,
               enable_gm2,
               use_pool]
    
    config_int = (2**arange(len(configs)) * configs).sum()
    
    fname_prefix += f"_nwalkers={nwalkers}_nsample={nsample}_nburnin={nburnin}_config={config_int}"
    print(f"Results will be exported to {fname_prefix}_.gz")

    model = LeptophilicDM(path_config_fname,
                          enable_vacuum_stability,
                          enable_collider_const,
                          enable_micromegas_likeli,
                          enable_micromegas_prior,
                          enable_gm2,
                          project_name,
                          dir_models
                         )
    #nwalkers = 20

    #loc = (model.config.hi.values + model.config.lo.values ) / 2
    #scale = (model.config.hi.values - model.config.lo.values ) * 0.1
    #p0 = norm.rvs(size=(nwalkers,len(model.config)),loc=loc,scale=scale)
    
    # Note: uniform(loc,scale) = [loc, loc+scale]
    # Here 
    

    if p0 is None:
        config = LeptophilicDM(path_config_fname).config
        adopts_logprior = config.prior=="log"
        _a = config.lo.values
        _b = config.hi.values
        a = _a[adopts_logprior]
        b = _b[adopts_logprior]
        loc = _a[~adopts_logprior]
        scale = (_b-_a)[~adopts_logprior]

        #print(loc,scale)

        p0 = empty((nwalkers,len(config)))
        #print(p0.shape)
        p0[:,~adopts_logprior]  = uniform.rvs(size=(nwalkers,(~adopts_logprior).sum()),loc=loc,scale=scale) 
        p0[:,adopts_logprior] = loguniform.rvs(size=(nwalkers,adopts_logprior.sum()),a=a,b=b) 

    #pnames = model.param_names
    #idx_mchi = pnames[pnames=="m_chi"].index[0]
    #loc[idx_mchi] = 500
    
    #idx_mchi = pnames[pnames=="m_phi_L"].index[0]
    #loc[idx_mchi] = 1000
    
    #idx_mchi = pnames[pnames=="m_phi_R"].index[0]
    #loc[idx_mchi] = 1000

    sampler = Sampler(model.lnposterior,p0,nwalkers)

    #nsample = 1000
    sampler.sample(nburnin,use_pool=use_pool,burnin=True)
    sampler.sample(nsample,use_pool=use_pool)

    sampler.save(fname_prefix)
    sampler.save_pickle(fname_prefix,overwrite=overwrite)
    
    log = dict(fname_prefix=fname_prefix,
        p0 = p0,
        nwalkers = nwalkers,
        nsample = nsample,
        nburnin = nburnin,
        enable_vacuum_stability = enable_vacuum_stability,
        enable_collider_const = enable_collider_const,
        enable_micromegas_likeli = enable_micromegas_likeli,
        enable_micromegas_prior = enable_micromegas_prior,
        enable_gm2 = enable_gm2,
        use_pool = use_pool,
        config_fname = config_fname,
        project_name = project_name,
        dir_models = dir_models,
        overwrite = overwrite,
        executed_time = executed_time)
    
    
    
    return sampler
    
    
if __name__ is "__main__":
    pass
