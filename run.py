from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from .model import LeptophilicDM
from .sampler import Sampler


def run():
    file_dir = os.path.dirname(__file__)


    model = LeptophilicDM(file_dir+"/config.csv")
    nwalkers = 20

    loc = (model.config.hi.values + model.config.lo.values ) / 2
    scale = (model.config.hi.values - model.config.lo.values ) / 10
    p0 = norm.rvs(size=(nwalkers,len(model.config)),loc=loc,scale=scale)
    
    #pnames = model.param_names
    #idx_mchi = pnames[pnames=="m_chi"].index[0]
    #loc[idx_mchi] = 500
    
    #idx_mchi = pnames[pnames=="m_phi_L"].index[0]
    #loc[idx_mchi] = 1000
    
    #idx_mchi = pnames[pnames=="m_phi_R"].index[0]
    #loc[idx_mchi] = 1000

    sampler = Sampler(model.lnposterior,p0,model.param_names,nwalkers)

    n_sample = 10000
    sampler.sample(n_sample,use_pool=False)

    sampler.save("test/")
    
    
if __name__ is "__main__":
    main()
