from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from .pyleptophilicdm import LeptophilicDM
from .sampler import Sampler


def main():
    file_dir = os.path.dirname(__file__)


    model = LeptophilicDM(file_dir+"/config.csv")
    nwalkers = 20

    loc = (model.config.hi.values + model.config.lo.values ) / 2
    scale = (model.config.hi.values - model.config.lo.values ) / 10
    p0 = norm.rvs(size=(nwalkers,len(model.config)),loc=loc,scale=scale)

    sampler = Sampler(model.lnposterior,p0,["p1","p2","p3","p4","p5"],nwalkers)

    n_sample = 10000
    sampler.sample(n_sample,use_pool=False)

    sampler.save("test/")
    
    
if __name__ is "__main__":
    main()