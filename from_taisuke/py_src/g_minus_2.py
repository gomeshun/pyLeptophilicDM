#!/usr/bin/python
import numpy as np

def fN(x, y):
    return x*y*(
    (-3+x+y+x*y)/(((x-1)**2)*((y-1)**2))
    +(2*x*np.log(x)/((x-y)*((x-1)**3)))
    -(2*y*np.log(y)/((x-y)*((y-1)**3)))
    )

m_mu=0.105 # muon mass
vev =246 # vacuum expectation value
# lagrangian = ((A*m_mu*H*L*R)+h.c.)

def a_mu(mx, ml, mr, A, yl, yr):
    return (-yl*yr/(16*np.pi**2)
    * m_mu*mx*(A*(vev/np.sqrt(2))*m_mu)/(ml**2*mr**2)
    *fN(ml**2/mx**2, mr**2/mx**2)
    )

delta_a_mu = 26.1 * 10**(-10) # diff between experiment and SM
sigma_a_mu = 8.0 * 10**(-10) # standard deviation of a_mu

def log_likelihood_g_minus_2(mx, ml, mr, A, yl, yr):
    return -(a_mu(mx, ml, mr, A, yl, yr)-delta_a_mu)**2/(2*sigma_a_mu**2)
