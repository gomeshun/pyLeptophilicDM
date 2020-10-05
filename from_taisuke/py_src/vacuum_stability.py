#!/usr/bin/python
import numpy as np
from matplotlib.pyplot import plot,hist,scatter, show
from scipy.optimize import minimize

from functools import lru_cache
# L:stau doublet
# R:stau singlet
# H:higgs doublet

# potential = ml^2 L^2 + mr^2 R^2 + mu^2 H^2  + lambda_l L^4 +lambda_r R^4 + lambda_lr L^2 R^2 + lambda_h H^4
#           + (A*m_tau*(H L R) + h.c) + lambda_hl1 H^2 L^2 + lambda_hl2 (H ta H  L ta L) + lambda_hr H^2 R^2
# using gauge transformation set R=a, L=(b,c+di), H=(0,e), where (a,b,c,d,e) are real number.

vev = 246

class Potential:
    
    def __init__(self,ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
        self.ml_square = ml_square
        self.mr_square = mr_square
        self.A = A
        self.lamhl1 = lamhl1
        self.lamhl2 = lamhl2
        self.lamhr = lamhr
    
    def __call__(self,a,b,c,d,e):
        mh = 125
        m_tau = 1.777
        lambda_h = mh**2/(2*vev**2)
        mu = - lambda_h*vev**2
        mr = self.mr_square
        ml = self.ml_square
        Aterm = self.A*m_tau
        lambda_hl1 = self.lamhl1
        lambda_hl2 = self.lamhl2
        lambda_hr = self.lamhr
        lambda_l = 1
        lambda_r = 1
        lambda_lr = 1
        f=b**2+c**2+d**2

        return (mr*a**2 + ml*f + mu*e**2 + lambda_l * f**2 + lambda_r * a**4 +lambda_lr*a**2*f+ lambda_h*e**4 + 2*Aterm*(a*c*e) + lambda_hl1*e**2*f + lambda_hl2*(c**2+d**2-b**2) +lambda_hr*e**2*a**2)


def search_vacuum1(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    """
    check that SM vacuum is a local minimum of the potential
    """
    potential = Potential(ml_square,mr_square,A,lamhl1,lamhl2,lamhr)
    
    a = 500 # set start point
    b = 500 # set start point
    c = 500 # set start point
    d = 500 # set start point
    e = 500 # set start point

    for i in range(2000):
        current_pot = potential(a,b,c,d,e)
        delta_a = 3*np.random.rand()
        delta_b = 3*np.random.rand()
        delta_c = 3*np.random.rand()
        delta_d = 3*np.random.rand()
        delta_e = 3*np.random.rand()
        if potential(a+delta_a,b,c,d,e)<current_pot :
            a = a + delta_a
        else :
            a = a - delta_a
        if potential(a,b+delta_b,c,d,e)<current_pot :
            b = b + delta_b
        else :
            b = b - delta_b
        if potential(a,b,c+delta_c,d,e)<current_pot :
            c = c + delta_c
        else :
            c = c - delta_c
        if potential(a,b,c,d+delta_d,e)<current_pot :
            d = d + delta_d
        else :
            d = d - delta_d
        if potential(a,b,c,d,e+delta_e)<current_pot :
            e = e + delta_e
        else :
            e = e - delta_e

    for i in range(100):
        current_pot = potential(a,b,c,d,e)
        delta_a = 0.1*np.random.rand()
        delta_b = 0.1*np.random.rand()
        delta_c = 0.1*np.random.rand()
        delta_d = 0.1*np.random.rand()
        delta_e = 0.1*np.random.rand()
        if potential(a+delta_a,b,c,d,e)<current_pot :
            a = a + delta_a
        else :
            a = a - delta_a
        if potential(a,b+delta_b,c,d,e)<current_pot :
            b = b + delta_b
        else :
            b = b - delta_b
        if potential(a,b,c+delta_c,d,e)<current_pot :
            c = c + delta_c
        else :
            c = c - delta_c
        if potential(a,b,c,d+delta_d,e)<current_pot :
            d = d + delta_d
        else :
            d = d - delta_d
        if potential(a,b,c,d,e+delta_e)<current_pot :
            e = e + delta_e
        else :
            e = e - delta_e
    
    vacuum_pot = potential(0,0,0,0,vev/np.sqrt(2))
    #print(a,b,c,d,e,vacuum_pot,potential(a,b,c,d,e))

    if potential(a,b,c,d,e) < vacuum_pot :
        return "meta stable or unstable"
    else :
        return "stable"

    
def search_vacuum2(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    """
    check that the VEV of the standard model is really a local minimum of the potential
    """
    potential = Potential(ml_square,mr_square,A,lamhl1,lamhl2,lamhr)
    vacuum_pot = potential(0,0,0,0,vev/np.sqrt(2))
    
    p = []
    for i in range(300):
        a,b,c,d,e = [np.random.randint(-1,1)*np.random.rand(),
                     np.random.randint(-1,1)*np.random.rand(),
                     np.random.randint(-1,1)*np.random.rand(),
                     np.random.randint(-1,1)*np.random.rand(),
                     246/np.sqrt(2)+np.random.randint(-1,1)*np.random.rand()]
        p.append([a,b,c,d,e])
        if potential(a,b,c,d,e) < vacuum_pot:
            return "unstable"
        else:
            print(potential(a,b,c,d,e) - vacuum_pot)
        
    else:
        return "stable"
        
        

@lru_cache(maxsize=1)
def is_local_min(ml_square,mr_square,A,lamhl1,lamhl2,lamhr,*,n_iter=1000):
    """
    pythonic modification of "search_vacuum2".
    check that the VEV of the standard model is really a local minimum of the potential
    """
    potential = Potential(ml_square,mr_square,A,lamhl1,lamhl2,lamhr)
    vacuum_pot = potential(0,0,0,0,vev/np.sqrt(2))
        
    p0 = np.array([0,0,0,0,vev/np.sqrt(2)]).reshape(5,1)  # p0.shape = (5,1)
    delta_p = 2 * np.random.rand(5,n_iter) - 1  # -1 <= delta_p < 1, delta_p.shape = (5,n_iter)
    p = (p0 + delta_p)
    pot_val = potential(*p)
    if np.any(pot_val < vacuum_pot):  # If there are any points giving smaller potential values
        exists_smaller_points = (pot_val<vacuum_pot)
        return "unstable"
    else: 
        return "stable"
    
       
        
def stability(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    if search_vacuum1(ml_square,mr_square,A,lamhl1,lamhl2,lamhr) == "meta stable or unstable":
        #if search_vacuum2(ml_square,mr_square,A,lamhl1,lamhl2,lamhr) == "unstable" :
        if is_local_min(ml_square,mr_square,A,lamhl1,lamhl2,lamhr) == "unstable" :
            return "unstable"
        else :
            return "meta stable"
    else :
        return "stable"

    