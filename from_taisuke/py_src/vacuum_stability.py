#!/usr/bin/python
import numpy as np
# L:stau doublet
# R:stau singlet
# H:higgs doublet

# potential = ml^2 L^2 + mr^2 R^2 + mu^2 H^2  + lambda_l L^4 +lambda_r R^4 + lambda_lr L^2 R^2 + lambda_h H^4
#           + (A*m_tau*(H L R) + h.c) + lambda_hl1 H^2 L^2 + lambda_hl2 (H ta H  L ta L) + lambda_hr H^2 R^2
# using gauge transformation set R=a, L=(b,c+di), H=(0,e), where (a,b,c,d,e) are real number.

def search_vacuum1(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    """
    check that SM vacuum is a local minimum of the potential
    """
    vev = 246
    mh = 125
    m_tau = 1.777
    lambda_h = mh**2/(2*vev**2)
    mu = - lambda_h*vev**2
    mr = mr_square
    ml = ml_square
    Aterm = A*m_tau
    lambda_hl1 = lamhl1
    lambda_hl2 = lamhl2
    lambda_hr = lamhr
    lambda_l = 1
    lambda_r = 1
    lambda_lr = 1
    def pot(a,b,c,d,e):
        f=b**2+c**2+d**2
        return (mr*a**2 + ml*f + mu*e**2 + lambda_l * f**2 + lambda_r * a**4 +lambda_lr*a**2*f+ lambda_h*e**4 +
        2*Aterm*(a*c*e) + lambda_hl1*e**2*f + lambda_hl2*(c**2+d**2-b**2) +lambda_hr*e**2*a**2)
    a = 500 # set start point
    b = 500 # set start point
    c = 500 # set start point
    d = 500 # set start point
    e = 500 # set start point

    for i in range(2000):
        current_pot = pot(a,b,c,d,e)
        delta_a = 3*np.random.rand()
        delta_b = 3*np.random.rand()
        delta_c = 3*np.random.rand()
        delta_d = 3*np.random.rand()
        delta_e = 3*np.random.rand()
        if pot(a+delta_a,b,c,d,e)<current_pot :
            a = a + delta_a
        else :
            a = a - delta_a
        if pot(a,b+delta_b,c,d,e)<current_pot :
            b = b + delta_b
        else :
            b = b - delta_b
        if pot(a,b,c+delta_c,d,e)<current_pot :
            c = c + delta_c
        else :
            c = c - delta_c
        if pot(a,b,c,d+delta_d,e)<current_pot :
            d = d + delta_d
        else :
            d = d - delta_d
        if pot(a,b,c,d,e+delta_e)<current_pot :
            e = e + delta_e
        else :
            e = e - delta_e

    for i in range(100):
        current_pot = pot(a,b,c,d,e)
        delta_a = 0.1*np.random.rand()
        delta_b = 0.1*np.random.rand()
        delta_c = 0.1*np.random.rand()
        delta_d = 0.1*np.random.rand()
        delta_e = 0.1*np.random.rand()
        if pot(a+delta_a,b,c,d,e)<current_pot :
            a = a + delta_a
        else :
            a = a - delta_a
        if pot(a,b+delta_b,c,d,e)<current_pot :
            b = b + delta_b
        else :
            b = b - delta_b
        if pot(a,b,c+delta_c,d,e)<current_pot :
            c = c + delta_c
        else :
            c = c - delta_c
        if pot(a,b,c,d+delta_d,e)<current_pot :
            d = d + delta_d
        else :
            d = d - delta_d
        if pot(a,b,c,d,e+delta_e)<current_pot :
            e = e + delta_e
        else :
            e = e - delta_e
    vacuum_pot = pot(0,0,0,0,246/np.sqrt(2))
    #print(a,b,c,d,e,vacuum_pot,pot(a,b,c,d,e))

    if pot(a,b,c,d,e) < vacuum_pot :
        return "meta stable or unstable"
    else :
        return "stable"

def search_vacuum2(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    """
    check that the VEV of the standard model is really a local minimum of the potential
    """
    vev = 246
    mh = 125
    m_tau = 1.777
    lambda_h = mh**2/(2*vev**2)
    mu = - lambda_h*vev**2
    mr = mr_square
    ml = ml_square
    Aterm = A*m_tau
    lambda_hl1 = lamhl1
    lambda_hl2 = lamhl2
    lambda_hr = lamhr
    lambda_l = 1
    lambda_r = 1
    lambda_lr = 1
    def pot(a,b,c,d,e):
        f=b**2+c**2+d**2
        return (mr*a**2 + ml*f + mu*e**2 + lambda_l * f**2 + lambda_r * a**4 +lambda_lr*a**2*f+ lambda_h*e**4 +
        2*Aterm*(a*c*e) + lambda_hl1*e**2*f + lambda_hl2*(c**2+d**2-b**2) +lambda_hr*e**2*a**2)
    vacuum_pot = pot(0,0,0,0,246/np.sqrt(2))

    for i in range(300):
        a,b,c,d,e = np.random.randint(-1,1)*np.random.rand(),np.random.randint(-1,1)*np.random.rand(),np.random.randint(-1,1)*np.random.rand(),np.random.randint(-1,1)*np.random.rand(),246/np.sqrt(2)+np.random.randint(-1,1)*np.random.rand()
        if pot(a,b,c,d,e)< vacuum_pot:
            return "unstable"
            break
        if  i == 299 :
            return "stable"

def stability(ml_square,mr_square,A,lamhl1,lamhl2,lamhr):
    if search_vacuum1(ml_square,mr_square,A,lamhl1,lamhl2,lamhr) == "meta stable or unstable":
        if search_vacuum2(ml_square,mr_square,A,lamhl1,lamhl2,lamhr) == "unstable" :
            return "unstable"
        else :
            return "meta stable"
    else :
        return "stable"
