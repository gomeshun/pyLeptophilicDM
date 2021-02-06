#!/usr/bin/python
import numpy as np
import sympy


#SMのinput
vev = 246

def par_to_phys(par):

    # parameterの読み込み

    m_e_L = par['m_e_L']
    m_N   = par['m_N']
    m_e_R = par['m_e_R']

    m_chi   = par['m_chi']
    y_L     = par['y_L']
    y_R     = par['y_R']

    laphiRH = -0.161505 + 0.00149599*m_e_R + 0.0000129516*(m_e_R**2)  # higgs to gamma gamma
    laphiLH2 = (m_e_L**2 - m_N**2)/(vev**2)  
    
    # higgs to gamma gamma
    laphiLH = -0.161505 + 0.00149599*m_e_L + 0.0000129516*(m_e_L**2)-laphiLH2 
    
    if laphiRH > 1:
        laphiRH = 1
    
    if laphiLH > 1:
        laphiLH = 1
        
    # higgs invisible decay
    if m_N < 62:
        laphiLH = laphiLH2
    

    if min(m_N, m_e_L, m_e_R) < m_chi:
        return 'unstable'

    # electroweak precision measurement
    elif abs(laphiLH2) > 0.0078493 + 0.000402212*m_e_L:
        return 'unstable'

    # vacuum stability
    elif min(laphiLH,laphiRH) < 0:
        return 'unstable'


    else:
        Mx = m_chi
        MSLE = m_e_L
        MSRE = m_e_R
        MSNE = m_N

        MSLM = m_e_L
        MSRM = m_e_R
        MSNM = m_N

        MSLT = m_e_L
        MSRT = m_e_R
        MSNT = m_N

        lamHSLE = vev*(laphiLH+laphiLH2)
        lamHSNE = vev*(laphiLH-laphiLH2)
        lamHHSLE = laphiLH + laphiLH2
        lamHHSNE = laphiLH - laphiLH2
        lamHR   = laphiRH

        yL = y_L
        yR = y_R

        list_val = [Mx,MSLE,MSRE,MSNE,MSLM,MSRM,MSNM,MSLT,MSRT,MSNT,
                    lamHSLE,lamHSNE,lamHHSLE,lamHHSNE,lamHR,
                    yL,yR]

        list_name = ['Mx','MSLE','MSRE','MSNE','MSLM','MSRM','MSNM','MSLT','MSRT','MSNT',
                    'lamHSLE','lamHSNE','lamHHSLE','lamHHSNE','lamHR',
                    'yL','yR']
        return dict(zip(list_name,list_val))
#print(par_to_phys({'m_e_L':160,'m_e_R':200,'m_chi':110,'y_L':1,'y_R':1,'laphiLH2':0.2}))