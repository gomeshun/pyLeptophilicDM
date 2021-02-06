#!/usr/bin/python
import numpy as np


#SMのinput
vev = 246

# par = {'m_e_L':0,'m_e_R':0,'m_chi':0,'y_L':0,'y_R':0,'laphiLH2':0}
# を引数にもち、physical massなどの値を計算して、辞書型で出力する。もともとのparameter数は9個。

def par_to_phys(par):

    # parameterの読み込み

    m_e_L = par['m_e_L']
    m_e_R = par['m_e_R']
    m_chi   = par['m_chi']
    y_L     = par['y_L']
    y_R     = par['y_R']
    m_nu = par['m_nu']
    
    laphiLH2 = (m_e_L**2 - m_nu**2)/vev**2


    if min(m_e_L**2 - laphiLH2*(vev**2),m_e_L**2, m_e_R**2) < m_chi**2:
        return 'unstable'
    

    if abs(laphiLH2) > 1: return 'too large coupling'

    else:

        Mx = m_chi
        MSLE = m_e_L
        MSRE = m_e_R
        MSNE = m_nu #np.sqrt(m_e_L**2 - laphiLH2*(vev**2))

        MSLM = m_e_L
        MSRM = m_e_R
        MSNM = m_nu #np.sqrt(m_e_L**2 - laphiLH2*(vev**2))

        MSLT = m_e_L
        MSRT = m_e_R
        MSNT = m_nu #np.sqrt(m_e_L**2 - laphiLH2*(vev**2))

        lamHSLE = vev*(1+laphiLH2)
        lamHSNE = vev*(1-laphiLH2)
        lamHHSLE = 1 + laphiLH2
        lamHHSNE = 1 - laphiLH2

        yL = y_L
        yR = y_R









        list_val = [Mx,MSLE,MSRE,MSNE,MSLM,MSRM,MSNM,MSLT,MSRT,MSNT,
                    lamHSLE,lamHSNE,lamHHSLE,lamHHSNE,
                    yL,yR]

        list_name = ['Mx','MSLE','MSRE','MSNE','MSLM','MSRM','MSNM','MSLT','MSRT','MSNT',
                    'lamHSLE','lamHSNE','lamHHSLE','lamHHSNE',
                    'yL','yR']
        return dict(zip(list_name,list_val))
#print(par_to_phys({'m_e_L':160,'m_e_R':200,'m_chi':110,'y_L':1,'y_R':1,'laphiLH2':0.2}))
