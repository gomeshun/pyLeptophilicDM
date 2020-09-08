#!/usr/bin/python
import numpy as np
from .vacuum_stability import search_vacuum2, is_local_min
import sympy


# 対角化の計算
def diag(x):
    eig = np.linalg.eig(x)
    e = eig[0]
    if eig[1][0][0] < 0:
        p = -eig[1]
    else:
        p = eig[1]
    ip = np.linalg.inv(p)

    return [e, p, ip]

#SMのinput
vev = 246
tau_mass = 1.77

# par = {'m_e_L':0,'m_e_R':0,'m_chi':0,'A':0,'y_L':0,'y_R':0,'laphiLH1':0,'laphiLH2':0,'laphiRH':0}
# を引数にもち、physical massなどの値を計算して、辞書型で出力する。もともとのparameter数は9個。

def par_to_phys(par):

    # parameterの読み込み

    m_e_L = par['m_e_L']
    m_e_R = par['m_e_R']
    m_chi   = par['m_chi']
    A       = par['A']
    y_L     = par['y_L']
    y_R     = par['y_R']
    laphiLH1= par['laphiLH1']
    laphiLH2= par['laphiLH2']
    laphiRH = par['laphiRH']

    #m_e_Lとm_e_Rからm_phi_Lとm_phi_Rへの変換
    m_phi_L = m_e_L**2 - (laphiLH1*(vev**2)/2 + laphiLH2*(vev**2)/2)
    m_phi_R = m_e_R**2 - laphiRH*(vev**2)/2

    # massなどの計算
    mass_mat = diag(np.array([[m_phi_L + laphiLH1*(vev**2)/2 + laphiLH2*(vev**2)/2, A*tau_mass*vev/np.sqrt(2) ],
                    [A*tau_mass*vev/np.sqrt(2), m_phi_R +laphiRH*(vev**2)/2]]))


    Mx = m_chi
    MSLE = m_phi_L + laphiLH1*(vev**2)/2 + laphiLH2*(vev**2)/2
    MSRE = m_phi_R + laphiRH*(vev**2)/2
    MSNE = m_phi_L + laphiLH1*(vev**2)/2 - laphiLH2*(vev**2)/2

    MSLM = m_phi_L + laphiLH1*(vev**2)/2 + laphiLH2*(vev**2)/2
    MSRM = m_phi_R + laphiRH*(vev**2)/2
    MSNM = m_phi_L + laphiLH1*(vev**2)/2 - laphiLH2*(vev**2)/2

    MSLT = mass_mat[0][0]
    MSRT = mass_mat[0][1]
    MSNT = m_phi_L + laphiLH1*(vev**2)/2 - laphiLH2*(vev**2)/2

    
    if min([MSLE,MSRE,MSNE,MSLM,MSRM,MSNM,MSLT,MSRT,MSNT]) < Mx**2 or min([MSLE,MSRE,MSNE,MSLM,MSRM,MSNM,MSLT,MSRT,MSNT,Mx]) < 0:
        return 'unstable'
    
    #if search_vacuum2(m_phi_L,m_phi_R,A,laphiLH1,laphiLH2,laphiRH) == 'unstable' :
    if is_local_min(m_phi_L,m_phi_R,A,laphiLH1,laphiLH2,laphiRH) == 'unstable' :
         return 'unstable'
    

    else:

        # Calculation of physical mass

        MSLE = np.sqrt(MSLE)
        MSRE = np.sqrt(MSRE)
        MSNE = np.sqrt(MSNE)

        MSLM = np.sqrt(MSLM)
        MSRM = np.sqrt(MSRM)
        MSNM = np.sqrt(MSNM)

        MSLT = np.sqrt(MSLT)
        MSRT = np.sqrt(MSRT)
        MSNT = np.sqrt(MSNT)

        # Caluculation of interaction

        sympy.var('SLE_plus SRE_plus SNE_plus SLE_minus SRE_minus SNE_minus')
        sympy.var('SLM_plus SRM_plus SNM_plus SLM_minus SRM_minus SNM_minus')
        sympy.var('diag_SLT_plus diag_SRT_plus SNT_plus diag_SLT_minus diag_SRT_minus SNT_minus')
        sympy.var('h')

        SLT_minus = np.dot(mass_mat[1][0] , np.array([diag_SLT_minus,diag_SRT_minus]))
        SRT_minus = np.dot(mass_mat[1][1] , np.array([diag_SLT_minus,diag_SRT_minus]))
        SLT_plus = np.dot(mass_mat[1][0] , np.array([diag_SLT_plus,diag_SRT_plus]))
        SRT_plus = np.dot(mass_mat[1][1] , np.array([diag_SLT_plus,diag_SRT_plus]))

        SL_minus = [SLE_minus,SLM_minus,SLT_minus]
        SL_plus = [SLE_plus,SLM_plus,SLT_plus]
        SR_minus = [SRE_minus,SRM_minus,SRT_minus]
        SR_plus = [SRE_plus,SRM_plus,SRT_plus]
        SN_minus = [SNE_minus,SNM_minus,SNT_minus]
        SN_plus = [SNE_plus,SNM_plus,SNT_plus]

        higgs = (vev + h)/np.sqrt(2)
        lepton_mass =[0,0,tau_mass]
        lagrangian =[0,0,0]

        for i in range(3):
            lagrangian[i] = (A * lepton_mass[i] * higgs * SL_minus[i] * SR_plus[i] + A * lepton_mass[i] * higgs * SL_plus[i] * SR_minus[i]
                        + laphiLH1 * SL_minus[i] * SL_plus[i] * higgs * higgs
                        + laphiLH2 * SL_minus[i] * SL_plus[i] * higgs * higgs
                        + laphiLH1 * SN_minus[i] * SN_plus[i] * higgs * higgs
                        - laphiLH2 * SN_minus[i] * SN_plus[i] * higgs * higgs
                        + laphiRH * SR_minus[i] * SR_plus[i] * higgs * higgs )

        lamHSLE  = sympy.expand(lagrangian[0]).coeff('h',1).coeff('SLE_minus',1).coeff('SLE_plus',1)
        lamHSRE  = sympy.expand(lagrangian[0]).coeff('h',1).coeff('SRE_minus',1).coeff('SRE_plus',1)
        lamHSNE  = sympy.expand(lagrangian[0]).coeff('h',1).coeff('SNE_minus',1).coeff('SNE_plus',1)
        lamHHSLE = sympy.expand(lagrangian[0]).coeff('h',2).coeff('SLE_minus',1).coeff('SLE_plus',1)
        lamHHSRE = sympy.expand(lagrangian[0]).coeff('h',2).coeff('SRE_minus',1).coeff('SRE_plus',1)
        lamHHSNE = sympy.expand(lagrangian[0]).coeff('h',2).coeff('SNE_minus',1).coeff('SNE_plus',1)

        lamHSLM  = sympy.expand(lagrangian[1]).coeff('h',1).coeff('SLM_minus',1).coeff('SLM_plus',1)
        lamHSRM  = sympy.expand(lagrangian[1]).coeff('h',1).coeff('SRM_minus',1).coeff('SRM_plus',1)
        lamHSNM  = sympy.expand(lagrangian[1]).coeff('h',1).coeff('SNM_minus',1).coeff('SNM_plus',1)
        lamHHSLM = sympy.expand(lagrangian[1]).coeff('h',2).coeff('SLM_minus',1).coeff('SLM_plus',1)
        lamHHSRM = sympy.expand(lagrangian[1]).coeff('h',2).coeff('SRM_minus',1).coeff('SRM_plus',1)
        lamHHSNM = sympy.expand(lagrangian[1]).coeff('h',2).coeff('SNM_minus',1).coeff('SNM_plus',1)

        lamHSLT  = sympy.expand(lagrangian[2]).coeff('h',1).coeff('diag_SLT_minus',1).coeff('diag_SLT_plus',1)
        lamHSLRT = sympy.expand(lagrangian[2]).coeff('h',1).coeff('diag_SRT_minus',1).coeff('diag_SLT_plus',1)
        lamHSRLT = sympy.expand(lagrangian[2]).coeff('h',1).coeff('diag_SLT_minus',1).coeff('diag_SRT_plus',1)
        lamHSRT  = sympy.expand(lagrangian[2]).coeff('h',1).coeff('diag_SRT_minus',1).coeff('diag_SRT_plus',1)
        lamHSNT  = sympy.expand(lagrangian[2]).coeff('h',1).coeff('SNT_minus',1).coeff('SNT_plus',1)
        lamHHSLT = sympy.expand(lagrangian[2]).coeff('h',2).coeff('diag_SLT_minus',1).coeff('diag_SLT_plus',1)
        lamHHSLRT= sympy.expand(lagrangian[2]).coeff('h',2).coeff('diag_SRT_minus',1).coeff('diag_SLT_plus',1)
        lamHHSRLT= sympy.expand(lagrangian[2]).coeff('h',2).coeff('diag_SLT_minus',1).coeff('diag_SRT_plus',1)
        lamHHSRT = sympy.expand(lagrangian[2]).coeff('h',2).coeff('diag_SRT_minus',1).coeff('diag_SRT_plus',1)
        lamHHSNT = sympy.expand(lagrangian[2]).coeff('h',2).coeff('SNT_minus',1).coeff('SNT_plus',1)

        yL     = y_L
        yR     = y_R
        yLT    = y_L * mass_mat[1][0][0]
        yLRT   = y_L * mass_mat[1][0][1]
        yRLT   = y_R * mass_mat[1][1][0]
        yRT    = y_R * mass_mat[1][1][1]


        list_val = [Mx,MSLE,MSRE,MSNE,MSLM,MSRM,MSNM,MSLT,MSRT,MSNT,
                    lamHSLE,lamHSRE,lamHSNE,lamHHSLE,lamHHSRE,lamHHSNE,
                    lamHSLM,lamHSRM,lamHSNM,lamHHSLM,lamHHSRM,lamHHSNM,
                    lamHSLT,lamHSLRT,lamHSRLT,lamHSRT,lamHSNT,lamHHSLT,lamHHSLRT,lamHHSRLT,lamHHSRT,lamHHSRT,lamHHSNT,
                    yL,yR,yLT,yLRT,yRLT,yRT]

        list_name = ['Mx','MSLE','MSRE','MSNE','MSLM','MSRM','MSNM','MSLT','MSRT','MSNT',
                    'lamHSLE','lamHSRE','lamHSNE','lamHHSLE','lamHHSRE','lamHHSNE',
                    'lamHSLM','lamHSRM','lamHSNM','lamHHSLM','lamHHSRM','lamHHSNM',
                    'lamHSLT','lamHSLRT','lamHSRLT','lamHSRT','lamHSNT','lamHHSLT','lamHHSLRT','lamHHSRLT','lamHHSRT','lamHHSRT','lamHHSNT',
                    'yL','yR','yLT','yLRT','yRLT','yRT']

        return dict(zip(list_name,list_val))
