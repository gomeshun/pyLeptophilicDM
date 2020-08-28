#!/usr/bin/python
import numpy as np
import re
import par_to_phys as ptp
import g_minus_2 as g2

#テストとしてパラメータカードを読み込んで計算させる。
#m_phi_Lとm_phi_Rはmassの2乗の次元

file = open("Leptophilic_par.dat", "r")
for line in file:
    if not re.match('#',line):
        lines = line.rstrip('\n').split()
        par_name = lines[0]
        par_value= float(lines[1])

        if par_name == "m_phi_L":
            m_phi_L = par_value

        elif par_name == "m_phi_R":
            m_phi_R = par_value

        elif par_name == "m_chi":
            m_chi = par_value

        elif par_name == "A":
            A = par_value

        elif par_name == "y_L":
            y_L = par_value

        elif par_name == "y_R":
            y_R = par_value

        elif par_name == "laphiLH1":
            laphiLH1 = par_value

        elif par_name == "laphiLH2":
            laphiLH2 = par_value

        elif par_name == "laphiRH":
            laphiRH = par_value

file.close()

param = {'m_phi_L':m_phi_L,'m_phi_R':m_phi_R,'m_chi':m_chi,'A':A,'y_L':y_L,'y_R':y_R,'laphiLH1':laphiLH1,'laphiLH2':laphiLH2,'laphiRH':laphiRH}

output = ptp.par_to_phys(param)

print(output)

print(g2.log_likelihood_g_minus_2(output['Mx'], output['MSLM'],output['MSRM'], A, y_L, y_R))
