#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:17:49 2026

@author: chingchen
"""

import numpy as np 
import matplotlib.pyplot as plt


COLORS = [
    '#CD5C5C',
    '#35838D',
    '#849DAB',
    '#414F67',
    '#97795D',
    '#7B9E87',
    '#9B6B8A',
    '#4E6E8E',
    '#C47F3E',
    '#5C7A5C',
    '#8B6F6F',
    '#4A7C7C',
    '#7A6B9B',
    '#6B8E7A',
]


# from Kuwayama 2020 get density 7.03 at 1811K 
rho_0_Fe = 7.03
molar_Fe = 55.845
v_0_Fe = molar_Fe/ rho_0_Fe # cm³/mol

# FeS system
v_0_FeS = 24.4 # Samuel 2023 SM table 4 @ 1650 K
T_0_FeS = 1650
alpha_FeS = 11.8e-5

def V_FeS(T):
    return v_0_FeS * np.exp(alpha_FeS*(T-T_0_FeS))

T_core = np.linspace(1800,3600,200)
x = np.linspace(0,1,100)
v_ideal = x * v_0_Fe + (1-x) * v_0_Fe

W_Fe_FeS = -9.9     # Samuel Table 4,單位 cm³/mol
W_FeS_Fe = -3.54
v_ex = x*(1-x) * ( (1-x)*W_Fe_FeS + x*W_FeS_Fe )

v_real = v_ideal + v_ex
plt.plot(x,v_real,color=COLORS[4])
plt.plot(x,v_ideal,color=COLORS[2])