#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:18:19 2026

@author: chingchen
"""

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import fsolve

R = 8.314
T_fix = 1500.0          # fix T

# ΔV(298K, 1bar)，cm³/mol —— Akaogi 1989 Table 3
dv298 = {('Fo','ab'): -3.16, ('Fo','bg'): -0.98,('Fo','ag'): -4.14,
         ('Fa','ab'): -3.20, ('Fa','bg'): -1.04,('Fa','ag'): -4.24,} 

#   Fo a→b: 14.2 GPa @1600 K   Katsura & Ito 2004
#   Fo b→g: 19.0 GPa @1473 K   Katsura & Ito 1989
#   Fa a→g:  5.25 GPa @1273 K  Yagi 1987 eq.(1): P=2.75+2.5e-3·T(°C)，±0.1 GPa
#            （附贈斜率 dP/dT = 2.5 MPa/K，當 Clapeyron 驗證用）
transformation_point = {('Fo','ab'): (14.2, 1600.), ('Fo','bg'): (19.0, 1473.),
                        ('Fa','ag'): (5.25, 1273.)}

dS_Akaogi= {('Fo','ab'): -7.7, ('Fo','bg'): -7.3,('Fo','ag'): -15.0,
             ('Fa','ab'): -10.9, ('Fa','bg'): -3.1,('Fa','ag'): -14.0,}

dH_Akaogi = {('Fo','ab'): 29970, ('Fo','bg'): 9080,('Fo','ag'): 39050,
             ('Fa','ab'): 9620, ('Fa','bg'): -5790,('Fa','ag'): 3830}
dG_Akaogi = {}
dV={}

for k in dH_Akaogi:
    dG_Akaogi[k] =   dH_Akaogi[k]-300*  dS_Akaogi[k]        # ΔG₀  = ΔH - T₀ΔS  J/mol
for k in [('Fo','ab'), ('Fo','bg')]:
    Ptr, Ttr = transformation_point[k]
    dV[k] = -(dH_Akaogi[k] - Ttr*dS_Akaogi[k]) / (Ptr*1000.)

# Fa in a→g β-Fa metastable）
Ptr, Ttr = transformation_point[('Fa','ag')]
DH_ag = dH_Akaogi[('Fa','ab')] + dH_Akaogi[('Fa','bg')]   # Hess
DS_ag = dS_Akaogi[('Fa','ab')] + dS_Akaogi[('Fa','bg')]
DV_ag = -(DH_ag - Ttr*DS_ag) / (Ptr*1000.)
for k in [('Fa','ab'), ('Fa','bg')]:
    dV[k] = DV_ag * dv298[k] / dv298[('Fa','ag')]
    


W_SETS = {
 'akaogi':    {'a': lambda P: 12600. + 260.*P,
               'b': lambda P: 7800.,  'g': lambda P: 7800.},
 'slb2024':   {'a': lambda P: 2500.,               # Table C2: 2.5±21 kJ
               'b': lambda P: 12900.,              # 12.9±25（V12=0 → 無 P 項）
               'g': lambda P: 6200.},              # 6.2±17
 'chanyshev': {'a': lambda P: 4074.,               # 2.037×2（half-f.u.→f.u.）
               'b': lambda P: 12900.,              # SLB
               'g': lambda P: 5394.},              
}
W = W_SETS['slb2024']    


DH_Fo_ab, DS_Fo_ab, DV_Fo_ab = dH_Akaogi[('Fo','ab')], dS_Akaogi[('Fo','ab')], dV[('Fo','ab')]
DH_Fo_bg, DS_Fo_bg, DV_Fo_bg = dH_Akaogi[('Fo','bg')], dS_Akaogi[('Fo','bg')], dV[('Fo','bg')]

DH_Fa_ab, DS_Fa_ab, DV_Fa_ab = dH_Akaogi[('Fa','ab')], dS_Akaogi[('Fa','ab')], dV[('Fa','ab')]
DH_Fa_bg, DS_Fa_bg, DV_Fa_bg = dH_Akaogi[('Fa','bg')], dS_Akaogi[('Fa','bg')], dV[('Fa','bg')]
DH_Fo_ag = DH_Fo_ab + DH_Fo_bg          
DS_Fo_ag = DS_Fo_ab + DS_Fo_bg         
DV_Fo_ag = DV_Fo_ab + DV_Fo_bg
DH_Fa_ag = DH_Fa_ab + DH_Fa_bg         
DS_Fa_ag = DS_Fa_ab + DS_Fa_bg         
DV_Fa_ag = DV_Fa_ab + DV_Fa_bg        



# ---- 標準態自由能差 = DH - T*DS + DV*P ----
# (DV 單位 cm3/mol，乘 1000 變 J/(mol·GPa))
def dG_Fo_ab(P,T): return DH_Fo_ab - T*DS_Fo_ab + DV_Fo_ab*1000.0*P
def dG_Fo_bg(P,T): return DH_Fo_bg - T*DS_Fo_bg + DV_Fo_bg*1000.0*P
def dG_Fa_ab(P,T): return DH_Fa_ab - T*DS_Fa_ab + DV_Fa_ab*1000.0*P
def dG_Fa_bg(P,T): return DH_Fa_bg - T*DS_Fa_bg + DV_Fa_bg*1000.0*P
def dG_Fo_ag(P,T): return dG_Fo_ab(P,T) + dG_Fo_bg(P,T)   # Hess
def dG_Fa_ag(P,T): return dG_Fa_ab(P,T) + dG_Fa_bg(P,T)

# ============================================================
# 4 條方程，每一條直接寫出來
# ============================================================
def equations(v):
    T = T_fix
    Xa, Xb, Xg, P = v
    Wa, Wb,Wg = W['a'](P), W['b'](P), W['g'](P)

    # ---- alpha-beta 平衡：Fo 相等、Fa 相等 ----
    eq1_ab_Fo = ( 2*np.log(Xb/Xa)
                  + Wb*(1-Xb)**2/(R*T) - Wa*(1-Xa)**2/(R*T)
                  + dG_Fo_ab(P,T_fix)/(R*T) )

    eq2_ab_Fa = ( 2*np.log((1-Xb)/(1-Xa))
                  + Wb*Xb**2/(R*T) - Wa*Xa**2/(R*T)
                  + dG_Fa_ab(P,T_fix)/(R*T) )

    # ---- beta-gamma 平衡：Fo 相等、Fa 相等 ----
    eq3_bg_Fo = ( 2*np.log(Xg/Xb)
                  + Wg*(1-Xg)**2/(R*T) - Wb*(1-Xb)**2/(R*T)
                  + dG_Fo_bg(P,T_fix)/(R*T) )

    eq4_bg_Fa = ( 2*np.log((1-Xg)/(1-Xb))
                  + Wg*Xg**2/(R*T) - Wb*Xb**2/(R*T)
                  + dG_Fa_bg(P,T_fix)/(R*T) )

    return [eq1_ab_Fo, eq2_ab_Fa, eq3_bg_Fo, eq4_bg_Fa]

# 初始猜值 (Xa, Xb, Xg, P)
Xa, Xb, Xg, P3 = fsolve(equations, [0.8, 0.8, 0.6, 13.9])




# two phase loop：F=1


fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.set_xlim(0,1); ax1.set_ylim(4,22)
ax1.set_xlabel('Mg#  ($X_{Fo}$)', fontsize=14); ax1.set_ylabel('P (GPa)', fontsize=14)
ax1.set_title(f'fix T = {T_fix:.0f}  K', fontsize=14)
ax1.tick_params(labelsize=14); ax1.grid(alpha=0.3)

# alpha to bata
P_Fo_ab = -(DH_Fo_ab- T_fix*DS_Fo_ab)/DV_Fo_ab/1000
P_Fa_ab = -(DH_Fa_ab- T_fix*DS_Fa_ab)/DV_Fa_ab/1000
Pg_ab = np.linspace(P_Fo_ab,P_Fa_ab, 250)         
ab = []
X = [0.999, 0.995]                         
for P in Pg_ab:
    Wa, Wb = W['a'](P), W['b'](P)
    def eqs(v):                             
        Xa, Xb = v
        
        e1 = 2*np.log(Xb/Xa)         + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T_fix) + dG_Fo_ab(P,T_fix)/(R*T_fix)
        e2 = 2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2     - Wa*Xa**2)/(R*T_fix)     + dG_Fa_ab(P,T_fix)/(R*T_fix)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        ab.append(s); X = list(s)           
    else:
        ab.append([np.nan, np.nan])
ab = np.array(ab) 
m_ab = Pg_ab > P3
ax1.plot(ab[m_ab,0], Pg_ab[m_ab], '-',  color=COLORS[0], lw=2, label=r'$X^\alpha$ (a+b)')
ax1.plot(ab[m_ab,1], Pg_ab[m_ab], '--', color=COLORS[1], lw=2, label=r'$X^\beta$ (a+b)')


# bata to gamma
P_Fo_bg = -(DH_Fo_bg- T_fix*DS_Fo_bg)/DV_Fo_bg/1000
P_Fa_bg = -(DH_Fa_bg- T_fix*DS_Fa_bg)/DV_Fa_bg/1000
Pg_bg = np.linspace(P_Fo_bg,P_Fa_bg, 250)          
bg = []
X = [0.999, 0.995]                          
for P in Pg_bg:
    Wb, Wg = W['b'](P), W['g'](P)
    def eqs(v):                             
        Xb, Xr = v
        e1 = 2*np.log(Xr/Xb)         + (Wg*(1-Xr)**2 - Wb*(1-Xb)**2)/(R*T_fix) + dG_Fo_bg(P,T_fix)/(R*T_fix)
        e2 = 2*np.log((1-Xr)/(1-Xb)) + (Wg*Xr**2     - Wb*Xb**2)/(R*T_fix)     + dG_Fa_bg(P,T_fix)/(R*T_fix)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        bg.append(s); X = list(s)          
    else:
        bg.append([np.nan, np.nan])
bg = np.array(bg) 
m_bg = Pg_bg > P3

ax1.plot(bg[m_bg,0], Pg_bg[m_bg], '-',  color=COLORS[2], lw=2, label=r'$X^\beta$ (b+g)')
ax1.plot(bg[m_bg,1], Pg_bg[m_bg], '--', color=COLORS[3], lw=2, label=r'$X^\gamma$ (b+g)')



# alpha to gamma
P_Fo_ag = -(DH_Fo_ag- T_fix*DS_Fo_ag)/DV_Fo_ag/1000
P_Fa_ag = -(DH_Fa_ag- T_fix*DS_Fa_ag)/DV_Fa_ag/1000
Pg_ag = np.linspace(P_Fo_ag,P_Fa_ag, 250)          
ag = []
X = [0.999, 0.995]                            
for P in Pg_ag:
    Wa, Wg = W['a'](P), W['g'](P)
    def eqs(v):                             
        Xa, Xr = v

        e1 = 2*np.log(Xr/Xa)         + (Wg*(1-Xr)**2 - Wa*(1-Xa)**2)/(R*T_fix) + dG_Fo_ag(P,T_fix)/(R*T_fix)
        e2 = 2*np.log((1-Xr)/(1-Xa)) + (Wg*Xr**2     - Wa*Xa**2)/(R*T_fix)     + dG_Fa_ag(P,T_fix)/(R*T_fix)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        ag.append(s); X = list(s)          
    else:
        ag.append([np.nan, np.nan])
ag = np.array(ag) 
m_ag = Pg_ag < P3
ax1.plot(ag[m_ag,0], Pg_ag[m_ag], '-',  color=COLORS[4], lw=2, label=r'$X^\alpha$ (a+g)')
ax1.plot(ag[m_ag,1], Pg_ag[m_ag], '--', color=COLORS[5], lw=2, label=r'$X^\gamma$ (a+g)')
ax1.legend(fontsize=10)


# P at three phases 
ax1.plot([Xg, Xa], [P3, P3], 'k-', lw=3, zorder=5)
ax1.plot([Xa, Xb, Xg], [P3]*3, 'ko', ms=9, zorder=6)
ax1.text(0.35, 18.5, r'$\gamma$', fontsize=20, fontweight='bold')
ax1.text(0.95, 15.5, r'$\beta$',  fontsize=20, fontweight='bold', ha='center')
ax1.text(0.5, 7.0,  r'$\alpha$', fontsize=20, fontweight='bold')




fig2, ax2 = plt.subplots(figsize=(8, 8))
P_fix = 16.0
ax2.set_xlim(0, 1)
ax2.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14)
ax2.set_ylabel('T (K)', fontsize=14)
ax2.set_title(f'fix P = {P_fix:.0f} GPa : each curve <- one equation pair', fontsize=14)
ax2.tick_params(labelsize=14)
ax2.grid(alpha=0.3)

def equations_T(v):
    Xa, Xb, Xg, T = v
    Wa, Wb, Wg = W['a'](P_fix), W['b'](P_fix), W['g'](P_fix)
    eq1 = 2*np.log(Xb/Xa) + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ab(P_fix,T)/(R*T)
    eq2 = 2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2 - Wa*Xa**2)/(R*T) + dG_Fa_ab(P_fix,T)/(R*T)
    eq3 = 2*np.log(Xg/Xb) + (Wg*(1-Xg)**2 - Wb*(1-Xb)**2)/(R*T) + dG_Fo_bg(P_fix,T)/(R*T)
    eq4 = 2*np.log((1-Xg)/(1-Xb)) + (Wg*Xg**2 - Wb*Xb**2)/(R*T) + dG_Fa_bg(P_fix,T)/(R*T)
    return [eq1, eq2, eq3, eq4]

Xa3, Xb3, Xg3, T3 = fsolve(equations_T, [0.7, 0.55, 0.35, 2400.])


# alpha to bata
T_Fo_ab = (DH_Fo_ab + 1000*DV_Fo_ab * P_fix) / DS_Fo_ab
T_Fa_ab = (DH_Fa_ab+1000*DV_Fa_ab* P_fix)/DS_Fa_ab
print(T_Fo_ab,T_Fa_ab)
Tg_ab = np.linspace(T_Fo_ab,T_Fa_ab, 250)         
ab = []
X = [0.999, 0.995]                         
for T in Tg_ab:
    Wa, Wb = W['a'](P_fix), W['b'](P_fix)
    def eqs(v):                            
        Xa, Xb = v
        e1 = 2*np.log(Xb/Xa)         + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ab(P_fix,T)/(R*T)
        e2 = 2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2     - Wa*Xa**2)/(R*T)     + dG_Fa_ab(P_fix,T)/(R*T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        ab.append(s); X = list(s)           
    else:
        ab.append([np.nan, np.nan])
ab = np.array(ab) 

ax2.plot(ab[:,0], Tg_ab, '-',  color=COLORS[0], lw=2, label=r'$X^\alpha$ (a+b)')
ax2.plot(ab[:,1], Tg_ab, '--', color=COLORS[1], lw=2, label=r'$X^\beta$ (a+b)')



# bata to gamma
T_Fo_bg = (DH_Fo_bg+DV_Fo_bg*1000*P_fix)/DS_Fo_bg
T_Fa_bg = (DH_Fa_bg+DV_Fa_bg*1000*P_fix)/DS_Fa_bg
print(T_Fo_bg,T_Fa_bg)
Tg_bg = np.linspace(T_Fo_bg,T_Fa_bg, 250)          
bg = []
X = [0.999, 0.995]                          
for T in Tg_bg:
    Wb, Wg = W['b'](P_fix), W['g'](P_fix)
    def eqs(v):                             
        Xb, Xr = v

        e1 = 2*np.log(Xr/Xb)         + (Wg*(1-Xr)**2 - Wb*(1-Xb)**2)/(R*T) + dG_Fo_bg(P_fix,T)/(R*T)
        e2 = 2*np.log((1-Xr)/(1-Xb)) + (Wg*Xr**2     - Wb*Xb**2)/(R*T)     + dG_Fa_bg(P_fix,T)/(R*T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        bg.append(s); X = list(s)          
    else:
        bg.append([np.nan, np.nan])
bg = np.array(bg) 


ax2.plot(bg[:,0], Tg_bg, '-',  color=COLORS[2], lw=2, label=r'$X^\beta$ (b+g)')
ax2.plot(bg[:,1], Tg_bg, '--', color=COLORS[3], lw=2, label=r'$X^\gamma$ (b+g)')



# alpha to gamma
T_Fo_ag = (DH_Fo_ag+DV_Fo_ag*1000*P_fix)/DS_Fo_ag
T_Fa_ag = (DH_Fa_ag+DV_Fa_ag*1000*P_fix)/DS_Fa_ag
print(T_Fo_ag,T_Fa_ag)
Tg_ag = np.linspace(T_Fo_ag,T_Fa_ag, 250)          
ag = []
X = [0.999, 0.995]                            
for T in Tg_ag:
    Wa, Wg = W['a'](P_fix), W['g'](P_fix)
    def eqs(v):                             
        Xa, Xr = v

        e1 = 2*np.log(Xr/Xa)         + (Wg*(1-Xr)**2 - Wa*(1-Xa)**2)/(R*T) + dG_Fo_ag(P_fix,T)/(R*T)
        e2 = 2*np.log((1-Xr)/(1-Xa)) + (Wg*Xr**2     - Wa*Xa**2)/(R*T)     + dG_Fa_ag(P_fix,T)/(R*T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        ag.append(s); X = list(s)          
    else:
        ag.append([np.nan, np.nan])
ag = np.array(ag) 

ax2.plot(ag[:,0], Tg_ag, '-',  color=COLORS[4], lw=2, label=r'$X^\alpha$ (a+g)')
ax2.plot(ag[:,1], Tg_ag, '--', color=COLORS[5], lw=2, label=r'$X^\gamma$ (a+g)')

### melting loop 

DH_FO_fus = 142000.0     # Richet et al. 1993, 142±14 kJ/mol @ 2174 K
DH_FA_fus = 89300.0      # Richet et al. 1993, 89.3±11 kJ/mol @ 1490 K


def Tm_Fo_func(P):
    K0, Kp = 125.4, 5.33
    def P_of_x(x):                    
        f = ((1/x)**(2/3) - 1)/2
        return 3*K0*f*(1+2*f)**2.5 * (1 + 1.5*(Kp-4)*f) - P
    x = brentq(P_of_x, 0.5, 1.0)       
    return 2163.0*(1 + 3.0*(1-x))       

Tm_Fo_16 = Tm_Fo_func(P_fix)



FA_MELT = {0.0:(1490.,'a'), 3.0:(1650.,'a'), 6.3:(1793.,'a'),
           12.0:(1990.,'g'), 20.0:(2200.,'g')}   

def Tm_eff(P, Tm, ph):
    d = 0.0 if ph=='a' else dG_Fa_ag(P, Tm)     # 解析換算式 (5.3)
    return Tm/(1.0 - d/DH_FA_fus)

Ps  = sorted(FA_MELT)
Tms = [Tm_eff(p, *FA_MELT[p]) for p in Ps]
Tm_Fa_16 = np.interp(P_fix, Ps, Tms)             # P_fix=16 落在 12~20 之間，直接內插


print(Tm_Fo_16,Tm_Fa_16)
W_MELT = 8000.0

Tg_aL = np.linspace(Tm_Fo_16*0.999, Tm_Fa_16*1.001, 300)
aL = []
X = [0.999, 0.995]
for T in Tg_aL:
    def eqs(v):
        Xs, Xl = v
        e1 = 2*np.log(Xl/Xs)         - W_MELT*(1-Xs)**2/(R*T) \
             - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T)
        e2 = 2*np.log((1-Xl)/(1-Xs)) - W_MELT*Xs**2/(R*T) \
             - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        aL.append(s); X = list(s)
    else:
        aL.append([np.nan, np.nan])
aL = np.array(aL)
ax2.plot(aL[:,0], Tg_aL, '-',  color=COLORS[6], lw=2, label=r'$X^\alpha$ (a+L)')
ax2.plot(aL[:,1], Tg_aL, '--',  color=COLORS[7], lw=2, label=r'$X^L$ (a+L)')


T_close_bL = brentq(lambda T: DH_FO_fus*(1 - T/Tm_Fo_16) - dG_Fo_ab(P_fix, T), 1500., 2769.)
T_close_gL = brentq(lambda T: DH_FO_fus*(1 - T/Tm_Fo_16) - dG_Fo_ag(P_fix, T), 1500., 2769.)
print(T_close_bL, T_close_gL)      

T_close_bL_Fa = brentq(lambda T: DH_FA_fus*(1 - T/Tm_Fa_16) - dG_Fa_ab(P_fix, T), 1200., 3000.)
T_close_gL_Fa = brentq(lambda T: DH_FA_fus*(1 - T/Tm_Fa_16) - dG_Fa_ag(P_fix, T), 1200., 3000.)




Tg_bL = np.linspace(T_close_bL*0.999, T_close_bL_Fa*1.001, 300)
bL = []
X = [0.999, 0.995]
for T in Tg_bL:
    def eqs(v):
        Xs, Xl = v
        e1 = 2*np.log(Xl/Xs)         - W_MELT*(1-Xs)**2/(R*T) \
             - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix,T)/(R*T)
        e2 = 2*np.log((1-Xl)/(1-Xs)) - W_MELT*Xs**2/(R*T) \
             - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T)- dG_Fa_ab(P_fix,T)/(R*T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        bL.append(s); X = list(s)
    else:
        bL.append([np.nan, np.nan])
bL = np.array(bL)
ax2.plot(bL[:,0], Tg_bL, '-',  color=COLORS[8], lw=2, label=r'$X^\beta$ (b+L)')
ax2.plot(bL[:,1], Tg_bL, '--',  color=COLORS[9], lw=2, label=r'$X^L$ (b+L)')
# 

Tg_gL = np.linspace(T_close_gL*0.999, T_close_gL_Fa*1.001, 300)
gL = []
X = [0.999, 0.995]
for T in Tg_gL:
    def eqs(v):
        Xs, Xl = v
        e1 = 2*np.log(Xl/Xs)         - W_MELT*(1-Xs)**2/(R*T) \
             - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ag(P_fix,T)/(R*T)
        e2 = 2*np.log((1-Xl)/(1-Xs)) - W_MELT*Xs**2/(R*T) \
             - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T)- dG_Fa_ag(P_fix,T)/(R*T)
        return [e1, e2]
    s, info, ok, _ = fsolve(eqs, X, full_output=True)
    if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
        gL.append(s); X = list(s)
    else:
        gL.append([np.nan, np.nan])
gL = np.array(gL)
ax2.plot(gL[:,0], Tg_gL, '-',  color=COLORS[10], lw=2, label=r'$X^\gamma$ (g+L)')
ax2.plot(gL[:,1], Tg_gL, '--',  color=COLORS[11], lw=2, label=r'$X^L$ (g+L)')


ax2.set_ylim(1000,3000)
ax2.legend(fontsize=10, loc='lower left')



fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.set_xlim(0, 1)
ax3.set_xlabel(r'Mg#  ($X_{Fo}$)', fontsize=14)
ax3.set_ylabel('T (K)', fontsize=14)
ax3.set_title(f'fix P = {P_fix:.0f} GPa : stable phase diagram', fontsize=14)
ax3.tick_params(labelsize=14)
ax3.grid(alpha=0.3)

# ---- 兩個熔融 4×4（各只解一次，帶安檢）----
def eqs4_abL(v):
    Xa_, Xb_, Xl_, T = v
    return [2*np.log(Xl_/Xa_) - W_MELT*(1-Xa_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T),
            2*np.log((1-Xl_)/(1-Xa_)) - W_MELT*Xa_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T),
            2*np.log(Xl_/Xb_) - W_MELT*(1-Xb_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix,T)/(R*T),
            2*np.log((1-Xl_)/(1-Xb_)) - W_MELT*Xb_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ab(P_fix,T)/(R*T)]

def eqs4_bgL(v):
    Xb_, Xg_, Xl_, T = v
    return [2*np.log(Xl_/Xb_) - W_MELT*(1-Xb_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ab(P_fix,T)/(R*T),
            2*np.log((1-Xl_)/(1-Xb_)) - W_MELT*Xb_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ab(P_fix,T)/(R*T),
            2*np.log(Xl_/Xg_) - W_MELT*(1-Xg_)**2/(R*T) - DH_FO_fus/R*(1/Tm_Fo_16 - 1/T) - dG_Fo_ag(P_fix,T)/(R*T),
            2*np.log((1-Xl_)/(1-Xg_)) - W_MELT*Xg_**2/(R*T) - DH_FA_fus/R*(1/Tm_Fa_16 - 1/T) - dG_Fa_ag(P_fix,T)/(R*T)]

sol, info, ok, _ = fsolve(eqs4_abL, [0.95, 0.9, 0.7, 2500.], full_output=True)
Xa_abL, Xb_abL, Xl_abL, T_abL = sol
has_abL = (ok==1 and max(abs(info['fvec']))<1e-8
           and 0 < Xl_abL < Xb_abL < Xa_abL < 1)      # 殘差 + 物理排序

sol, info, ok, _ = fsolve(eqs4_bgL, [0.85, 0.75, 0.45, 2300.], full_output=True)
Xb_bgL, Xg_bgL, Xl_bgL, T_bgL = sol
has_bgL = (ok==1 and max(abs(info['fvec']))<1e-8
           and 0 < Xl_bgL < Xg_bgL < Xb_bgL < 1)

print('has_abL =', has_abL, ' has_bgL =', has_bgL)

# ---- mask 與畫圖：依三相線存不存在分兩套 ----
if has_abL and has_bgL:
    # ===== 高壓拓樸（如 16 GPa）：三層結構 =====
    m_aL = Tg_aL > T_abL
    m_bL = (Tg_bL > T_bgL) & (Tg_bL < T_abL)
    m_gL = Tg_gL < T_bgL
    m_ab = Tg_ab < T_abL
    m_bg = Tg_bg < T_bgL

    ax3.plot(aL[m_aL,0], Tg_aL[m_aL], '-',  color=COLORS[6], lw=2, label=r'$X^\alpha$ (a+L)')
    ax3.plot(aL[m_aL,1], Tg_aL[m_aL], '--', color=COLORS[7], lw=2, label=r'$X^L$ (a+L)')
    ax3.plot(bL[m_bL,0], Tg_bL[m_bL], '-',  color=COLORS[8], lw=2, label=r'$X^\beta$ (b+L)')
    ax3.plot(bL[m_bL,1], Tg_bL[m_bL], '--', color=COLORS[9], lw=2, label=r'$X^L$ (b+L)')
    ax3.plot(gL[m_gL,0], Tg_gL[m_gL], '-',  color=COLORS[10], lw=2, label=r'$X^\gamma$ (g+L)')
    ax3.plot(gL[m_gL,1], Tg_gL[m_gL], '--', color=COLORS[11], lw=2, label=r'$X^L$ (g+L)')
    ax3.plot(ab[m_ab,0], Tg_ab[m_ab], '-',  color=COLORS[0], lw=2, label=r'$X^\alpha$ (a+b)')
    ax3.plot(ab[m_ab,1], Tg_ab[m_ab], '--', color=COLORS[1], lw=2, label=r'$X^\beta$ (a+b)')
    ax3.plot(bg[m_bg,0], Tg_bg[m_bg], '-',  color=COLORS[2], lw=2, label=r'$X^\beta$ (b+g)')
    ax3.plot(bg[m_bg,1], Tg_bg[m_bg], '--', color=COLORS[3], lw=2, label=r'$X^\gamma$ (b+g)')

    # 兩條三相 tie-line
    ax3.plot([Xl_abL, Xa_abL], [T_abL]*2, 'k-', lw=3, zorder=5)
    ax3.plot([Xa_abL, Xb_abL, Xl_abL], [T_abL]*3, 'ko', ms=8, zorder=6)
    ax3.plot([Xl_bgL, Xb_bgL], [T_bgL]*2, 'k-', lw=3, zorder=5)
    ax3.plot([Xb_bgL, Xg_bgL, Xl_bgL], [T_bgL]*3, 'ko', ms=8, zorder=6)

    # metastable 固固 T3 參考線（可留可刪）
    ax3.plot([Xg3, Xa3], [T3]*2, ':', color='grey', lw=1.5, zorder=4)

else:
    # ===== 低壓拓樸（如 1 atm）：只有 α+L，其餘全 metastable =====
    m_aL = np.ones_like(Tg_aL, dtype=bool)
    ax3.plot(aL[m_aL,0], Tg_aL[m_aL], '-',  color=COLORS[6], lw=2, label=r'$X^\alpha$ (a+L)')
    ax3.plot(aL[m_aL,1], Tg_aL[m_aL], '--', color=COLORS[7], lw=2, label=r'$X^L$ (a+L)')
    # b+L, g+L, a+b, b+g, tie-lines：一律不畫

ax3.set_ylim(1000, 3000)
ax3.legend(fontsize=10, loc='lower left')