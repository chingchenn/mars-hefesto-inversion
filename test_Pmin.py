#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:55:25 2026

@author: chingchen
"""

# -*- coding: utf-8 -*-
"""P_mid vs P_int 自洽性測試：相圖抄自 59.mcmc_nGibbs_Mars_a25.py"""
import numpy as np
from scipy.optimize import fsolve, brentq
_R_GAS = 8.314
_dv298 = {('Fo','ab'):-3.16,('Fo','bg'):-0.98,('Fa','ab'):-3.20,('Fa','bg'):-1.04,('Fa','ag'):-4.24}
_tp = {('Fo','ab'):(14.2,1600.),('Fo','bg'):(19.0,1473.),('Fa','ag'):(5.25,1273.)}
_dS = {('Fo','ab'):-7.7,('Fo','bg'):-7.3,('Fa','ab'):-10.9,('Fa','bg'):-3.1}
_dH = {('Fo','ab'):29970,('Fo','bg'):9080,('Fa','ab'):9620,('Fa','bg'):-5790}
_dV = {}
for _k in [('Fo','ab'),('Fo','bg')]:
    _P0,_T0=_tp[_k]; _dV[_k]=-(_dH[_k]-_T0*_dS[_k])/(_P0*1000.)
_P0,_T0=_tp[('Fa','ag')]
_DHa=_dH[('Fa','ab')]+_dH[('Fa','bg')]; _DSa=_dS[('Fa','ab')]+_dS[('Fa','bg')]
_DVa=-(_DHa-_T0*_DSa)/(_P0*1000.)
for _k in [('Fa','ab'),('Fa','bg')]:
    _dV[_k]=_DVa*_dv298[_k]/_dv298[('Fa','ag')]
_DH_Fo_ag=_dH[('Fo','ab')]+_dH[('Fo','bg')]; _DS_Fo_ag=_dS[('Fo','ab')]+_dS[('Fo','bg')]
_DV_Fo_ag=_dV[('Fo','ab')]+_dV[('Fo','bg')]
_DH_Fa_ag=_dH[('Fa','ab')]+_dH[('Fa','bg')]; _DS_Fa_ag=_dS[('Fa','ab')]+_dS[('Fa','bg')]
_DV_Fa_ag=_dV[('Fa','ab')]+_dV[('Fa','bg')]
_dG_Fo_ab=lambda P,T:_dH[('Fo','ab')]-T*_dS[('Fo','ab')]+_dV[('Fo','ab')]*1000.*P
_dG_Fa_ab=lambda P,T:_dH[('Fa','ab')]-T*_dS[('Fa','ab')]+_dV[('Fa','ab')]*1000.*P
_dG_Fo_ag=lambda P,T:_DH_Fo_ag-T*_DS_Fo_ag+_DV_Fo_ag*1000.*P
_dG_Fa_ag=lambda P,T:_DH_Fa_ag-T*_DS_Fa_ag+_DV_Fa_ag*1000.*P
_W_GAMMA,_W_BETA,_DH_FO_FUS,_DH_FA_FUS = 6200.,12900.,142000.,89300.
_FA_MELT={0.0:(1490.,'a'),3.0:(1650.,'a'),6.3:(1793.,'a'),12.0:(1990.,'g'),20.0:(2200.,'g')}
def _Tm_Fo(P):
    K0,Kp=125.4,5.33
    def f(x):
        ff=((1/x)**(2/3)-1)/2
        return 3*K0*ff*(1+2*ff)**2.5*(1+1.5*(Kp-4)*ff)-P
    return 2163.0*(1+3.0*(1-brentq(f,0.5,1.0)))
def _Tm_eff(P,Tm,ph):
    d=0.0 if ph=='a' else _dG_Fa_ag(P,Tm); return Tm/(1.0-d/_DH_FA_FUS)
_FA_P=sorted(_FA_MELT)
_Tm_Fa=lambda P: float(np.interp(P,_FA_P,[_Tm_eff(p,*_FA_MELT[p]) for p in _FA_P]))
def _loop(T,P,TmFo,TmFa,W,dGFo,dGFa,guesses):
    def eqs(v):
        Xs,Xl=np.clip(v,1e-9,1-1e-9)
        e1=2*np.log(Xl/Xs)+(-W*(1-Xs)**2)/(_R_GAS*T)-_DH_FO_FUS/_R_GAS*(1/TmFo-1/T)-dGFo(P,T)/(_R_GAS*T)
        e2=2*np.log((1-Xl)/(1-Xs))+(-W*Xs**2)/(_R_GAS*T)-_DH_FA_FUS/_R_GAS*(1/TmFa-1/T)-dGFa(P,T)/(_R_GAS*T)
        return [e1,e2]
    for x0 in guesses:
        s,info,ier,_=fsolve(eqs,x0,full_output=True)
        Xs,Xl=s
        if ier==1 and np.max(np.abs(info['fvec']))<1e-8 and 0<Xl<Xs<1: return float(Xs),float(Xl)
    return np.nan,np.nan
_GG=([0.85,0.65],[0.70,0.45],[0.55,0.32],[0.40,0.22],[0.28,0.14],[0.18,0.08],[0.10,0.04],[0.92,0.78])
_GB=([0.97,0.75],[0.93,0.60],[0.90,0.50],[0.88,0.45],[0.85,0.55],[0.80,0.40])
_g=lambda T,P,a,b:_loop(T,P,a,b,_W_GAMMA,_dG_Fo_ag,_dG_Fa_ag,_GG)
_b=lambda T,P,a,b:_loop(T,P,a,b,_W_BETA,_dG_Fo_ab,_dG_Fa_ab,_GB)
def _three(P,TmFo,TmFa):
    def eqs(v):
        Xb,Xg,Xl,T=v
        return [2*np.log(Xl/Xb)+(-_W_BETA*(1-Xb)**2)/(_R_GAS*T)-_DH_FO_FUS/_R_GAS*(1/TmFo-1/T)-_dG_Fo_ab(P,T)/(_R_GAS*T),
                2*np.log((1-Xl)/(1-Xb))+(-_W_BETA*Xb**2)/(_R_GAS*T)-_DH_FA_FUS/_R_GAS*(1/TmFa-1/T)-_dG_Fa_ab(P,T)/(_R_GAS*T),
                2*np.log(Xl/Xg)+(-_W_GAMMA*(1-Xg)**2)/(_R_GAS*T)-_DH_FO_FUS/_R_GAS*(1/TmFo-1/T)-_dG_Fo_ag(P,T)/(_R_GAS*T),
                2*np.log((1-Xl)/(1-Xg))+(-_W_GAMMA*Xg**2)/(_R_GAS*T)-_DH_FA_FUS/_R_GAS*(1/TmFa-1/T)-_dG_Fa_ag(P,T)/(_R_GAS*T)]
    for T0 in np.linspace(2000.,2900.,12):
        s,info,ier,_=fsolve(eqs,[0.84,0.71,0.44,T0],full_output=True)
        if ier==1 and np.max(np.abs(info['fvec']))<1e-8 and 0<s[2]<s[1]<s[0]<1 and 1500<s[3]<3500:
            return dict(Xb=float(s[0]),Xg=float(s[1]),XL=float(s[2]),T=float(s[3]))
    return None
_PT=dict(P=17.7,Tsol=2200.,XS=0.74,XL=0.50)
def _raw_sol(P,Xref):
    TmFo,TmFa=_Tm_Fo(P),_Tm_Fa(P); Ts=np.linspace(1900.,3200.,400)
    XS=np.array([_g(t,P,TmFo,TmFa)[0] for t in Ts]); ok=np.isfinite(XS); Ts,XS=Ts[ok],XS[ok]
    if len(XS)<2 or not (XS.min()<=Xref<=XS.max()): return np.nan
    o=np.argsort(XS); return float(np.interp(Xref,XS[o],Ts[o]))
_SHIFT=_PT['Tsol']-_raw_sol(_PT['P'],_PT['XS'])
def _scan(loopf,P,TmFo,TmFa,n=200):
    Tp=np.linspace(1600.,3200.,n); raw=Tp-_SHIFT
    XS=np.empty(n);XL=np.empty(n)
    for i,tr in enumerate(raw): XS[i],XL[i]=loopf(tr,P,TmFo,TmFa)
    ok=np.isfinite(XS)&np.isfinite(XL)&(XL<XS); return Tp[ok],XS[ok],XL[ok]
def bml_phase_diagram(T_int,P,Mg):
    TmFo,TmFa=_Tm_Fo(P),_Tm_Fa(P); tp=_three(P,TmFo,TmFa)
    def branches(Tsol,Tliq,loop_at):
        if T_int<=Tsol: return dict(melting=False,XS=Mg,XL=np.nan,f_solid=1.,f_liquid=0.,T_solidus=Tsol,T_liquidus=Tliq)
        if T_int>=Tliq: return dict(melting=True,XS=np.nan,XL=Mg,f_solid=0.,f_liquid=1.,T_solidus=Tsol,T_liquidus=Tliq)
        XS,XL=loop_at(T_int)
        if not(np.isfinite(XS) and np.isfinite(XL)) or abs(XS-XL)<1e-6: return None
        f=float(np.clip((Mg-XS)/(XL-XS),0.,1.))
        return dict(melting=True,XS=XS,XL=XL,f_solid=1-f,f_liquid=f,T_solidus=Tsol,T_liquidus=Tliq)
    def sol_liq(loopf,loop_at,Tso=None):
        Tg,XS,XL=_scan(loopf,P,TmFo,TmFa)
        if len(XS)<3: return None
        o,o2=np.argsort(XS),np.argsort(XL)
        if Tso is None:
            if not(XS.min()<=Mg<=XS.max()): return None
            Tso=float(np.interp(Mg,XS[o],Tg[o]))
        if not(XL.min()<=Mg<=XL.max()): return None
        Tliq=float(np.interp(Mg,XL[o2],Tg[o2]))
        return branches(Tso,max(Tliq,Tso+1.),loop_at)
    ga=lambda T:_g(T-_SHIFT,P,TmFo,TmFa); ba=lambda T:_b(T-_SHIFT,P,TmFo,TmFa)
    if tp is None: return sol_liq(_g,ga)
    if Mg<=tp['Xg']: return sol_liq(_g,ga)
    if Mg>=tp['Xb']: return sol_liq(_b,ba)
    return sol_liq(_b,ba,Tso=tp['T']+_SHIFT)

# ── Mars P(z) 近似（貼合 a31 的 P_int 分布：18.9 [18.0, 20.2] GPa）
CMB, P_CMB, DPDZ = 1743.32, 20.9, 0.0123     # km, GPa, GPa/km
P_of_z = lambda z: P_CMB - DPDZ*(CMB - z)
DTDP_LIQ = 10.0

def solve(D, Mg, T_core, mode, n_iter=15):
    """mode='pmid' = 現行做法; mode='pint' = 相圖也在 P_int 求，共同迭代"""
    P_top, P_bot = P_of_z(CMB-D), P_of_z(CMB)
    P_mid = (P_top+P_bot)/2
    T_int, h_s, P_int = T_core, D, P_mid
    for _ in range(n_iter):
        T_old, h_old = T_int, h_s
        P_use = P_mid if mode=='pmid' else P_int
        pd = bml_phase_diagram(T_int, P_use, Mg)
        if pd is None: return None
        if not pd['melting']:
            return dict(state='solid', h_s=D, h_l=0., f_l=0., XS=Mg, XL=np.nan, P=P_use, T_int=T_core)
        if pd['f_solid'] < 1e-3:
            return dict(state='molten', h_s=0., h_l=D, f_l=1., XS=np.nan, XL=Mg, P=P_use,
                        T_int=T_core-DTDP_LIQ*(P_bot-P_top))
        h_s = max(D*pd['f_solid'], 0.)
        if h_s < 1.0:
            return dict(state='molten', h_s=0., h_l=D, f_l=1., XS=np.nan, XL=Mg, P=P_use, T_int=T_int)
        P_int = P_top + (P_bot-P_top)*h_s/D
        T_int = T_core - DTDP_LIQ*(P_bot-P_int)
        if abs(T_int-T_old) < 5.0 and abs(h_s-h_old) < 0.5: break
    return dict(state='layered', h_s=h_s, h_l=D-h_s, f_l=pd['f_liquid'],
                XS=pd['XS'], XL=pd['XL'], P=P_use, T_int=T_int,
                Tsol=pd['T_solidus'], Tliq=pd['T_liquidus'])

print(f"{'D':>5}{'Mg':>6}{'Tcore':>7} | {'P_mid 做法':>28} | {'P_int 迭代':>28} | {'Δh_s':>8}{'Δf_l':>8}")
print(f"{'':>18} | {'state':>9}{'h_s':>7}{'f_l':>6}{'P':>6} | {'state':>9}{'h_s':>7}{'f_l':>6}{'P':>6} |")
print('-'*104)
rows=[]
for D in [150., 225., 300.]:
    for Mg in [0.35, 0.42, 0.50]:
        for Tc in [2400., 2700., 3000.]:
            a = solve(D,Mg,Tc,'pmid'); b = solve(D,Mg,Tc,'pint')
            if a is None or b is None: continue
            dh = b['h_s']-a['h_s']; df = b['f_l']-a['f_l']
            rows.append((a['state'],b['state'],dh,df))
            print(f"{D:5.0f}{Mg:6.2f}{Tc:7.0f} | {a['state']:>9}{a['h_s']:7.1f}{a['f_l']:6.2f}{a['P']:6.2f} | "
                  f"{b['state']:>9}{b['h_s']:7.1f}{b['f_l']:6.2f}{b['P']:6.2f} | {dh:8.2f}{df:8.3f}")

lay=[r for r in rows if r[0]=='layered' or r[1]=='layered']
if lay:
    dh=np.array([r[2] for r in lay]); df=np.array([r[3] for r in lay])
    print(f"\n分層態 n={len(lay)}:  |Δh_s| median {np.median(np.abs(dh)):.2f} km, max {np.abs(dh).max():.2f} km")
    print(f"               |Δf_l| median {np.median(np.abs(df)):.4f}, max {np.abs(df).max():.4f}")
flip=[r for r in rows if r[0]!=r[1]]
print(f"三態分類改變的案例: {len(flip)} / {len(rows)}")
for r in flip: print('   ', r[0], '->', r[1])