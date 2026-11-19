#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:04:44 2026

@author: chingchen
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# numpy 1.x 叫 trapz,2.x 改名 trapezoid
_trapz = getattr(np, 'trapezoid', None) or np.trapz

G_GRAV = 6.674e-11
R_GAS  = 8.314462
M_FE, M_S = 55.845, 32.060
M_FES  = M_FE + M_S

# ══════════════════════════════════════════════════════════════════════════
# 端元的選擇 —— 有科學後果,不是風格
# ══════════════════════════════════════════════════════════════════════════
FE_SOURCE    = 'kuwayama'   # 'kuwayama'(Mie-Grueneisen) | 'komabayashi'(Anderson-G)
FES_SOURCE   = 'sakai'      # 'sakai' | 'samuel'
IDEAL_MIXING = False        # True 則關掉 Margules 過剩體積


# ══════════════════════════════════════════════════════════════════════════
# [1] 組成換算
#     1 mol 混合物:n_FeS = chi, n_Fe = 1-chi
#     w_S = chi*M_S/(M_FE + chi*M_S)   ->   chi = M_FE*w_S/(M_S*(1-w_S))
# ══════════════════════════════════════════════════════════════════════════
def wS_to_chi(w_S):
    w_S = float(np.clip(w_S, 0.0, 0.36485))
    return M_FE*w_S/(M_S*(1.0-w_S))

def wS_to_atS(w_S):
    nS, nFe = w_S/M_S, (1.0-w_S)/M_FE
    return 100.0*nS/(nS+nFe)


# ══════════════════════════════════════════════════════════════════════════
# [2] 端元 EoS
#
#   兩種形式共用同一條等溫線（Vinet）,只在熱項分歧:
#
#   (A) Mie-Grueneisen  P(V,T),核心量 gamma（熱壓力）
#       P = P_vinet(x) + gamma(x)/V*[E(x,T)-E(x,T0)]
#       E = 3nR[T + e(x)T^2]  （古典均分 + Sommerfeld 電子）
#       gamma(x) = gamma0*x^b
#
#   (B) Anderson-Grueneisen  V(P,T),核心量 alpha（熱膨脹）
#       alpha(V)/alpha0 = exp{-(delta0/kappa)[1-(V/V0)^kappa]}
#       V 由 dV/dT|_P = alpha(V)*V 從 T0 積到 T
#       必須積 ODE:用 V=V(T0)exp[alpha*dT] 的固定點在 delta0/kappa 大時發散
#       （Komabayashi 的 Fe:5.1/0.56 = 9.1 -> overflow）
#
#   兩者由 alpha = gamma*Cv/(V*K_T) 相連。混合在體積層次,
#   所以兩端元用不同形式沒問題;但同一端元不可混用兩篇參數。
# ══════════════════════════════════════════════════════════════════════════
def _vinet(V, V0, K0, Kp):
    x = V/V0; x13 = x**(1.0/3.0)
    return 3*K0*x**(-2.0/3.0)*(1.0-x13)*np.exp(1.5*(Kp-1.0)*(1.0-x13))

# liquid Fe — Kuwayama et al. 2020 PRL 124,165701, SM Sec.4
#   對 Table SII 驗證 0.05%;絕熱線對 SM Fig.4c 差 +3~+4 K
#   火星核 19-56 GPa 完全在量測範圍（21.5-116 GPa）內
#   Li et al. 2022 批評其「高壓外推」,但明言低壓（實際量測處）與其一致
FE_KUW = dict(form='MG', M=M_FE, rho0=7.03, T0=1811.0, K_T0=82.1, Kp0=5.80,
              gam0=2.02, b=0.63, e0=0.68e-4, g=-1.0, n=1.0)
FE_KUW['V0'] = FE_KUW['M']/FE_KUW['rho0']

# liquid Fe — Komabayashi 2014 JGR Solid Earth, Table 2
#   kappa = 0.56（註腳 c）,不是其他相的 1.4
#   參考態 T0 = 298 K（介穩過冷液體）
#   對 Kuwayama Table 1:rms 3.5%,平均 -3.0%
#   1 bar/1811 K:rho=6.374（實測 7.034）、alpha=3.37e-4（實測 1.316e-4）
#   保留供敏感度測試
FE_KOM = dict(form='AG', M=M_FE, V0=6.88, T0=298.0, K_T0=148.0, Kp0=5.80,
              a0=9.0e-5, d0=5.1, kappa=0.56, Cp=46.0, n=1.0)

# liquid FeS — Sakai & Hirose 2026a JGR Planets
#   25-55 GPa/2320-3260 K 直接實測密度,涵蓋火星核
#   對其自身 6 點 rms 1.70%（Samuel 那組 3.20%）
#   只擬合密度,沒擬合聲速 -> K_S 外推,Vp 不可靠
#   沒給 Cp,沿用 Samuel 的 62.5（唯一跨論文借用,對結果不敏感）
FES_SAK = dict(form='AG', M=M_FES, V0=M_FES/3.625, T0=1650.0, K_T0=23.8,
               Kp0=4.1, a0=11.8e-5, d0=2.7, kappa=1.4, Cp=62.5, n=2.0)

# liquid FeS — Samuel et al. 2023 Nature SM Table 4
#   擬合納入聲速（Kawaguchi 2017, Nishida 2020）,gamma=0.62 自洽
#   delta_T = 0.4（表上印成 "0.4 +- -0.5"）會讓 gamma 在壓縮下上升 4.6 倍
#   （Anderson 經驗法則:高溫 delta_T ~ K'_T;此處 K'=6.9）
FES_SAM = dict(form='AG', M=M_FES, V0=24.4, T0=1650.0, K_T0=12.0,
               Kp0=6.9, a0=11.8e-5, d0=0.4, kappa=1.4, Cp=62.5, n=2.0)

_FE_TABLE  = {'kuwayama': FE_KUW,  'komabayashi': FE_KOM}
_FES_TABLE = {'sakai':    FES_SAK, 'samuel':      FES_SAM}
def FE():  return _FE_TABLE[FE_SOURCE]
def FES(): return _FES_TABLE[FES_SOURCE]


def _P_MG(V, T, em):
    x  = V/em['V0']
    P0 = _vinet(V, em['V0'], em['K_T0'], em['Kp0'])
    e  = em['e0']*x**em['g']
    E  = lambda TT: 3*em['n']*R_GAS*(TT + e*TT*TT)
    return P0 + (em['gam0']*x**em['b'])/(V*1e-6)*(E(T)-E(em['T0']))/1e9

def _alpha_AG(V, em):
    return em['a0']*np.exp(-(em['d0']/em['kappa'])*(1.0-(V/em['V0'])**em['kappa']))

def V_endmember(P, T, em):
    if not (np.isfinite(P) and np.isfinite(T)) or P <= 0:
        return np.nan
    if em['form'] == 'MG':
        f = lambda V: _P_MG(V, T, em) - P
        lo, hi = em['V0']*0.25, em['V0']*3.0
        if f(lo)*f(hi) > 0: return np.nan
        return brentq(f, lo, hi, xtol=1e-12)
    f = lambda V: _vinet(V, em['V0'], em['K_T0'], em['Kp0']) - P
    lo, hi = em['V0']*0.25, em['V0']*4.0
    if f(lo)*f(hi) > 0: return np.nan
    V_T0 = brentq(f, lo, hi, xtol=1e-13)
    sol  = solve_ivp(lambda t, y: [_alpha_AG(y[0], em)*y[0]],
                     [em['T0'], T], [V_T0], rtol=1e-10, atol=1e-12)
    return float(sol.y[0][-1])

def Cv_endmember(V, T, em):
    if em['form'] == 'MG':
        x = V/em['V0']
        return 3*em['n']*R_GAS*(1.0 + 2.0*em['e0']*x**em['g']*T)
    # Cp - Cv = alpha^2 K_T V T,只在參考態換算一次
    # 若每個 (P,T) 重算會得到低於 Dulong-Petit 的 Cv,物理上不可能
    return em['Cp'] - em['a0']**2*(em['K_T0']*1e9)*(em['V0']*1e-6)*em['T0']


# ══════════════════════════════════════════════════════════════════════════
# [3] 混合律  Samuel 2023 SM Eq.12（出處 Irving 2023）;W(P) 出自 Xu 2021
#     V = (1-chi)V_Fe + chi V_FeS + chi(1-chi)[chi W_Fe-FeS + (1-chi)W_FeS-Fe]
#     只有體積是廣延量,可加;密度與模數是強度量,不可平均
# ══════════════════════════════════════════════════════════════════════════
W_FE_FES, W_FES_FE = -9.9, -3.54
B0_MARG,  BP_MARG  = 3.02, 2.6

def W_of_P(w, P):
    return w*np.exp((1.0/BP_MARG)*(1.0-np.sqrt(1.0+2.0*(BP_MARG/B0_MARG)*P)))

def V_mix(P, T, chi):
    vFe  = V_endmember(P, T, FE())
    vFeS = V_endmember(P, T, FES())
    if not (np.isfinite(vFe) and np.isfinite(vFeS)): return np.nan
    V = (1.0-chi)*vFe + chi*vFeS
    if not IDEAL_MIXING:
        V += (1.0-chi)*chi*(chi*W_of_P(W_FE_FES, P) + (1.0-chi)*W_of_P(W_FES_FE, P))
    return V

def Cv_mix(P, T, chi):
    vFe  = V_endmember(P, T, FE())
    vFeS = V_endmember(P, T, FES())
    return (1.0-chi)*Cv_endmember(vFe, T, FE()) + chi*Cv_endmember(vFeS, T, FES())


# ══════════════════════════════════════════════════════════════════════════
# [4][5] 數值微分 + 熱力學鏈
#     K_T = -V(dP/dV)_T ;  alpha = (1/V)(dV/dT)_P
#     gamma = alpha K_T V/Cv ;  K_S = K_T(1+alpha*gamma*T)
#     Vp = sqrt(K_S/rho) ;  dT/dP|_S = gamma T/K_S
# ══════════════════════════════════════════════════════════════════════════
def core_properties(P, T, w_S, dP=0.05, dT=5.0, verbose=False):
    chi   = wS_to_chi(w_S)
    M_mix = (1.0-chi)*M_FE + chi*M_FES
    V   = V_mix(P,    T,    chi)
    Vpp = V_mix(P+dP, T,    chi);  Vpm = V_mix(P-dP, T,    chi)
    Vtp = V_mix(P,    T+dT, chi);  Vtm = V_mix(P,    T-dT, chi)
    if not np.all(np.isfinite([V, Vpp, Vpm, Vtp, Vtm])):
        return dict(rho=np.nan, Vp=np.nan, dTdP=np.nan, gamma=np.nan,
                    K_T=np.nan, K_S=np.nan, alpha=np.nan, cv=np.nan, V=np.nan)
    rho   = M_mix/V
    K_T   = -V*(2*dP)/(Vpp-Vpm)
    alpha = (Vtp-Vtm)/(2*dT)/V
    cv    = Cv_mix(P, T, chi)
    gamma = alpha*(K_T*1e9)*(V*1e-6)/cv
    K_S   = K_T*(1.0+alpha*gamma*T)
    Vp    = np.sqrt(K_S*1e9/(rho*1e3))/1000.0
    if verbose:
        print(f"      chi_FeS={chi:.4f}  V={V:.4f} cm3/mol  rho={rho:.4f} g/cm3")
        print(f"      K_T={K_T:.2f} GPa  alpha={alpha:.4e}/K  Cv={cv:.2f} J/K/mol")
        print(f"      gamma={gamma:.4f}  K_S={K_S:.2f} GPa  Vp={Vp:.4f} km/s"
              f"  dT/dP={gamma*T/K_S:.2f} K/GPa")
    return dict(rho=rho, Vp=Vp, K_T=K_T, K_S=K_S, alpha=alpha, gamma=gamma,
                cv=cv, V=V, dTdP=gamma*T/K_S)


# ══════════════════════════════════════════════════════════════════════════
# [6]-[9] 核剖面自洽積分
#     M(r)=int 4 pi r^2 rho dr ; g=GM/r^2 ; dP/dr=-rho g ; dT/dP=gamma T/K_S
#     中心壓力對組成極敏感（純 FeS ~33,純 Fe ~50 GPa）,不能用固定壓力場
# ══════════════════════════════════════════════════════════════════════════
def build_core_profile(T_core, w_S, R_cmb_km=1650.0, P_cmb=22.0,
                       n=80, n_iter=40, tol=1.0, verbose=False):
    r   = np.linspace(R_cmb_km*1e3, 1.0, n)
    dr  = np.diff(r)
    rho = np.full(n, 6500.0); P = np.full(n, P_cmb); T = np.full(n, T_core)
    for it in range(n_iter):
        ra, rhoa = r[::-1], rho[::-1]
        M = np.zeros(n)
        for i in range(1, n):
            rm  = 0.5*(ra[i]+ra[i-1]); rhm = 0.5*(rhoa[i]+rhoa[i-1])
            M[i] = M[i-1] + 4*np.pi*rm*rm*rhm*(ra[i]-ra[i-1])
        M = M[::-1]
        g = np.zeros(n); g[r > 1.0] = G_GRAV*M[r > 1.0]/r[r > 1.0]**2
        P[0], T[0] = P_cmb, T_core
        for i in range(1, n):
            P[i] = P[i-1] - 0.5*(rho[i]+rho[i-1])*0.5*(g[i]+g[i-1])*dr[i-1]/1e9
            d = core_properties(P[i-1], T[i-1], w_S)['dTdP']
            T[i] = T[i-1] + (d if np.isfinite(d) else 0.0)*(P[i]-P[i-1])
        rho_new = np.array([core_properties(P[i], T[i], w_S)['rho']*1e3
                            for i in range(n)])
        if not np.all(np.isfinite(rho_new)): return None
        dmax = np.max(np.abs(rho_new-rho))
        rho  = 0.5*rho + 0.5*rho_new
        if verbose:
            print(f"      iter {it+1:2d}: max|drho|={dmax:8.2f}  "
                  f"P_c={P[-1]:6.2f} GPa  T_c={T[-1]:6.0f} K")
        if dmax < tol: break
    Vp = np.array([core_properties(P[i], T[i], w_S)['Vp'] for i in range(n)])
    return dict(r=r, P=P, T=T, rho=rho, Vp=Vp, g=g, M=M,
                P_center=P[-1], T_center=T[-1], n_iter=it+1,
                rho_mean=3*_trapz(rho[::-1]*r[::-1]**2, r[::-1])/r[0]**3)


# ══════════════════════════════════════════════════════════════════════════
# 液相線 Sakai & Hirose 2026b:T = T0[(P-P0)/a + 1]^(1/c)
# ══════════════════════════════════════════════════════════════════════════
def T_peritectic(P): return 1473.0*((P-21.0)/3.2 + 1.0)**(1.0/5.5)
def T_eutectic(P):   return 1373.0*((P-21.0)/5.3 + 1.0)**(1.0/5.1)
def dT_liquidus_dP(P, h=0.5):
    return (T_peritectic(P+h)-T_peritectic(P-h))/(2*h)



# ══════════════════════════════════════════════════════════════════════════
#                        以下是逐步執行的示範
#          （沒有 __main__，直接由上往下跑，Spyder 可逐段選取執行）
# ══════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

w_S    = 0.17          # S 的重量分率
T_core = 2200.0        # K，CMB 溫度（在你的 MCMC 裡是自由參數）
R_cmb  = 1650.0        # km
P_cmb  = 22.0          # GPa

# ── 步驟 1:組成換算 ──────────────────────────────────────────────────────
chi = wS_to_chi(w_S)
print("="*76)
print(f"[1] w_S = {w_S:.3f} (wt)  ->  chi_FeS = {chi:.4f}   at% S = {wS_to_atS(w_S):.1f}%")

# ── 步驟 2:端元體積 ──────────────────────────────────────────────────────
V_fe  = V_endmember(22., 2200., FE())
V_fes = V_endmember(22., 2200., FES())
print("\n"+"="*76)
print(f"[2] 端元體積 @22 GPa, 2200 K   (FE={FE_SOURCE}, FES={FES_SOURCE})")
print(f"    V_Fe  = {V_fe :8.4f} cm3/mol   rho = {M_FE /V_fe :.3f} g/cm3")
print(f"    V_FeS = {V_fes:8.4f} cm3/mol   rho = {M_FES/V_fes:.3f} g/cm3")

# 驗收:FeS 端元對 Sakai 2026a Table 1 的六個實測點
OBS = [(24.9,2320,5281),(35.1,3040,5775),(40.0,2850,6155),
       (48.8,2400,6501),(54.9,3260,6588),(53.3,2710,6799)]
print("\n    [驗收] FeS 端元 vs Sakai 實測")
for src in ('sakai','samuel'):
    d = [100*(M_FES/V_endmember(P,T,_FES_TABLE[src])*1000-o)/o for P,T,o in OBS]
    print(f"      {src:>8}: rms {np.sqrt(np.mean(np.array(d)**2)):5.2f}%   "
          f"{' '.join(f'{x:+5.1f}' for x in d)}")

# ── 步驟 3-5:混合 + 微分 + 熱力學鏈 ──────────────────────────────────────
print("\n"+"="*76)
print("[3][4][5] 混合與熱力學鏈 @22 GPa, 2200 K")
V_id = (1-chi)*V_fe + chi*V_fes
V_ex = (1-chi)*chi*(chi*W_of_P(W_FE_FES,22.) + (1-chi)*W_of_P(W_FES_FE,22.))
print(f"    V_ideal = {V_id:.4f}   V_ex = {V_ex:+.4f}  ({100*V_ex/V_id:+.2f}%)"
      f"   V = {V_id+V_ex:.4f} cm3/mol")
print(f"    Margules W_Fe-FeS: 1 bar {W_FE_FES:.2f} -> 22 GPa {W_of_P(W_FE_FES,22.):.3f}"
      f"  ({100*W_of_P(W_FE_FES,22.)/W_FE_FES:.0f}% 剩餘)")
prop = core_properties(22., 2200., w_S, verbose=True)

# ── 步驟 6-9:核剖面自洽積分 ──────────────────────────────────────────────
print("\n"+"="*76)
print(f"[6]-[9] 核剖面自洽積分   T_core={T_core:.0f} K, w_S={w_S:.2f}")
prof = build_core_profile(T_core, w_S, R_cmb_km=R_cmb, P_cmb=P_cmb, verbose=True)

r_km = prof['r']/1e3
print(f"\n    {'r(km)':>8}{'P(GPa)':>9}{'T(K)':>8}{'rho':>8}{'Vp':>7}{'g':>7}")
for i in [0, 20, 40, 60, 79]:
    print(f"    {r_km[i]:8.0f}{prof['P'][i]:9.2f}{prof['T'][i]:8.0f}"
          f"{prof['rho'][i]/1e3:8.3f}{prof['Vp'][i]:7.2f}{prof['g'][i]:7.2f}")

# 驗收:中心壓力（Sakai 的硬對照,兩個極端組成）
print("\n    [驗收] 中心壓力")
for w, ref in [(0.0,"純Fe 應 49-56 GPa"),(w_S,""),(0.3648,"純FeS 應 33-34 GPa")]:
    pp = build_core_profile(T_core, w, R_cmb_km=R_cmb, P_cmb=P_cmb)
    print(f"      w_S={w:.4f}  P_center={pp['P_center']:5.1f} GPa  "
          f"rho_mean={pp['rho_mean']/1e3:.3f} g/cm3   {ref}")

# ── 液相線比較:ICB 在哪裡? ──────────────────────────────────────────────
T_liq = np.array([T_peritectic(P) if P > 27. else np.nan for P in prof['P']])
print("\n"+"="*76)
print("[ICB] 絕熱線 vs 液相線")
print(f"    液相線斜率 {dT_liquidus_dP(34.):.1f} K/GPa @34 GPa,"
      f" {dT_liquidus_dP(40.):.1f} @40 GPa")
print(f"    絕熱線斜率 {core_properties(34.,2500.,w_S)['dTdP']:.1f} K/GPa @34 GPa,"
      f" {core_properties(40.,2600.,w_S)['dTdP']:.1f} @40 GPa")
ok = np.isfinite(T_liq) & (prof['T'] < T_liq)
print(f"    T < T_liquidus 的節點數: {ok.sum()} / {len(ok)}"
      f"   -> {'有內核' if ok.any() else '全液態,無內核'}")

# ── 端元組合的敏感度 ─────────────────────────────────────────────────────
print("\n"+"="*76)
print("★ 端元組合的敏感度:ICB 溫度計是否成立?")
print("  收斂率 = 液相線斜率 - 絕熱斜率;必須 > 0,兩線才會相交")
print(f"\n  {'Fe':>12}{'FeS':>9}{'dTdP@34':>10}{'收斂@34':>10}"
      f"{'dTdP@40':>10}{'收斂@40':>10}")
_fe0, _fes0 = FE_SOURCE, FES_SOURCE
for fe in ('kuwayama','komabayashi'):
    for fs in ('sakai','samuel'):
        FE_SOURCE, FES_SOURCE = fe, fs
        a = core_properties(34., 2500., w_S)['dTdP']
        b = core_properties(40., 2600., w_S)['dTdP']
        print(f"  {fe:>12}{fs:>9}{a:10.1f}{dT_liquidus_dP(34.)-a:+10.1f}"
              f"{b:10.1f}{dT_liquidus_dP(40.)-b:+10.1f}")
FE_SOURCE, FES_SOURCE = _fe0, _fes0


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
86a_Bi2025_model.py —— 從發表參數重建 Bi et al. 2025 的 BS_SKS_GD_IC 速度模型

Bi 沒有提供剖面檔;模型 = 5 個參數 + cosh 內插(GitHub HX-B/Mars_IC)。
OC 參數固定為 Irving 2023 SKS_GD(SM Table 8):
    R_OC = 1780-1810 km, VP_CMB = 4.9-5.0 km/s, VP_OC_ICB = 5.0-6.5 km/s
IC(refined inversion): R_IC = 612+-50 km, dVP_ICB = 32+-8 %
   -> IC 頂部 VP = 7.3-8.3 km/s;IC 梯度 = OC 平均梯度(他們的假設)
密度:Bi *沒有* 密度模型(code 裡 OC 密度沿用 SKS_GD、IC 寫死 6.5),
     唯一密度資訊是 ICB 密度跳躍 drho = 7+-5 %(振幅比)。

把本檔接在 84 之後執行,會直接疊到 84 的 fig 上;
或單獨執行(獨立畫一張 Vp 圖)。
"""
import numpy as np
import matplotlib.pyplot as plt

R_MARS_BI = 3389.5

# ── BS_SKS_GD_IC 中央值與範圍 ────────────────────────────────────────────────
BI = dict(R_OC=1795.0,  R_OC_rng=(1780., 1810.),
          VP_CMB=4.95,  VP_CMB_rng=(4.9, 5.0),
          VP_OC_ICB=5.91,                    # = IC_VP/1.32,IC_VP 取 7.8
          R_IC=612.0,   R_IC_err=50.0,
          JUMP=0.32,    IC_VP_rng=(7.3, 8.3))

def _bi_cosh_curve(p1, p2, n=60):
    """Bi GitHub 的 draw_curve:depth = a*cosh(vp)+b,x 軸是 vp"""
    a = (p2[1]-p1[1])/(np.cosh(p2[0])-np.cosh(p1[0]))
    b = p1[1] - a*np.cosh(p1[0])
    x = np.linspace(p1[0], p2[0], n)
    return x, a*np.cosh(x)+b               # (vp, depth)

def bi_vp_profile(m=BI):
    """回傳 (vp, depth) 連續剖面:OC cosh 段 + ICB 跳躍 + IC 線性段"""
    d_cmb, d_icb = R_MARS_BI-m['R_OC'], R_MARS_BI-m['R_IC']
    # OC:cosh 內插 (depth_CMB, VP_CMB) -> (depth_ICB, VP_OC_ICB)
    vp_oc, dep_oc = _bi_cosh_curve([m['VP_CMB'], d_cmb],
                                   [m['VP_OC_ICB'], d_icb])
    # IC:跳 32%,梯度 = OC 平均梯度 (他們 create_models 的做法)
    grad   = (m['VP_OC_ICB']-m['VP_CMB'])/(m['R_OC']-m['R_IC'])   # km/s per km
    vp_icb = m['VP_OC_ICB']*(1.0+m['JUMP'])
    vp_ctr = vp_icb + grad*m['R_IC']
    dep_ic = np.linspace(d_icb, R_MARS_BI, 20)
    vp_ic  = vp_icb + grad*(dep_ic-d_icb)
    return (np.concatenate([vp_oc, vp_ic]),
            np.concatenate([dep_oc, dep_ic]))

def overlay_bi(ax_vp, ax_rho=None, color='#7B3FA0'):
    """把 Bi 模型疊到既有的 Vp(必要)與 rho(可選)面板"""
    vp, dep = bi_vp_profile()
    d_icb = R_MARS_BI - BI['R_IC']
    ax_vp.plot(vp, dep, lw=2.2, color=color, zorder=6,
               label='Bi 2025 (BS_SKS_GD_IC)')
    # IC 約束方塊:R_IC 612+-50 km x IC VP 7.3-8.3 km/s
    ax_vp.fill_betweenx([d_icb-BI['R_IC_err'], d_icb+BI['R_IC_err']],
                        BI['IC_VP_rng'][0], BI['IC_VP_rng'][1],
                        color=color, alpha=0.18, zorder=3)
    ax_vp.axhline(d_icb, ls='--', lw=1.0, color=color, alpha=0.6)
    ax_vp.text(0.03, d_icb-15, 'Bi ICB 612$\\pm$50 km',
               transform=ax_vp.get_yaxis_transform(),
               fontsize=8, color=color)
    if ax_rho is not None:      # 密度只有 ICB 跳躍,沒有剖面
        ax_rho.axhline(d_icb, ls='--', lw=1.0, color=color, alpha=0.6)
        ax_rho.text(0.03, d_icb+40, r'Bi: $\delta\rho_{ICB}$=7$\pm$5%',
                    transform=ax_rho.get_yaxis_transform(),
                    fontsize=8, color=color)
    # 自動放寬 y 上緣,避免 Bi 的 CMB(1594 km 深)被裁掉
    y_bot, y_top = ax_vp.get_ylim()          # 深度軸反向:y_top 是上緣(較小值)
    d_cmb_bi = R_MARS_BI - BI['R_OC']
    if y_top > d_cmb_bi - 60:
        ax_vp.set_ylim(y_bot, d_cmb_bi - 60)
    ax_vp.legend(fontsize=8.5, loc='lower left', framealpha=0.85)

# ── 用法 A:接在 84 或 85 之後(namespace 裡有現存的 fig/ax 就疊上去)──────
try:
    overlay_bi(ax[3], ax_rho=ax[2])
    fig.canvas.draw_idle(); plt.show()
    print('已疊到現存的四聯圖上(84 或 85)')
# ── 用法 B:單獨執行,自畫一張 ────────────────────────────────────────────
except NameError:
    vp, dep = bi_vp_profile()
    f2, a2 = plt.subplots(figsize=(4.5, 7))
    overlay_bi(a2)
    a2.set_xlabel('V$_P$ (km/s)', fontsize=13)
    a2.set_ylabel('Depth (km)', fontsize=13)
    a2.set_ylim(R_MARS_BI, R_MARS_BI-1810-100)
    a2.grid(alpha=0.3, ls='--'); a2.tick_params(labelsize=11)
    a2.set_title('Bi 2025 BS_SKS_GD_IC (reconstructed)', fontsize=11)
    plt.tight_layout(); plt.show()
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
85_all_EoS_options.py —— 一次畫出全部 EoS 選項:Fe(2) x FeS(2) x 混合(2) = 8 組

用法:接在 82_core_teaching.py 的函式區之後執行。
需要:build_core_profile, T_peritectic, 全域 FE_SOURCE/FES_SOURCE/IDEAL_MIXING

顏色 = FeS 端元(對結果影響最大),線型 = Fe 端元,透明度 = 混合律。
右下角表格列出每組的 rho_mean 與 P_center,方便挑 MCMC baseline。
"""
import numpy as np
import matplotlib.pyplot as plt

R_MARS = 3389.5
w_S    = 0.17
T_core = 2200.0
R_cmb  = 1650.0
P_cmb  = 22.0

RHO_TARGET = (6.5, 6.65)     # Samuel/Khan MSL 情境的核平均密度,畫參考帶

# (fe, fes, ideal, color, ls, alpha)
combos = []
for fes, c in [('sakai', '#C0392B'), ('samuel', '#2C3E50')]:
    for fe, ls in [('kuwayama', '-'), ('komabayashi', '--')]:
        for ideal, al in [(False, 1.0), (True, 0.45)]:
            combos.append((fe, fes, ideal, c, ls, al))

_fe0, _fes0, _mix0 = FE_SOURCE, FES_SOURCE, IDEAL_MIXING

fig, ax = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
rows = []
for fe, fes, ideal, c, ls, al in combos:
    globals()['FE_SOURCE']    = fe
    globals()['FES_SOURCE']   = fes
    globals()['IDEAL_MIXING'] = ideal
    # 若 82 是 import 的模組,改成 ct.FE_SOURCE = fe 等三行
    p = build_core_profile(T_core, w_S, R_cmb_km=R_cmb, P_cmb=P_cmb)
    if p is None:
        print(f'{fe}+{fes}+{"ideal" if ideal else "marg"} 未收斂,跳過')
        continue
    d   = R_MARS - p['r']/1e3
    mix = 'id' if ideal else 'mg'
    lab = f'{fe[:3]}+{fes[:3]}+{mix}'
    ax[0].plot(p['P'],       d, lw=1.8, color=c, ls=ls, alpha=al)
    ax[1].plot(p['T'],       d, lw=1.8, color=c, ls=ls, alpha=al)
    ax[2].plot(p['rho']/1e3, d, lw=1.8, color=c, ls=ls, alpha=al, label=lab)
    ax[3].plot(p['Vp'],      d, lw=1.8, color=c, ls=ls, alpha=al)
    rows.append((lab, p['rho_mean']/1e3, p['P_center'], p['T_center']))

globals()['FE_SOURCE'], globals()['FES_SOURCE'] = _fe0, _fes0
globals()['IDEAL_MIXING'] = _mix0

# 液相線(只依 P,用最後一組的壓力場)
T_liq = np.array([T_peritectic(P) if P > 27. else np.nan for P in p['P']])
ax[1].plot(T_liq, d, ':', lw=2, color='C1', label='liquidus (Fe$_{12}$S$_7$)')
ax[1].legend(fontsize=9, loc='lower left')

# 參考:核平均密度帶 + Irving CMB Vp
ax[2].axvspan(*RHO_TARGET, color='gray', alpha=0.15)
ax[2].text(np.mean(RHO_TARGET), R_MARS-100, 'MSL $\\bar\\rho$',
           fontsize=9, color='gray', ha='center')
ax[3].plot([4.95], [R_MARS - R_cmb], 'o', ms=8, color='k', zorder=6,
           label='Irving 2023 (CMB)')
ax[3].legend(fontsize=9, loc='lower left')

ax[0].set_xlabel('P (GPa)', fontsize=14)
ax[1].set_xlabel('T (K)', fontsize=14)
ax[2].set_xlabel(r'$\rho$ (g/cm$^3$)', fontsize=14)
ax[3].set_xlabel('V$_P$ (km/s)', fontsize=14)
ax[0].set_ylabel('Depth (km)', fontsize=14)
ax[2].legend(fontsize=7.5, loc='lower left', framealpha=0.85)
for a in ax:
    a.tick_params(labelsize=12); a.grid(alpha=0.3, ls='--', lw=0.8)
    a.axhline(R_MARS - R_cmb, ls=':', color='gray', lw=1)
ax[0].set_ylim(R_MARS, R_MARS - R_cmb - 100)

fig.suptitle(f'All EoS options   (w$_S$={w_S:.2f}, T$_{{CMB}}$={T_core:.0f} K, '
             f'R$_{{CMB}}$={R_cmb:.0f} km)\n'
             '  colour = FeS end-member (red sakai / dark samuel),'
             '  linestyle = Fe (solid kuwayama / dashed komabayashi),'
             '  faint = ideal mixing', fontsize=12)
plt.tight_layout()
plt.show()

# 摘要表:挑 baseline 用
print('\n' + '='*64)
print(f"{'combo':<16}{'rho_mean':>10}{'P_center':>10}{'T_center':>10}")
print('-'*64)
for lab, rm, pc, tc in rows:
    flag = ' <-- in MSL band' if RHO_TARGET[0] <= rm <= RHO_TARGET[1] else ''
    print(f'{lab:<16}{rm:>10.3f}{pc:>10.2f}{tc:>10.0f}{flag}')
    
    
