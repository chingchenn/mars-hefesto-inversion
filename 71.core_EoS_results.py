#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:48:13 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:42:09 2026

@author: chingchen

82_core_teaching.py  ——  火星核:從端元 EoS 到 P(r), T(r), rho(r), Vp(r)

   [1] 組成換算   w_S -> chi_FeS
   [2] 端元體積   V_Fe(P,T), V_FeS(P,T)      <- 可切換不同文獻
   [3] 混合律     V = V_ideal + V_ex
   [4] 數值微分   K_T, alpha
   [5] 熱力學鏈   gamma -> K_S -> Vp, dT/dP
   [6]-[9] 核剖面 自洽積分 P(r), T(r), rho(r)

最後一節測試「ICB 溫度計」在不同端元組合下是否成立。
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from mars_config import DATA_BASE, KHAN_MODEL_DIR

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





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
84_core_profiles_vs_KhanSamuel.py —— EoS 核剖面 vs Samuel 2023 vs Khan 2023

用法:接在 82_core_teaching.py 的函式區之後執行。
需要:build_core_profile, core_properties, T_peritectic,
     FE_SOURCE / FES_SOURCE

Samuel 檔案格式:(value, radius_km) 兩欄,vp 單位 m/s、rho 單位 kg/m3
Khan  檔案格式:0=Depth 1=Vp 2=Vs 3=Rho 4=T 5=g 6=P
"""
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMUEL_BASE    = (f'{DATA_BASE}/Mars_Samuel_2023/'
                  'Nature_Samuel/METADATA_BML/DATA_FIG1')

R_MARS   = 3389.5
DEPTH_LO = 1200.0            # 圖的上緣,留一點 mantle/BML 看 CMB 跳躍
Z_CORE   = np.linspace(DEPTH_LO, R_MARS, 500)   # Khan ensemble 統計用格點

# 我們的 EoS 模型參數
w_S    = 0.1
T_core = 2200.0
R_cmb  = 1650.0
P_cmb  = 22.0

C_EOS, C_SAM, C_KH = '#C0392B', '#708090', '#4198B9'

# ── LOAD SAMUEL 2023(value, radius)→ depth ─────────────────────────────────
def r2d(r): return R_MARS - r

print("Loading Samuel 2023...")
sam_vp,  dep_vp  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vp_profile.dat').T
sam_rho, dep_rho = np.loadtxt(f'{SAMUEL_BASE}/PANEL_J/rho_profile.dat').T
sam_T,   dep_T   = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tprofile.dat').T
dep_vp, dep_rho, dep_T = r2d(dep_vp), r2d(dep_rho), r2d(dep_T)
sam_vp, sam_rho = sam_vp/1000.0, sam_rho/1000.0     # → km/s, g/cm3

# ── LOAD KHAN 2023 ENSEMBLE(全深度,不截 1800)─────────────────────────────
print("Loading Khan models...")
COL_DEPTH, COL_VP, COL_VS, COL_RHO, COL_T, COL_G, COL_P = range(7)
khan = []          # list of dict(depth, Vp, rho, T, P)
for fpath in sorted(glob.glob(f'{KHAN_MODEL_DIR}/Model_*.txt')):
    try:
        d = np.loadtxt(fpath, comments='#')
        if d[:, COL_DEPTH].max() < R_MARS*0.9:      # 沒到核中心的跳過
            continue
        o = np.argsort(d[:, COL_DEPTH])             # 保險:按深度排序
        P_k   = d[o, COL_P]
        rho_k = d[o, COL_RHO]
        Vp_k  = d[o, COL_VP]
        if np.nanmax(P_k)   > 1e4:  P_k   = P_k/1e9     # Pa   → GPa
        if np.nanmax(rho_k) > 100.: rho_k = rho_k/1e3   # kg/m3→ g/cm3
        if np.nanmax(Vp_k)  > 100.: Vp_k  = Vp_k/1e3    # m/s  → km/s
        khan.append(dict(depth=d[o, COL_DEPTH], Vp=Vp_k,
                         rho=rho_k, T=d[o, COL_T], P=P_k))
    except Exception:
        continue
print(f"  {len(khan)} Khan models (full depth)")

def khan_stats(key):
    """在 Z_CORE 格點上算 median 與 16-84%;內插只在各模型深度範圍內有效"""
    mat = np.full((len(khan), len(Z_CORE)), np.nan)
    for i, m in enumerate(khan):
        mat[i] = np.interp(Z_CORE, m['depth'], m[key], left=np.nan, right=np.nan)
    return (np.nanmedian(mat, 0), np.nanpercentile(mat, 16, 0),
            np.nanpercentile(mat, 84, 0))

kh = {k: khan_stats(k) for k in ('Vp', 'rho', 'T', 'P')} if khan else None

# ── 我們的 EoS 核剖面 ─────────────────────────────────────────────────────────
prof  = build_core_profile(T_core, w_S, R_cmb_km=R_cmb, P_cmb=P_cmb, verbose=True)
depth = R_MARS - prof['r']/1e3
T_liq = np.array([T_peritectic(P) if P > 27. else np.nan for P in prof['P']])

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 4, figsize=(15, 8), sharey=True)

def draw_khan(a, key):
    if kh is None: return
    med, p16, p84 = kh[key]
    for m in khan:                                   # 個別模型細線
        a.plot(m[key], m['depth'], color=C_KH, alpha=0.04, lw=0.5,
               rasterized=True)
    a.fill_betweenx(Z_CORE, p16, p84, color=C_KH, alpha=0.20,
                    label='Khan 16–84%')
    a.plot(med, Z_CORE, color=C_KH, lw=2.0, label='Khan median')

# Panel 1: P —— Samuel 沒存 P profile,只畫 Khan + EoS
draw_khan(ax[0], 'P')
ax[0].plot(prof['P'], depth, color=C_EOS, lw=2.5, zorder=5, label='EoS core')
ax[0].set_xlabel('P (GPa)', fontsize=14)
ax[0].set_ylabel('Depth (km)', fontsize=14)

# Panel 2: T
draw_khan(ax[1], 'T')
ax[1].plot(sam_T, dep_T, color=C_SAM, lw=2.0, label='Samuel 2023')
ax[1].plot(prof['T'], depth, color=C_EOS, lw=2.5, zorder=5, label='EoS adiabat')
ax[1].plot(T_liq, depth, '--', lw=1.8, color='C3',
           label='liquidus (Fe$_{12}$S$_7$)')
ax[1].set_xlabel('T (K)', fontsize=14)

# Panel 3: rho
draw_khan(ax[2], 'rho')
ax[2].plot(sam_rho, dep_rho, color=C_SAM, lw=2.0, label='Samuel 2023')
ax[2].plot(prof['rho']/1e3, depth, color=C_EOS, lw=2.5, zorder=5,
           label='EoS core')
ax[2].set_xlabel(r'$\rho$ (g/cm$^3$)', fontsize=14)

# Panel 4: Vp
draw_khan(ax[3], 'Vp')
ax[3].plot(sam_vp, dep_vp, color=C_SAM, lw=2.0, label='Samuel 2023')
ax[3].plot(prof['Vp'], depth, color=C_EOS, lw=2.5, zorder=5, label='EoS core')
ax[3].plot([4.95], [R_MARS - R_cmb], 'o', ms=8, color='k', zorder=6,
           label='Irving 2023 (CMB)')
ax[3].set_xlabel('V$_P$ (km/s)', fontsize=14)

for a in ax:
    a.tick_params(labelsize=12)
    a.grid(alpha=0.3, ls='--', lw=0.8)
    a.axhline(R_MARS - R_cmb, ls=':', color=C_EOS, lw=1.0, alpha=0.6)
    a.legend(fontsize=8.5, loc='lower left', framealpha=0.85)
ax[0].set_ylim(R_MARS, DEPTH_LO)
ax[0].text(0.03, R_MARS - R_cmb - 15, 'EoS CMB',
           transform=ax[0].get_yaxis_transform(),
           fontsize=9, color=C_EOS, alpha=0.8)

fig.suptitle(f'EoS core (w$_S$={w_S:.2f}, T$_{{CMB}}$={T_core:.0f} K, '
             f'R$_{{CMB}}$={R_cmb:.0f} km, {FE_SOURCE}+{FES_SOURCE})'
             f'   vs   Samuel 2023  vs  Khan 2023 ensemble', fontsize=13)
plt.tight_layout()
plt.show()




# ══════════════════════════════════════════════════════════════════════════
# 圖 2:w_S(顏色)x T_core(alpha)敏感度
# ══════════════════════════════════════════════════════════════════════════
from matplotlib.lines import Line2D

WS_LIST = [0.10, 0.15, 0.20]
TC_LIST = [1800, 2200, 2400, 2600, 2800, 3000, 3200]
COLORS  = {0.10: 'C0', 0.15: 'C1', 0.20: 'C2'}
ALPHAS  = np.linspace(0.25, 1.0, len(TC_LIST))   # 冷→淡,熱→深

fig, ax = plt.subplots(1, 4, figsize=(14, 7), sharey=True)

last_p = None
for w in WS_LIST:
    c = COLORS[w]
    for a_val, tc in zip(ALPHAS, TC_LIST):
        p = build_core_profile(tc, w, R_cmb_km=R_cmb, P_cmb=P_cmb)
        if p is None:
            print(f'w_S={w:.2f} T_core={tc} 未收斂,跳過'); continue
        last_p = p
        d = R_MARS - p['r']/1e3
        ax[0].plot(p['P'],       d, lw=1.5, color=c, alpha=a_val)
        ax[1].plot(p['T'],       d, lw=1.5, color=c, alpha=a_val)
        ax[2].plot(p['rho']/1e3, d, lw=1.5, color=c, alpha=a_val)
        ax[3].plot(p['Vp'],      d, lw=1.5, color=c, alpha=a_val)

# 液相線:只跟 P 有關,用最後一組壓力場近似畫一次
T_liq = np.array([T_peritectic(P) if P > 27. else np.nan for P in last_p['P']])
ax[1].plot(T_liq, R_MARS - last_p['r']/1e3, '--', lw=2, color='C3')
ax[3].plot([4.95], [R_MARS - R_cmb], 'o', ms=8, color='k')

# 圖例拆兩組:顏色 = w_S(放 panel 3),alpha = T_core(放 panel 2)
h_ws = [Line2D([], [], color=COLORS[w], lw=2, label=f'w$_S$={w:.2f}')
        for w in WS_LIST]
h_tc = [Line2D([], [], color='gray', lw=2, alpha=a_val, label=f'{tc} K')
        for a_val, tc in zip(ALPHAS, TC_LIST)]
h_tc.append(Line2D([], [], color='C3', ls='--', lw=2, label='liquidus'))
ax[2].legend(handles=h_ws, fontsize=10, loc='lower left')
ax[1].legend(handles=h_tc, fontsize=8, loc='lower left')
ax[3].legend(handles=[Line2D([], [], marker='o', color='k', ls='',
                             label='Irving 2023 (CMB)')],
             fontsize=9, loc='lower left')

ax[0].set_xlabel('P (GPa)', fontsize=14)
ax[1].set_xlabel('T (K)', fontsize=14)
ax[2].set_xlabel(r'$\rho$ (g/cm$^3$)', fontsize=14)
ax[3].set_xlabel('V$_P$ (km/s)', fontsize=14)
ax[0].set_ylabel('Depth (km)', fontsize=14)
for a in ax:
    a.tick_params(labelsize=12); a.grid(alpha=0.3)
    a.axhline(R_MARS - R_cmb, ls=':', color='gray', lw=1)
ax[0].set_ylim(R_MARS, R_MARS - R_cmb - 100)
fig.suptitle(f'Sensitivity to w$_S$ (color) and T$_{{CMB}}$ (alpha)   '
             f'({FE_SOURCE}+{FES_SOURCE})', fontsize=14)
plt.tight_layout()
plt.show()



