#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:34:19 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
53_pxt_3d.py — Fo-Fa P-X-T 3D prism（52 版 solver 核心 + 等壓截面堆疊）
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ============================================================
# ★ 改動這裡 ★
# ============================================================
T_PX_list = [1500.0, 1873.0]           # fixed-T P-X 面板
P_TX_list = [8.0, 14.1, 16.0, 17.7]    # fixed-P T-X 面板
W_MELT    = 8000.0                     # 固液 W（Pierru Table 4 fit, 48 版）
STABLE_ONLY = True                     # True: 用三相線裁掉 metastable 延伸
SHOW_MELT   = False                    # P-X 圖是否疊固液 loop
outdir    = '/mnt/user-data/outputs'

# 固固 Margules（per Mg2SiO4 = 2 × one-site, Akaogi Table 5, α 對稱化）
def W_ss(ph, P):
    if ph == 'a': return 12600.0 + 260.0*P   # 2*(8.4+4.2)/2 kJ + 2*0.13 kJ/GPa
    else:         return 7800.0              # β, γ: 2*3.9 kJ

# ============================================================
# 常數
# ============================================================
R   = 8.314
X26 = 0.75
DH_FO, DH_FA = 142000.0, 89000.0   # α 端元熔化焓（olivine, Stixrude+Hess）

# ============================================================
# Akaogi 1989 Table 3（Fo a→b 用調整值 27130 / −9.0, footnote b）
# [DH J/mol, DS J/mol/K, DV cm3/mol]  ← DV 會被錨點校準覆蓋
# ============================================================
TRANS = {
    ('Fo','a','b'): [27130.0,  -9.0, -3.16],
    ('Fo','b','g'): [ 9080.0,  -7.3, -0.98],
    ('Fa','a','b'): [ 9620.0, -10.9, -3.20],
    ('Fa','b','g'): [-5790.0,  -3.1, -1.04],
}

# ── ΔV_eff 校準 ──────────────────────────────────────────────
# Fo a→b: 14.4 GPa @1473K, Fo b→g: 18.5 GPa @1473K (Katsura & Ito 1989)
for key, (P_tr, T_tr) in {('Fo','a','b'): (14.4, 1473.0),
                          ('Fo','b','g'): (18.5, 1473.0)}.items():
    DH, DS, _ = TRANS[key]
    TRANS[key][2] = -(DH - T_tr*DS) / (P_tr*1000.0)

# Fa a→g: 5.3 GPa @1273K (Yagi 1987) → 縮放因子同時套用 a→b, b→g
DHg = TRANS[('Fa','a','b')][0] + TRANS[('Fa','b','g')][0]
DSg = TRANS[('Fa','a','b')][1] + TRANS[('Fa','b','g')][1]
DVg_eff = -(DHg - 1273.0*DSg) / (5.3*1000.0)
f_Fa = DVg_eff / (TRANS[('Fa','a','b')][2] + TRANS[('Fa','b','g')][2])
TRANS[('Fa','a','b')][2] *= f_Fa
TRANS[('Fa','b','g')][2] *= f_Fa

_ORD = {'a': 0, 'b': 1, 'g': 2}

def dG(comp, f1, f2, P, T):
    """μ°(f2) − μ°(f1), J/mol, per Mg2SiO4/Fe2SiO4"""
    if f1 == f2:
        return 0.0
    sgn = 1.0
    if _ORD[f1] > _ORD[f2]:
        f1, f2, sgn = f2, f1, -1.0
    tot = 0.0
    for (u, v) in [('a','b'), ('b','g')]:
        if _ORD[u] >= _ORD[f1] and _ORD[v] <= _ORD[f2]:
            DH, DS, DV = TRANS[(comp, u, v)]
            tot += DH - T*DS + DV*1000.0*P
    return sgn*tot

# ============================================================
# 端元熔化
# ============================================================
def Tm_Fo(P):   # Ohtani & Kumazawa 1981
    return 2163.0 * (P/65.67 + 1.0) ** (1.0/0.7809)

# Pierru 2026 Table 3 錨點：{P: (T_sol, 觀測相)}
PIERRU = {7.95: (1890.0, 'a'), 9.6: (1960.0, 'a'),
          14.1: (2090.0, 'b'), 17.7: (2200.0, 'g')}

def _melt_eqs(XS, XL, T, P, ph, TmFa):
    """48 版固液方程 + dG(a→φ) 位移；ph='a' 時完全還原 48 版"""
    e1 = (2*np.log(XL/XS) + W_MELT/(R*T)*(1-XS)**2
          - (DH_FO/R)*(1/Tm_Fo(P) - 1/T) - dG('Fo','a',ph,P,T)/(R*T))
    e2 = (2*np.log((1-XL)/(1-XS)) + W_MELT/(R*T)*XS**2
          - (DH_FA/R)*(1/TmFa - 1/T) - dG('Fa','a',ph,P,T)/(R*T))
    return e1, e2

def _backcalc_TmFa(P, Tsol, ph):
    def eqs(v):
        TmFa, XL = v
        XL = np.clip(XL, 1e-9, 1-1e-9)
        return _melt_eqs(X26, XL, Tsol, P, ph, TmFa)
    for g in ([1400,0.40],[1600,0.35],[1200,0.45],[1800,0.30]):
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8:
            return sol
    return [np.nan, np.nan]

_aP  = np.array(sorted(PIERRU))
_aTm = np.array([_backcalc_TmFa(p, *PIERRU[p])[0] for p in _aP])

def Tm_Fa(P):
    """α-Fa 參考熔點：錨點線性內插 + 端點斜率外插"""
    if P <= _aP[0]:
        s = (_aTm[1]-_aTm[0])/(_aP[1]-_aP[0])
        return _aTm[0] + s*(P-_aP[0])
    if P >= _aP[-1]:
        s = (_aTm[-1]-_aTm[-2])/(_aP[-1]-_aP[-2])
        return _aTm[-1] + s*(P-_aP[-1])
    return float(np.interp(P, _aP, _aTm))

# ============================================================
# solvers
# ============================================================
def solve_melt(ph, P, T):
    """φ+L loop at (P,T) → (XS, XL)"""
    TmFa = Tm_Fa(P)
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        return _melt_eqs(XS, XL, T, P, ph, TmFa)
    for x0 in ([0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70],
               [0.5,0.3],[0.3,0.15]):
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                and 1e-6 < XL < XS < 1-1e-6):
            return float(XS), float(XL)
    return np.nan, np.nan

def solve_ss(f1, f2, P, T):
    """f1+f2 固固 loop at (P,T) → (X1, X2)；f2=高壓相（局部較 Fe-rich）"""
    dgFo = dG('Fo', f1, f2, P, T)
    dgFa = dG('Fa', f1, f2, P, T)
    W1, W2 = W_ss(f1, P), W_ss(f2, P)
    def eqs(v):
        X1, X2 = np.clip(v, 1e-9, 1-1e-9)
        e1 = (2*np.log(X2/X1)
              + (W2*(1-X2)**2 - W1*(1-X1)**2)/(R*T) + dgFo/(R*T))
        e2 = (2*np.log((1-X2)/(1-X1))
              + (W2*X2**2 - W1*X1**2)/(R*T) + dgFa/(R*T))
        return [e1, e2]
    for x0 in ([0.90,0.80],[0.80,0.65],[0.65,0.45],[0.45,0.25],
               [0.97,0.92],[0.25,0.10],[0.10,0.03]):
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        X1, X2 = sol
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                and 1e-6 < X2 < X1 < 1-1e-6):
            return float(X1), float(X2)
    return np.nan, np.nan

def solve_3ph_L(f1, f2, P):
    """f1+f2+L 三相線 at fixed P → (X1, X2, XL, T)；順序 XL < X2 < X1"""
    TmFa = Tm_Fa(P)
    def eqs(v):
        X1, X2, XL, T = v
        X1, X2, XL = np.clip([X1, X2, XL], 1e-9, 1-1e-9)
        e1, e2 = _melt_eqs(X1, XL, T, P, f1, TmFa)
        e3, e4 = _melt_eqs(X2, XL, T, P, f2, TmFa)
        return [e1, e2, e3, e4]
    for x0 in ([0.90,0.80,0.55,2200.],[0.85,0.72,0.45,2100.],
               [0.95,0.88,0.65,2400.],[0.75,0.60,0.35,2000.],
               [0.60,0.45,0.25,1900.],[0.98,0.94,0.75,2600.]):
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        X1, X2, XL, T = sol
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                and 0 < XL < X2 < X1 < 1 and 1000 < T < 3500):
            return sol
    return [np.nan]*4

def solve_univariant(T):
    """α+β+γ 三相線 at fixed T → (Xa, Xb, Xg, P)；順序 Xg < Xa < Xb"""
    def eqs(v):
        Xa, Xb, Xg, P = v
        Xa, Xb, Xg = np.clip([Xa, Xb, Xg], 1e-9, 1-1e-9)
        Wa, Wb, Wg = W_ss('a',P), W_ss('b',P), W_ss('g',P)
        e1 = (2*np.log(Xb/Xa) + (Wb*(1-Xb)**2 - Wa*(1-Xa)**2)/(R*T)
              + dG('Fo','a','b',P,T)/(R*T))
        e2 = (2*np.log((1-Xb)/(1-Xa)) + (Wb*Xb**2 - Wa*Xa**2)/(R*T)
              + dG('Fa','a','b',P,T)/(R*T))
        e3 = (2*np.log(Xg/Xb) + (Wg*(1-Xg)**2 - Wb*(1-Xb)**2)/(R*T)
              + dG('Fo','b','g',P,T)/(R*T))
        e4 = (2*np.log((1-Xg)/(1-Xb)) + (Wg*Xg**2 - Wb*Xb**2)/(R*T)
              + dG('Fa','b','g',P,T)/(R*T))
        return [e1, e2, e3, e4]
    for x0 in ([0.87,0.74,0.55,13.9],[0.82,0.68,0.48,13.0],
               [0.90,0.80,0.62,14.5],[0.75,0.60,0.40,12.0]):
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        Xa, Xb, Xg, P = sol
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                and 0 < Xg < Xb < Xa < 1 and 3 < P < 22):
            return sol
    return [np.nan]*4


melt_colors = {'a':'tab:red', 'b':'darkorange', 'g':'brown'}
X26 = 0.75

# 3D 專用設定
P3D_MIN, P3D_MAX, P3D_STEP = 6.0, 20.0, 1.0
ELEV,  AZIM  = 22, -60
ELEV2, AZIM2 = 12, -35

# ============================================================
# 3D P-X-T prism：等壓 T-X 截面沿 P 疊起
# ============================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa

P_sections = np.arange(P3D_MIN, P3D_MAX + 1e-9, P3D_STEP)
nT = 140

fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(projection='3d')

tri_locus = {'ab': [], 'bg': [], 'ag': []}

for P in P_sections:
    TmFo_P = Tm_Fo(P)
    T_grid = np.linspace(1300.0, TmFo_P*1.02, nT)

    ab = solve_3ph_L('a', 'b', P)
    bg = solve_3ph_L('b', 'g', P)
    ag = solve_3ph_L('a', 'g', P)
    T_abL, T_bgL, T_agL = ab[3], bg[3], ag[3]
    three_poly = (not np.isnan(T_abL)) and (not np.isnan(T_bgL)) \
                 and (T_abL > T_bgL)

    if three_poly:
        tri_locus['ab'].append((P, *ab))
        tri_locus['bg'].append((P, *bg))
        windows = {'a': (T_abL, np.inf), 'b': (T_bgL, T_abL),
                   'g': (-np.inf, T_bgL)}
    elif not np.isnan(T_agL):
        tri_locus['ag'].append((P, *ag))
        windows = {'a': (T_agL, np.inf), 'g': (-np.inf, T_agL),
                   'b': (np.nan, np.nan)}
    else:
        windows = {'a': (-np.inf, np.inf), 'b': (np.nan, np.nan),
                   'g': (np.nan, np.nan)}

    for ph in ['a', 'b', 'g']:
        Tlo, Thi = windows[ph]
        if np.isnan(Tlo):
            continue
        XS = np.full(nT, np.nan); XL = np.full(nT, np.nan)
        for i, T in enumerate(T_grid):
            if not (Tlo <= T <= Thi):
                continue
            XS[i], XL[i] = solve_melt(ph, P, T)
        c = melt_colors[ph]
        ax.plot(XS, np.full(nT, P), T_grid, '-',  color=c, lw=1.4)
        ax.plot(XL, np.full(nT, P), T_grid, '--', color=c, lw=1.0,
                alpha=0.55)

    # 每個截面的三相水平線
    for tri, ok in [(ab, three_poly), (bg, three_poly),
                    (ag, (not three_poly) and (not np.isnan(T_agL)))]:
        X1t, X2t, XLt, Tt = tri
        if ok and not np.isnan(Tt):
            ax.plot([XLt, X1t], [P, P], [Tt, Tt], 'k-', lw=1.6,
                    alpha=0.8)

# 三相線 locus（沿 P 的曲線）
for key, lab in [('ab', r'$\alpha\beta$L'), ('bg', r'$\beta\gamma$L'),
                 ('ag', r'$\alpha\gamma$L')]:
    arr = np.array(tri_locus[key])
    if len(arr) < 2:
        continue
    Pv, X1v, X2v, XLv, Tv = arr.T
    for Xv, lw in [(X1v, 2.5), (X2v, 1.5), (XLv, 2.5)]:
        ax.plot(Xv, Pv, Tv, '-', color='k', lw=lw)
    ax.text(X1v[-1], Pv[-1], Tv[-1]+40, lab, fontsize=13,
            fontweight='bold')

# Pierru 錨點
for Pp, (Tsol, phx) in PIERRU.items():
    if P3D_MIN <= Pp <= P3D_MAX:
        ax.scatter([X26], [Pp], [Tsol], marker='s', s=140,
                   color='limegreen', edgecolors='darkgreen',
                   lw=1.5, zorder=10)

# 端元熔化曲線（prism 的兩條稜線）
Pv = np.linspace(P3D_MIN, P3D_MAX, 60)
ax.plot(np.ones_like(Pv), Pv, [Tm_Fo(p) for p in Pv],
        'b-', lw=2.5)
ax.plot(np.zeros_like(Pv), Pv, [Tm_Fa(p) for p in Pv],
        'r-', lw=2.5)

ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14, labelpad=12)
ax.set_ylabel('P (GPa)', fontsize=14, labelpad=12)
ax.set_zlabel('T (K)', fontsize=14, labelpad=12)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=14)
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_title('Fo-Fa P-X-T:isobaric sections + 3-phase loci\n'
             '(solid=solidus, dashed=liquidus, black=3-phase)',
             fontsize=13)

#fig.savefig(f'{outdir}/53_PXT_3d.png', dpi=150, bbox_inches='tight')
ax.view_init(elev=ELEV2, azim=AZIM2)
#fig.savefig(f'{outdir}/53_PXT_3d_view2.png', dpi=150,
#            bbox_inches='tight')
print(f"Saved: {outdir}/53_PXT_3d.png (+view2)")