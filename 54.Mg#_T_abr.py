#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:31:57 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
52_phase_diagram_abgL.py
Fo-Fa 相圖：alpha / beta(Wad) / gamma(Rw) / Liquid
===================================================
固-固：Akaogi 1989 Table 3 + Table 5
       ΔV_eff 校準到 Katsura & Ito 1989 (Fo) / Yagi 1987 (Fa) 錨點
固-液：48_melting_Mg#_W.py 的方程（W_MELT 在 solid 上, liquid ideal）
       + dG(a→φ)(P,T) 位移把 β+L, γ+L 接上
       Tm_Fa 由 Pierru 2026 Table 3 反推（各錨點用觀測相 a/a/b/g）

參考相 = α。所有多形體位移：
  dG(a→φ)(P,T) = ΔH − TΔS + ΔV_eff·P   （ΔV_eff 吸收壓縮+熱膨脹）

輸出：
  Fig A: fixed-T P-X（α+β / β+γ / α+γ / φ+L loops + 三相線）
  Fig B: fixed-P T-X
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

# ============================================================
# 校準摘要
# ============================================================
print("="*70)
print("ΔV_eff 校準結果 (cm3/mol)")
print("="*70)
for k, v in TRANS.items():
    print(f"  {k[0]} {k[1]}→{k[2]}:  DH={v[0]:>8.0f}  DS={v[1]:>6.1f}  "
          f"DV_eff={v[2]:.3f}")

print("\n端元轉變壓力 (GPa):")
for comp in ['Fo', 'Fa']:
    for (u, v) in [('a','b'), ('b','g'), ('a','g')]:
        row = f"  {comp} {u}→{v}: "
        for T in [1473.0, 1873.0]:
            p = fsolve(lambda P: dG(comp,u,v,float(P[0]),T), 10.0)[0]
            row += f"  {p:6.2f} @{T:.0f}K"
        print(row)

print("\nTm_Fa 錨點（含相位移的反推）:")
for p, tm in zip(_aP, _aTm):
    print(f"  P={p:<5} phase={PIERRU[p][1]}  Tsol={PIERRU[p][0]:.0f}K"
          f"  → Tm_Fa(α ref) = {tm:.0f} K")

# ============================================================
# Fig A: fixed-T P-X
# ============================================================
P_grid = np.linspace(3.0, 22.0, 240)
ss_pairs   = [('a','b'), ('b','g'), ('a','g')]
ss_colors  = {('a','b'):'tab:blue', ('b','g'):'tab:purple',
              ('a','g'):'tab:green'}
melt_colors = {'a':'tab:red', 'b':'darkorange', 'g':'brown'}
ph_name = {'a':r'$\alpha$', 'b':r'$\beta$', 'g':r'$\gamma$'}

figA, axesA = plt.subplots(1, len(T_PX_list),
                           figsize=(8*len(T_PX_list), 9))
axesA = np.atleast_1d(axesA)

for ax, T in zip(axesA, T_PX_list):
    # 三相線（先解，供裁切用）
    Xa3, Xb3, Xg3, P3 = solve_univariant(T)

    # 固固 loops
    for pair in ss_pairs:
        X1 = np.full(len(P_grid), np.nan)
        X2 = np.full(len(P_grid), np.nan)
        for i, P in enumerate(P_grid):
            X1[i], X2[i] = solve_ss(pair[0], pair[1], P, T)
        if STABLE_ONLY and not np.isnan(P3):
            if pair == ('a','g'):
                m = P_grid > P3          # α+γ 只在三相線以下穩定
            else:
                m = P_grid < P3          # α+β, β+γ 只在三相線以上
            X1[m] = np.nan; X2[m] = np.nan
        c = ss_colors[pair]
        lbl = f'{ph_name[pair[0]]}+{ph_name[pair[1]]}'
        ax.plot(X1, P_grid, '-',  color=c, lw=2, label=lbl)
        ax.plot(X2, P_grid, '-',  color=c, lw=2)

    # 三相線
    if not np.isnan(P3):
        ax.plot([Xg3, Xb3], [P3, P3], 'k-', lw=2.5, zorder=8)
        for x in [Xa3, Xb3, Xg3]:
            ax.plot(x, P3, 'ko', ms=7, zorder=9)
        ax.text(Xb3+0.01, P3+0.15,
                f'{P3:.1f} GPa', fontsize=10, fontweight='bold')

    # 固液 loops（三個多形體）
    if SHOW_MELT:
        for ph in ['a', 'b', 'g']:
            XS = np.full(len(P_grid), np.nan)
            XL = np.full(len(P_grid), np.nan)
            for i, P in enumerate(P_grid):
                XS[i], XL[i] = solve_melt(ph, P, T)
            c = melt_colors[ph]
            ax.plot(XS, P_grid, '-',  color=c, lw=1.8,
                    label=f'{ph_name[ph]}+L sol')
            ax.plot(XL, P_grid, '--', color=c, lw=1.8)

    ax.axvline(0.75, color='gray', ls='--', lw=1, alpha=0.6)
    ax.text(0.755, P_grid[0]+0.3, 'Mars 0.75', fontsize=9, color='gray')

    # 相區標籤
    if STABLE_ONLY and not np.isnan(P3):
        P_ab = fsolve(lambda P: dG('Fo','a','b',float(P[0]),T), 14.0)[0]
        P_bg = fsolve(lambda P: dG('Fo','b','g',float(P[0]),T), 19.0)[0]
        ax.text(0.45, min(P_bg+1.2, P_grid[-1]-0.8), r'$\gamma$ (Rw)',
                fontsize=14, fontweight='bold')
        ax.text(0.90, (P_ab+P_bg)/2, r'$\beta$ (Wd)',
                fontsize=14, fontweight='bold', ha='center')
        ax.text(0.40, (P_grid[0]+P3)/2, r'$\alpha$ (Ol)',
                fontsize=14, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(P_grid[0], P_grid[-1])
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14)
    ax.set_ylabel('P (GPa)', fontsize=14)
    ax.set_title(f'const T = {T:.0f} K', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

figA.suptitle(
    'Fo-Fa P-X — solid loops: Akaogi 1989 (calibrated) | '
    'melt loops: 48-style, Pierru-anchored, W$_{melt}$=%.0f kJ' % (W_MELT/1000),
    fontsize=12)
figA.tight_layout()
#figA.savefig(f'{outdir}/52_PX_abgL.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {outdir}/52_PX_abgL.png")

# ============================================================
# Fig B: fixed-P T-X
# ============================================================
figB, axesB = plt.subplots(2, 2, figsize=(14, 13))
axesB = axesB.flatten()

for ax, P in zip(axesB, P_TX_list):
    TmFo_P = Tm_Fo(P)
    T_grid = np.linspace(1200.0, TmFo_P*1.05, 260)

    # 三相線（固+固+液）
    ab = solve_3ph_L('a', 'b', P)   # (Xa, Xb, XL, T)
    bg = solve_3ph_L('b', 'g', P)
    ag = solve_3ph_L('a', 'g', P)
    T_abL, T_bgL, T_agL = ab[3], bg[3], ag[3]

    three_poly = (not np.isnan(T_abL)) and (not np.isnan(T_bgL)) \
                 and (T_abL > T_bgL)

    def melt_window(ph):
        if not STABLE_ONLY:
            return (-np.inf, np.inf)
        if three_poly:
            if ph == 'a': return (T_abL, np.inf)
            if ph == 'b': return (T_bgL, T_abL)
            if ph == 'g': return (-np.inf, T_bgL)
        elif not np.isnan(T_agL):
            if ph == 'a': return (T_agL, np.inf)
            if ph == 'g': return (-np.inf, T_agL)
            return (np.nan, np.nan)          # β 不出現
        return (-np.inf, np.inf)

    def ss_window(pair):
        if not STABLE_ONLY:
            return np.inf
        if three_poly:
            if pair == ('a','b'): return T_abL
            if pair == ('b','g'): return T_bgL
            return -np.inf                   # α+γ 藏起來
        elif not np.isnan(T_agL):
            if pair == ('a','g'): return T_agL
            return -np.inf
        return np.inf

    # 固固 loops（近垂直帶，裁到三相線以下）
    for pair in ss_pairs:
        Tcut = ss_window(pair)
        X1 = np.full(len(T_grid), np.nan)
        X2 = np.full(len(T_grid), np.nan)
        for i, T in enumerate(T_grid):
            if T > Tcut:
                continue
            X1[i], X2[i] = solve_ss(pair[0], pair[1], P, T)
        if np.all(np.isnan(X1)):
            continue
        c = ss_colors[pair]
        lbl = f'{ph_name[pair[0]]}+{ph_name[pair[1]]}'
        ax.plot(X1, T_grid, '-', color=c, lw=2, label=lbl)
        ax.plot(X2, T_grid, '-', color=c, lw=2)

    # 固液 loops（裁到各自穩定溫窗）
    for ph in ['a', 'b', 'g']:
        Tlo, Thi = melt_window(ph)
        if np.isnan(Tlo):
            continue
        XS = np.full(len(T_grid), np.nan)
        XL = np.full(len(T_grid), np.nan)
        for i, T in enumerate(T_grid):
            if not (Tlo <= T <= Thi):
                continue
            XS[i], XL[i] = solve_melt(ph, P, T)
        if np.all(np.isnan(XS)):
            continue
        c = melt_colors[ph]
        ax.plot(XS, T_grid, '-',  color=c, lw=1.8,
                label=f'{ph_name[ph]}+L')
        ax.plot(XL, T_grid, '--', color=c, lw=1.8)

    # 水平三相線
    for tri, name in [(ab, r'$\alpha\beta$L'), (bg, r'$\beta\gamma$L'),
                      (ag, r'$\alpha\gamma$L')]:
        X1t, X2t, XLt, Tt = tri
        if np.isnan(Tt):
            continue
        if STABLE_ONLY and three_poly and name == r'$\alpha\gamma$L':
            continue
        if STABLE_ONLY and (not three_poly) and name != r'$\alpha\gamma$L':
            continue
        ax.plot([XLt, X1t], [Tt, Tt], 'k-', lw=2.5, zorder=8)
        for x in [X1t, X2t, XLt]:
            ax.plot(x, Tt, 'ko', ms=6, zorder=9)
        ax.text(XLt-0.02, Tt, name, fontsize=11, fontweight='bold',
                ha='right', va='center')

    # Pierru 錨點（若壓力吻合）
    for Pp, (Tsol, phx) in PIERRU.items():
        if abs(Pp - P) < 0.5:
            ax.scatter([X26], [Tsol], marker='s', s=220,
                       color='limegreen', edgecolors='darkgreen',
                       zorder=11, lw=2,
                       label='Pierru T$_{sol}$ (Table 3)')

    ax.plot(1.0, TmFo_P, 'b^', ms=10, zorder=6)
    ax.axvline(0.75, color='gray', ls='--', lw=1, alpha=0.6)
    ax.text(0.30, TmFo_P*1.01, 'L', fontsize=16, fontweight='bold',
            color='darkred')

    ax.set_xlim(0, 1)
    ax.set_ylim(1200, TmFo_P*1.05)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14)
    ax.set_ylabel('Temperature (K)', fontsize=14)
    ax.set_title(f'P = {P} GPa', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

figB.suptitle(
    'Fo-Fa T-X with $\\alpha/\\beta/\\gamma$ polymorphs + melting loops',
    fontsize=13)
figB.tight_layout()
#figB.savefig(f'{outdir}/52_TX_abgL.png', dpi=150, bbox_inches='tight')
print(f"Saved: {outdir}/52_TX_abgL.png")