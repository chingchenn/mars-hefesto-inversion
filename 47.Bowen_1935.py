#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 22:42:56 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
Fo-Fa Phase Diagram at 1 atm
=============================
比較兩個版本：

Version 1 — Bowen & Schairer 1935 (Fig. 28)：
  ln(XL/XS) = (L/R)*(1/Tm_Fo - 1/T)          ← 不帶 2
  ln((1-XL)/(1-XS)) = (L/R)*(1/Tm_Fa - 1/T)  ← 同一個 L，不帶 2
  L = 14000 cal/mol = 58576 J/mol

Version 2 — Philpotts & Ague (eq 10.18 / 10.20)：
  2*ln(XL/XS)         + W/(RT)*(1-XS)^2 = (DH_Fo/R)*(1/Tm_Fo - 1/T)
  2*ln((1-XL)/(1-XS)) + W/(RT)*XS^2     = (DH_Fa/R)*(1/Tm_Fa - 1/T)
  DH_Fo = 142 kJ/mol, DH_Fa = 89 kJ/mol

★ 改動 W_list 測試不同 W 值 ★
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ============================================================
# ★ 改動這裡 ★
# ============================================================
W_list = [0, 2500, 8000, 15000, 20000]   # J/mol，用在 Version 2
W_list = [0, 2500,]

# ============================================================
# 常數
# ============================================================
R     = 8.314
Tm_Fo = 2163.0     # K  (1890°C)
Tm_Fa = 1478.0     # K  (1205°C)
MW_Fo = 140.69
MW_Fa = 203.77

# B&S 1935
L_BS  = 14000 * 4.184   # 58576 J/mol

# Modern
DH_Fo = 142000.0   # J/mol
DH_Fa = 89000.0    # J/mol

# ============================================================
# 換算
# ============================================================
def mol_to_wt_Fa(X_Fo):
    """molar X_Fo → wt% Fa"""
    wt_Fo = X_Fo * MW_Fo
    wt_Fa = (1 - X_Fo) * MW_Fa
    return wt_Fa / (wt_Fo + wt_Fa) * 100

# ============================================================
# Version 1: B&S 1935
# ln(XL/XS) = (L/R)*(1/Tm_Fo - 1/T)
# ln((1-XL)/(1-XS)) = (L/R)*(1/Tm_Fa - 1/T)
# ============================================================
def solve_BS(T):
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = np.log(XL/XS)           - (L_BS/R)*(1/Tm_Fo - 1/T)
        e2 = np.log((1-XL)/(1-XS))   - (L_BS/R)*(1/Tm_Fa - 1/T)
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.90,0.70],[0.80,0.60],[0.70,0.50]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier==1 and np.max(np.abs(info['fvec']))<1e-8
                and 0<XS<1 and 0<XL<1 and XS>XL):
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

# ============================================================
# Version 2: Modern (Philpotts & Ague eq 10.18/10.20)
# 2*ln(XL/XS)         + W/(RT)*(1-XS)^2 = (DH_Fo/R)*(1/Tm_Fo - 1/T)
# 2*ln((1-XL)/(1-XS)) + W/(RT)*XS^2     = (DH_Fa/R)*(1/Tm_Fa - 1/T)
# ============================================================
def solve_modern(T, W):
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = (2*np.log(XL/XS)
              + W/(R*T)*(1-XS)**2
              - (DH_Fo/R)*(1/Tm_Fo - 1/T))
        e2 = (2*np.log((1-XL)/(1-XS))
              + W/(R*T)*XS**2
              - (DH_Fa/R)*(1/Tm_Fa - 1/T))
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.90,0.70],[0.80,0.60],[0.70,0.50]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier==1 and np.max(np.abs(info['fvec']))<1e-8
                and 0<XS<1 and 0<XL<1 and XS>XL):
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

# ============================================================
# 建立曲線
# ============================================================
def build_curves(solver):
    T_arr = np.linspace(Tm_Fa*0.997, Tm_Fo*1.001, 600)
    wS, wL, Tp = [], [], []
    for T in T_arr:
        if T >= Tm_Fo:
            wS.append(0.0);   wL.append(0.0)
        elif T <= Tm_Fa:
            wS.append(100.0); wL.append(100.0)
        else:
            XS, XL = solver(T)
            if np.isnan(XS): continue
            wS.append(mol_to_wt_Fa(XS))
            wL.append(mol_to_wt_Fa(XL))
        Tp.append(T - 273.15)
    return np.array(wS), np.array(wL), np.array(Tp)

# ============================================================
# Bowen & Schairer 1935 實驗數據點
# X: wt% Fa,  Y: °C
# solidus ±8°C, liquidus ±4°C
# ============================================================
bs_sol_wt = np.array([0,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100])
bs_sol_T  = np.array([1890,1800,1700,1595,1500,1420,1375,1340,1310,1260,1205])
bs_liq_wt = np.array([0,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100])
bs_liq_T  = np.array([1890,1860,1840,1810,1775,1730,1680,1620,1545,1430,1205])

# ============================================================
# 畫圖
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))

# --- Version 1: B&S 1935 equations ---
wS, wL, Tp = build_curves(solve_BS)
ax.plot(wS, Tp, 'b-',  lw=2.5,
        label=f'B&S 1935 eq: $\\ln(X^L/X^S)=(L/R)(1/T_m-1/T)$, L={L_BS/1000:.0f} kJ')
ax.plot(wL, Tp, 'b--', lw=2.5)

# --- Version 2: Modern equations, varying W ---
colors_mod = plt.cm.Reds(np.linspace(0.4, 0.9, len(W_list)))
for W, col in zip(W_list, colors_mod):
    wS, wL, Tp = build_curves(lambda T, W=W: solve_modern(T, W))
    lbl = (f'Modern eq: W={W/1000:.1f} kJ/mol'
           + (' (ideal)' if W == 0 else ''))
    ax.plot(wS, Tp, '-',  color=col, lw=2.0, label=lbl)
    ax.plot(wL, Tp, '--', color=col, lw=2.0)

# --- data---
ax.errorbar(bs_sol_wt, bs_sol_T, yerr=8,
            fmt='ko', ms=7, capsize=4, lw=1.5, zorder=10,
            label='B&S 1935 solidus data (±8°C)')
ax.errorbar(bs_liq_wt, bs_liq_T, yerr=4,
            fmt='k^', ms=7, capsize=4, lw=1.5, zorder=10,
            label='B&S 1935 liquidus data (±4°C)')

# --- 右軸 K ---
ax_K = ax.twinx()
ax_K.set_ylim(ax.get_ylim()[0]+273.15, ax.get_ylim()[1]+273.15)
ax_K.set_ylabel('Temperature (K)', fontsize=12)

# --- 上軸 wt% Fo ---
ax2 = ax.twiny()
ax2.set_xlim(0, 100)
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.set_xticklabels(['100', '80', '60', '40', '20', '0'])
ax2.set_xlabel('Wt% Fo (Mg₂SiO₄)', fontsize=11)

ax.set_xlabel('Wt% Fa (Fe₂SiO₄)', fontsize=13)
ax.set_ylabel('Temperature (°C)', fontsize=13)
ax.set_title(
    'Fo-Fa Phase Diagram at 1 atm\n'
    'Blue: B&S 1935  |  Red: Philpotts & Ague eq 10.18/10.20\n'
    'Solid = solidus, Dashed = liquidus  |  Black = experimental data',
    fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(1100, 2000)

ax.text(10, 1860, 'Liquid',
        fontsize=11, color='darkred', fontweight='bold')
ax.text(5,  1380, 'Olivine$_{ss}$',
        fontsize=11, color='darkblue', fontweight='bold')
ax.text(28, 1640, 'Ol + L',
        fontsize=11, color='darkgreen', fontweight='bold')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 atm Fo-Fa loop: model vs Bowen & Schairer 1935
Wa and W_LIQ are degenerate -- the loop only constrains a combination of them.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

R = 8.314
MW_Fo, MW_Fa = 140.69, 203.77
DH_Fo, DH_Fa = 142000., 89300.      # Richet et al. 1993
Tm_Fo, Tm_Fa = 2163., 1490.         # Bowen&Anderson 1914 / Richet 1993

# ---- Bowen & Schairer 1935 data (wt% Fa, degC) ----
wt   = np.array([0,10,20,30,40,50,60,70,80,90,100.])
Tsol = np.array([1890,1800,1700,1595,1500,1420,1375,1340,1310,1260,1205.])
Tliq = np.array([1890,1860,1840,1810,1775,1730,1680,1620,1545,1430,1205.])

# ---- cases: (W_alpha, W_LIQ, colour, label) ----
CASES = [
    (2500.,     0., '#35838D', r'SLB $W_\alpha$=2.5k, $W_{LIQ}$=0  (current)'),
    (2500.,  8000., '#C47F3E', r'SLB $W_\alpha$=2.5k, $W_{LIQ}$=8k'),
    (12600.,    0., '#7B9E87', r'Akaogi $W_\alpha$=12.6k, $W_{LIQ}$=0'),
    (25000., 20000.,'#CD5C5C', r'best fit: $W_\alpha$=25k, $W_{LIQ}$=20k'),
]

fig, ax = plt.subplots(figsize=(9,8))
Tg = np.linspace(Tm_Fo*0.999, Tm_Fa*1.001, 400)      # scan downward

for (Wa, WL, c, lbl) in CASES:
    ws, wl, tp = [], [], []
    X = [0.999, 0.995]                                # continuation seed
    for T in Tg:
        def eqs(v):
            Xs, Xl = np.clip(v, 1e-9, 1-1e-9)
            e1 = 2*np.log(Xl/Xs) + (WL*(1-Xl)**2 - Wa*(1-Xs)**2)/(R*T) \
                 - DH_Fo/R*(1/Tm_Fo - 1/T)
            e2 = 2*np.log((1-Xl)/(1-Xs)) + (WL*Xl**2 - Wa*Xs**2)/(R*T) \
                 - DH_Fa/R*(1/Tm_Fa - 1/T)
            return [e1, e2]
        s, info, ok, _ = fsolve(eqs, X, full_output=True)
        if ok==1 and max(abs(info['fvec']))<1e-8 and 0<s[1]<s[0]<1:
            X = list(s)
            ws.append((1-s[0])*MW_Fa/(s[0]*MW_Fo+(1-s[0])*MW_Fa)*100)
            wl.append((1-s[1])*MW_Fa/(s[1]*MW_Fo+(1-s[1])*MW_Fa)*100)
        else:
            ws.append(np.nan); wl.append(np.nan)
        tp.append(T - 273.15)
    ax.plot(ws, tp, '-',  color=c, lw=2, label=lbl)
    ax.plot(wl, tp, '--', color=c, lw=2)

ax.errorbar(wt, Tsol, yerr=8, fmt='ko', ms=7, capsize=4, lw=1.5, zorder=10,
            label='B&S 1935 solidus')
ax.errorbar(wt, Tliq, yerr=4, fmt='k^', ms=7, capsize=4, lw=1.5, zorder=10,
            label='B&S 1935 liquidus')

ax.set_xlim(0,100); ax.set_ylim(1150,1950)
ax.set_xlabel('Wt% Fa (Fe$_2$SiO$_4$)', fontsize=14)
ax.set_ylabel('Temperature ($^\\circ$C)', fontsize=14)
ax.tick_params(labelsize=14); ax.grid(alpha=0.3)
ax.set_title('1 atm Fo-Fa loop: model vs Bowen & Schairer 1935\n'
             'solid = solidus, dashed = liquidus', fontsize=14)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()