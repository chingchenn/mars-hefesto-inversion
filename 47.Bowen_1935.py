# 47.Bowen_1935.py
# Fo-Fa melting loop at 1 atm compared with the Bowen & Schairer 1935 data (Fig. 28).
# Part 1 (fig1): B&S 1935 equations vs Philpotts & Ague eq 10.18/10.20 with a W list.
# Part 2 (fig2): solid W_alpha + liquid W_LIQ model (Richet 1993 constants). W_alpha and W_LIQ
#                are degenerate: the loop only constrains a combination of them.
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

W_list = [0, 2500,]      # J/mol, Part 1 modern equations (also tried [0, 2500, 8000, 15000, 20000])

R     = 8.314
MW_Fo = 140.69
MW_Fa = 203.77

# Bowen & Schairer 1935 data: wt% Fa, degC; solidus +-8 degC, liquidus +-4 degC
bs_wt    = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.])
bs_sol_T = np.array([1890, 1800, 1700, 1595, 1500, 1420, 1375, 1340, 1310, 1260, 1205.])
bs_liq_T = np.array([1890, 1860, 1840, 1810, 1775, 1730, 1680, 1620, 1545, 1430, 1205.])

# molar X_Fo -> wt% Fa:  wt% Fa = (1-X) MW_Fa / (X MW_Fo + (1-X) MW_Fa) * 100

# ══ Part 1 ═════════════════════════════════════════════════════════════════
Tm_Fo = 2163.0          # K (1890 degC)
Tm_Fa = 1478.0          # K (1205 degC)
L_BS  = 14000 * 4.184   # B&S 1935 latent heat, 14000 cal/mol = 58576 J/mol
DH_Fo = 142000.0        # J/mol
DH_Fa = 89000.0         # J/mol

# Version 1, B&S 1935 (no factor 2, same L for both end members):
#   ln(XL/XS)         = (L/R)(1/Tm_Fo - 1/T)
#   ln((1-XL)/(1-XS)) = (L/R)(1/Tm_Fa - 1/T)
def eqs_BS(v, T):
    XS, XL = np.clip(v, 1e-9, 1-1e-9)
    return [np.log(XL/XS)         - (L_BS/R)*(1/Tm_Fo - 1/T),
            np.log((1-XL)/(1-XS)) - (L_BS/R)*(1/Tm_Fa - 1/T)]

# Version 2, Philpotts & Ague eq 10.18 / 10.20:
#   2 ln(XL/XS)         + W/(RT) (1-XS)^2 = (DH_Fo/R)(1/Tm_Fo - 1/T)
#   2 ln((1-XL)/(1-XS)) + W/(RT) XS^2     = (DH_Fa/R)(1/Tm_Fa - 1/T)
def eqs_modern(v, T, W):
    XS, XL = np.clip(v, 1e-9, 1-1e-9)
    return [2*np.log(XL/XS)         + W/(R*T)*(1-XS)**2 - (DH_Fo/R)*(1/Tm_Fo - 1/T),
            2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2     - (DH_Fa/R)*(1/Tm_Fa - 1/T)]

# loop on a T grid: pure Fo above Tm_Fo, pure Fa below Tm_Fa, otherwise the first of 4 guesses
# that converges with 0 < XL < XS < 1; T points with no solution are dropped
def build_curves(eqs, args=()):
    T_arr = np.linspace(Tm_Fa*0.997, Tm_Fo*1.001, 600)
    XS_l, XL_l, T_l = [], [], []
    for T in T_arr:
        if T >= Tm_Fo:
            XS, XL = 1.0, 1.0
        elif T <= Tm_Fa:
            XS, XL = 0.0, 0.0
        else:
            XS, XL = np.nan, np.nan
            for x0 in [[0.85,0.65], [0.90,0.70], [0.80,0.60], [0.70,0.50]]:
                sol, info, ier, _ = fsolve(eqs, x0, args=(T,) + args, full_output=True)
                if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8
                        and 0 < sol[0] < 1 and 0 < sol[1] < 1 and sol[0] > sol[1]):
                    XS, XL = sol
                    break
            if np.isnan(XS):
                continue
        XS_l.append(XS); XL_l.append(XL); T_l.append(T - 273.15)
    XS, XL = np.array(XS_l), np.array(XL_l)
    wS = (1-XS)*MW_Fa / (XS*MW_Fo + (1-XS)*MW_Fa) * 100
    wL = (1-XL)*MW_Fa / (XL*MW_Fo + (1-XL)*MW_Fa) * 100
    return wS, wL, np.array(T_l)

fig1, ax = plt.subplots(figsize=(10, 10))
wS, wL, Tp = build_curves(eqs_BS)
ax.plot(wS, Tp, 'b-',  lw=2.5,
        label=f'B&S 1935 eq: $\\ln(X^L/X^S)=(L/R)(1/T_m-1/T)$, L={L_BS/1000:.0f} kJ')
ax.plot(wL, Tp, 'b--', lw=2.5)

colors_mod = plt.cm.Reds(np.linspace(0.4, 0.9, len(W_list)))
for W, col in zip(W_list, colors_mod):
    wS, wL, Tp = build_curves(eqs_modern, (W,))
    ax.plot(wS, Tp, '-',  color=col, lw=2.0, label=f'Modern eq: W={W/1000:.1f} kJ/mol' + (' (ideal)' if W == 0 else ''))
    ax.plot(wL, Tp, '--', color=col, lw=2.0)

ax.errorbar(bs_wt, bs_sol_T, yerr=8, fmt='ko', ms=7, capsize=4, lw=1.5, zorder=10, label='B&S 1935 solidus data (±8°C)')
ax.errorbar(bs_wt, bs_liq_T, yerr=4, fmt='k^', ms=7, capsize=4, lw=1.5, zorder=10, label='B&S 1935 liquidus data (±4°C)')

# right axis in K (limits taken from the autoscaled degC axis, before set_ylim below)
ax_K = ax.twinx()
ax_K.set_ylim(ax.get_ylim()[0]+273.15, ax.get_ylim()[1]+273.15)
ax_K.set_ylabel('Temperature (K)', fontsize=12)
ax_K.tick_params(labelsize=14)
# top axis in wt% Fo
ax2 = ax.twiny()
ax2.set_xlim(0, 100)
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.set_xticklabels(['100', '80', '60', '40', '20', '0'])
ax2.set_xlabel('Wt% Fo (Mg₂SiO₄)', fontsize=11)
ax2.tick_params(labelsize=14)

ax.set_xlabel('Wt% Fa (Fe₂SiO₄)', fontsize=13)
ax.set_ylabel('Temperature (°C)', fontsize=13)
ax.set_title('Fo-Fa at 1 atm: B&S 1935 (blue) vs Philpotts & Ague (red)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(1100, 2000)
ax.tick_params(labelsize=14)

# ══ Part 2: 1 atm loop with solid W_alpha and liquid W_LIQ ═════════════════
# NOTE: different end-member constants from Part 1
DH_Fo, DH_Fa = 142000., 89300.      # Richet et al. 1993
Tm_Fo, Tm_Fa = 2163., 1490.         # Bowen & Anderson 1914 / Richet 1993

# cases: (W_alpha, W_LIQ, colour, label)
CASES = [
    (2500.,     0., '#35838D', r'SLB $W_\alpha$=2.5k, $W_{LIQ}$=0  (current)'),
    (2500.,  8000., '#C47F3E', r'SLB $W_\alpha$=2.5k, $W_{LIQ}$=8k'),
    (12600.,    0., '#7B9E87', r'Akaogi $W_\alpha$=12.6k, $W_{LIQ}$=0'),
    (25000., 20000.,'#CD5C5C', r'best fit: $W_\alpha$=25k, $W_{LIQ}$=20k'),
]

#   2 ln(Xl/Xs)         + [WL (1-Xl)^2 - Wa (1-Xs)^2]/RT - DH_Fo/R (1/Tm_Fo - 1/T) = 0
#   2 ln((1-Xl)/(1-Xs)) + [WL Xl^2     - Wa Xs^2    ]/RT - DH_Fa/R (1/Tm_Fa - 1/T) = 0
# traced downward in T by continuation (seed = last converged solution)
fig2, ax = plt.subplots(figsize=(9, 8))
Tg = np.linspace(Tm_Fo*0.999, Tm_Fa*1.001, 400)
for (Wa, WL, c, lbl) in CASES:
    sol_list = []
    X = [0.999, 0.995]
    for T in Tg:
        def eqs(v):
            Xs, Xl = np.clip(v, 1e-9, 1-1e-9)
            e1 = 2*np.log(Xl/Xs) + (WL*(1-Xl)**2 - Wa*(1-Xs)**2)/(R*T) \
                 - DH_Fo/R*(1/Tm_Fo - 1/T)
            e2 = 2*np.log((1-Xl)/(1-Xs)) + (WL*Xl**2 - Wa*Xs**2)/(R*T) \
                 - DH_Fa/R*(1/Tm_Fa - 1/T)
            return [e1, e2]
        s, info, ok, _ = fsolve(eqs, X, full_output=True)
        if ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < s[1] < s[0] < 1:
            X = list(s)
            sol_list.append(s)
        else:
            sol_list.append([np.nan, np.nan])
    Xs, Xl = np.array(sol_list).T
    ws = (1-Xs)*MW_Fa / (Xs*MW_Fo + (1-Xs)*MW_Fa) * 100
    wl = (1-Xl)*MW_Fa / (Xl*MW_Fo + (1-Xl)*MW_Fa) * 100
    ax.plot(ws, Tg - 273.15, '-',  color=c, lw=2, label=lbl)
    ax.plot(wl, Tg - 273.15, '--', color=c, lw=2)

ax.errorbar(bs_wt, bs_sol_T, yerr=8, fmt='ko', ms=7, capsize=4, lw=1.5, zorder=10, label='B&S 1935 solidus')
ax.errorbar(bs_wt, bs_liq_T, yerr=4, fmt='k^', ms=7, capsize=4, lw=1.5, zorder=10, label='B&S 1935 liquidus')
ax.set_xlim(0, 100); ax.set_ylim(1150, 1950)
ax.set_xlabel('Wt% Fa (Fe$_2$SiO$_4$)', fontsize=14)
ax.set_ylabel('Temperature ($^\\circ$C)', fontsize=14)
ax.tick_params(labelsize=14); ax.grid(alpha=0.3)
ax.set_title('1 atm Fo-Fa loop: model vs Bowen & Schairer 1935', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.show()
