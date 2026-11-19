# 54.Mg#_T_abr.py
# Fo-Fa phase diagram: alpha / beta (Wad) / gamma (Rw) / liquid.
# Solid-solid: Akaogi 1989 Table 3 + Table 5, DV_eff calibrated to Katsura & Ito 1989 (Fo)
#              and Yagi 1987 (Fa) anchors.
# Solid-liquid: equations of 48_melting_Mg#_W.py (W_MELT on solid, ideal liquid)
#              + dG(a->phi)(P,T) shift to connect beta+L and gamma+L.
#              Tm_Fa back-calculated from Pierru 2026 Table 3 (observed phase a/a/b/g at each anchor).
# Reference phase = alpha. Polymorph shift:
#   dG(a->phi)(P,T) = DH - T*DS + DV_eff*P   (DV_eff absorbs compression + thermal expansion)
# Fig A: fixed-T P-X (a+b / b+g / a+g loops + univariant line)
# Fig B: fixed-P T-X (solid loops + melting loops + three-phase lines)
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

T_PX_list   = [1500.0, 1873.0]           # fixed-T P-X panels
P_TX_list   = [8.0, 14.1, 16.0, 17.7]    # fixed-P T-X panels
W_MELT      = 8000.0                     # solid-liquid W (Pierru Table 4 fit, version 48)
STABLE_ONLY = True                       # True: cut metastable extensions at three-phase lines
SHOW_MELT   = False                      # overlay melting loops on the P-X panels

R   = 8.314
X26 = 0.75
DH_FO, DH_FA = 142000.0, 89000.0   # alpha end-member enthalpy of fusion (olivine, Stixrude + Hess)

# Solid-solid Margules (per Mg2SiO4 = 2 x one-site, Akaogi Table 5, alpha symmetrised)
# alpha: 2*(8.4+4.2)/2 kJ + 2*0.13 kJ/GPa * P ; beta, gamma: 2*3.9 kJ
def W_ss(ph, P):
    if ph == 'a': return 12600.0 + 260.0*P
    else:         return 7800.0

# Akaogi 1989 Table 3 (Fo a->b uses adjusted 27130 / -9.0, footnote b)
# [DH J/mol, DS J/mol/K, DV cm3/mol]; DV is overwritten by the anchor calibration below
TRANS = {
    ('Fo','a','b'): [27130.0,  -9.0, -3.16],
    ('Fo','b','g'): [ 9080.0,  -7.3, -0.98],
    ('Fa','a','b'): [ 9620.0, -10.9, -3.20],
    ('Fa','b','g'): [-5790.0,  -3.1, -1.04],
}

# DV_eff calibration: dG = 0 at the anchor -> DV_eff = -(DH - T_tr*DS) / P_tr
# Fo a->b: 14.4 GPa @1473 K, Fo b->g: 18.5 GPa @1473 K (Katsura & Ito 1989)
for key, (P_tr, T_tr) in {('Fo','a','b'): (14.4, 1473.0),
                          ('Fo','b','g'): (18.5, 1473.0)}.items():
    DH, DS, _ = TRANS[key]
    TRANS[key][2] = -(DH - T_tr*DS) / (P_tr*1000.0)

# Fa a->g: 5.3 GPa @1273 K (Yagi 1987) -> one scale factor applied to both a->b and b->g
DHg = TRANS[('Fa','a','b')][0] + TRANS[('Fa','b','g')][0]
DSg = TRANS[('Fa','a','b')][1] + TRANS[('Fa','b','g')][1]
DVg_eff = -(DHg - 1273.0*DSg) / (5.3*1000.0)
f_Fa = DVg_eff / (TRANS[('Fa','a','b')][2] + TRANS[('Fa','b','g')][2])
TRANS[('Fa','a','b')][2] *= f_Fa
TRANS[('Fa','b','g')][2] *= f_Fa

ORD = {'a': 0, 'b': 1, 'g': 2}

# mu0(f2) - mu0(f1), J/mol, per Mg2SiO4 / Fe2SiO4: sum of DH - T*DS + DV*P over the steps f1 -> f2
def dG(comp, f1, f2, P, T):
    if f1 == f2:
        return 0.0
    sgn = 1.0
    if ORD[f1] > ORD[f2]:
        f1, f2, sgn = f2, f1, -1.0
    tot = 0.0
    for (u, v) in [('a','b'), ('b','g')]:
        if ORD[u] >= ORD[f1] and ORD[v] <= ORD[f2]:
            DH, DS, DV = TRANS[(comp, u, v)]
            tot += DH - T*DS + DV*1000.0*P
    return sgn*tot

# Fo melting, Ohtani & Kumazawa 1981: Tm = 2163 * (P/65.67 + 1)^(1/0.7809)
def Tm_Fo(P):
    return 2163.0 * (P/65.67 + 1.0) ** (1.0/0.7809)

# Pierru 2026 Table 3 anchors: {P: (T_sol, observed phase)}
PIERRU = {7.95: (1890.0, 'a'), 9.6: (1960.0, 'a'),
          14.1: (2090.0, 'b'), 17.7: (2200.0, 'g')}

# Solid-liquid equilibrium (version 48 + dG(a->phi) shift; ph='a' reduces exactly to version 48)
# e1: 2 ln(XL/XS) + W/(RT) (1-XS)^2 - DH_Fo/R (1/Tm_Fo - 1/T) - dG_Fo(a->ph)/(RT) = 0
# e2: 2 ln((1-XL)/(1-XS)) + W/(RT) XS^2 - DH_Fa/R (1/Tm_Fa - 1/T) - dG_Fa(a->ph)/(RT) = 0
def melt_eqs(XS, XL, T, P, ph, TmFa):
    e1 = (2*np.log(XL/XS) + W_MELT/(R*T)*(1-XS)**2
          - (DH_FO/R)*(1/Tm_Fo(P) - 1/T) - dG('Fo','a',ph,P,T)/(R*T))
    e2 = (2*np.log((1-XL)/(1-XS)) + W_MELT/(R*T)*XS**2
          - (DH_FA/R)*(1/TmFa - 1/T) - dG('Fa','a',ph,P,T)/(R*T))
    return e1, e2

# Back-calculate Tm_Fa at each anchor: solve (TmFa, XL) with XS = 0.75 at T = T_sol
aP  = np.array(sorted(PIERRU))
aTm = np.full(len(aP), np.nan)
for i, p in enumerate(aP):
    Tsol, ph = PIERRU[p]
    def eqs(v):
        TmFa, XL = v
        XL = np.clip(XL, 1e-9, 1-1e-9)
        return melt_eqs(X26, XL, Tsol, p, ph, TmFa)
    for g in ([1400,0.40],[1600,0.35],[1200,0.45],[1800,0.30]):
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-8:
            aTm[i] = sol[0]
            break

# alpha-Fa reference melting point: linear interpolation between anchors, end slopes extrapolated
def Tm_Fa(P):
    if P <= aP[0]:
        s = (aTm[1]-aTm[0])/(aP[1]-aP[0])
        return aTm[0] + s*(P-aP[0])
    if P >= aP[-1]:
        s = (aTm[-1]-aTm[-2])/(aP[-1]-aP[-2])
        return aTm[-1] + s*(P-aP[-1])
    return float(np.interp(P, aP, aTm))

# ── solvers ──────────────────────────────────────────────────────────────────
# phi+L loop at (P,T) -> (XS, XL)
def solve_melt(ph, P, T):
    TmFa = Tm_Fa(P)
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        return melt_eqs(XS, XL, T, P, ph, TmFa)
    for x0 in ([0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70],
               [0.5,0.3],[0.3,0.15]):
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                and 1e-6 < XL < XS < 1-1e-6):
            return float(XS), float(XL)
    return np.nan, np.nan

# f1+f2 solid-solid loop at (P,T) -> (X1, X2); f2 = high-pressure phase (more Fe-rich)
# e1: 2 ln(X2/X1) + (W2 (1-X2)^2 - W1 (1-X1)^2)/(RT) + dG_Fo/(RT) = 0
# e2: 2 ln((1-X2)/(1-X1)) + (W2 X2^2 - W1 X1^2)/(RT) + dG_Fa/(RT) = 0
def solve_ss(f1, f2, P, T):
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

# f1+f2+L three-phase line at fixed P -> (X1, X2, XL, T); order XL < X2 < X1
def solve_3ph_L(f1, f2, P):
    TmFa = Tm_Fa(P)
    def eqs(v):
        X1, X2, XL, T = v
        X1, X2, XL = np.clip([X1, X2, XL], 1e-9, 1-1e-9)
        e1, e2 = melt_eqs(X1, XL, T, P, f1, TmFa)
        e3, e4 = melt_eqs(X2, XL, T, P, f2, TmFa)
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

# alpha+beta+gamma univariant line at fixed T -> (Xa, Xb, Xg, P); order Xg < Xb < Xa
def solve_univariant(T):
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

# ── calibration summary ──────────────────────────────────────────────────────
print("="*70)
print("DV_eff calibration (cm3/mol)")
print("="*70)
for k, v in TRANS.items():
    print(f"  {k[0]} {k[1]}->{k[2]}:  DH={v[0]:>8.0f}  DS={v[1]:>6.1f}  DV_eff={v[2]:.3f}")

print("\nEnd-member transition pressures (GPa):")
for comp in ['Fo', 'Fa']:
    for (u, v) in [('a','b'), ('b','g'), ('a','g')]:
        row = f"  {comp} {u}->{v}: "
        for T in [1473.0, 1873.0]:
            p = fsolve(lambda P: dG(comp,u,v,float(P[0]),T), 10.0)[0]
            row += f"  {p:6.2f} @{T:.0f}K"
        print(row)

print("\nTm_Fa anchors (back-calculated with phase shift):")
for p, tm in zip(aP, aTm):
    print(f"  P={p:<5} phase={PIERRU[p][1]}  Tsol={PIERRU[p][0]:.0f}K  -> Tm_Fa(alpha ref) = {tm:.0f} K")

# ── Fig A: fixed-T P-X ───────────────────────────────────────────────────────
P_grid = np.linspace(3.0, 22.0, 240)
ss_pairs    = [('a','b'), ('b','g'), ('a','g')]
ss_colors   = {('a','b'):'tab:blue', ('b','g'):'tab:purple', ('a','g'):'tab:green'}
melt_colors = {'a':'tab:red', 'b':'darkorange', 'g':'brown'}
ph_name     = {'a':r'$\alpha$', 'b':r'$\beta$', 'g':r'$\gamma$'}

figA, axesA = plt.subplots(1, len(T_PX_list), figsize=(8*len(T_PX_list), 9))
axesA = np.atleast_1d(axesA)

print("\nFig A: alpha+beta+gamma univariant line")
for ax, T in zip(axesA, T_PX_list):
    # univariant line first, used to cut the loops
    Xa3, Xb3, Xg3, P3 = solve_univariant(T)
    print(f"  T={T:.0f} K: P={P3:.2f} GPa  Xa={Xa3:.3f}  Xb={Xb3:.3f}  Xg={Xg3:.3f}")

    # solid-solid loops; a+g stable only below the univariant line, a+b and b+g only above
    for pair in ss_pairs:
        X1 = np.full(len(P_grid), np.nan)
        X2 = np.full(len(P_grid), np.nan)
        for i, P in enumerate(P_grid):
            X1[i], X2[i] = solve_ss(pair[0], pair[1], P, T)
        if STABLE_ONLY and not np.isnan(P3):
            m = P_grid > P3 if pair == ('a','g') else P_grid < P3
            X1[m] = np.nan; X2[m] = np.nan
        c = ss_colors[pair]
        ax.plot(X1, P_grid, '-', color=c, lw=2, label=f'{ph_name[pair[0]]}+{ph_name[pair[1]]}')
        ax.plot(X2, P_grid, '-', color=c, lw=2)

    if not np.isnan(P3):
        ax.plot([Xg3, Xb3], [P3, P3], 'k-', lw=2.5, zorder=8)
        for x in [Xa3, Xb3, Xg3]:
            ax.plot(x, P3, 'ko', ms=7, zorder=9)

    # melting loops for the three polymorphs
    if SHOW_MELT:
        for ph in ['a', 'b', 'g']:
            XS = np.full(len(P_grid), np.nan)
            XL = np.full(len(P_grid), np.nan)
            for i, P in enumerate(P_grid):
                XS[i], XL[i] = solve_melt(ph, P, T)
            c = melt_colors[ph]
            ax.plot(XS, P_grid, '-',  color=c, lw=1.8, label=f'{ph_name[ph]}+L sol')
            ax.plot(XL, P_grid, '--', color=c, lw=1.8)

    ax.axvline(0.75, color='gray', ls='--', lw=1, alpha=0.6)   # Mars Mg# 0.75
    ax.set_xlim(0, 1)
    ax.set_ylim(P_grid[0], P_grid[-1])
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14)
    ax.set_ylabel('P (GPa)', fontsize=14)
    ax.set_title(f'T = {T:.0f} K', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
figA.tight_layout()

# ── Fig B: fixed-P T-X ───────────────────────────────────────────────────────
figB, axesB = plt.subplots(2, 2, figsize=(14, 13))
axesB = axesB.flatten()

print("\nFig B: solid+solid+L three-phase lines (X1, X2, XL, T)")
for ax, P in zip(axesB, P_TX_list):
    TmFo_P = Tm_Fo(P)
    T_grid = np.linspace(1200.0, TmFo_P*1.05, 260)

    ab = solve_3ph_L('a', 'b', P)
    bg = solve_3ph_L('b', 'g', P)
    ag = solve_3ph_L('a', 'g', P)
    T_abL, T_bgL, T_agL = ab[3], bg[3], ag[3]
    three_poly = (not np.isnan(T_abL)) and (not np.isnan(T_bgL)) and (T_abL > T_bgL)

    # stability windows: melt loop of phase ph drawn for Tlo <= T <= Thi (nan = not shown),
    # solid loop drawn for T <= ss_cut
    if not STABLE_ONLY:
        melt_win = {'a': (-np.inf, np.inf), 'b': (-np.inf, np.inf), 'g': (-np.inf, np.inf)}
        ss_cut   = {('a','b'): np.inf, ('b','g'): np.inf, ('a','g'): np.inf}
    elif three_poly:
        melt_win = {'a': (T_abL, np.inf), 'b': (T_bgL, T_abL), 'g': (-np.inf, T_bgL)}
        ss_cut   = {('a','b'): T_abL, ('b','g'): T_bgL, ('a','g'): -np.inf}
    elif not np.isnan(T_agL):
        melt_win = {'a': (T_agL, np.inf), 'b': (np.nan, np.nan), 'g': (-np.inf, T_agL)}
        ss_cut   = {('a','b'): -np.inf, ('b','g'): -np.inf, ('a','g'): T_agL}
    else:
        melt_win = {'a': (-np.inf, np.inf), 'b': (-np.inf, np.inf), 'g': (-np.inf, np.inf)}
        ss_cut   = {('a','b'): np.inf, ('b','g'): np.inf, ('a','g'): np.inf}

    # solid-solid loops (near-vertical bands, cut below the three-phase line)
    for pair in ss_pairs:
        X1 = np.full(len(T_grid), np.nan)
        X2 = np.full(len(T_grid), np.nan)
        for i, T in enumerate(T_grid):
            if T > ss_cut[pair]:
                continue
            X1[i], X2[i] = solve_ss(pair[0], pair[1], P, T)
        if np.all(np.isnan(X1)):
            continue
        c = ss_colors[pair]
        ax.plot(X1, T_grid, '-', color=c, lw=2, label=f'{ph_name[pair[0]]}+{ph_name[pair[1]]}')
        ax.plot(X2, T_grid, '-', color=c, lw=2)

    # melting loops within each stability window
    for ph in ['a', 'b', 'g']:
        Tlo, Thi = melt_win[ph]
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
        ax.plot(XS, T_grid, '-',  color=c, lw=1.8, label=f'{ph_name[ph]}+L')
        ax.plot(XL, T_grid, '--', color=c, lw=1.8)

    # horizontal three-phase lines (stable set only: ab, bg if three_poly, else ag)
    for tri, name in [(ab, 'abL'), (bg, 'bgL'), (ag, 'agL')]:
        X1t, X2t, XLt, Tt = tri
        if np.isnan(Tt):
            continue
        if STABLE_ONLY and three_poly and name == 'agL':
            continue
        if STABLE_ONLY and (not three_poly) and name != 'agL':
            continue
        print(f"  P={P:5.1f} GPa {name}: X1={X1t:.3f} X2={X2t:.3f} XL={XLt:.3f} T={Tt:.0f} K")
        ax.plot([XLt, X1t], [Tt, Tt], 'k-', lw=2.5, zorder=8)
        for x in [X1t, X2t, XLt]:
            ax.plot(x, Tt, 'ko', ms=6, zorder=9)

    # Pierru anchor if the pressure matches
    for Pp, (Tsol, phx) in PIERRU.items():
        if abs(Pp - P) < 0.5:
            ax.scatter([X26], [Tsol], marker='s', s=220, color='limegreen', edgecolors='darkgreen',
                       zorder=11, lw=2, label='Pierru T$_{sol}$ (Table 3)')

    ax.plot(1.0, TmFo_P, 'b^', ms=10, zorder=6)
    ax.axvline(0.75, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(1200, TmFo_P*1.05)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14)
    ax.set_ylabel('Temperature (K)', fontsize=14)
    ax.set_title(f'P = {P} GPa', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
figB.tight_layout()
plt.show()
