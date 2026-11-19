# 55.PTX_3D.py
# Fo-Fa P-X-T 3D prism: isobaric T-X melting sections (solver core of 54.Mg#_T_abr.py)
# stacked along P, with the three-phase loci and the end-member melting curves.
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

W_MELT = 8000.0                       # solid-liquid W (Pierru Table 4 fit, version 48)
P3D_MIN, P3D_MAX, P3D_STEP = 6.0, 20.0, 1.0
nT = 140                              # T points per isobaric section
ELEV, AZIM = 12, -35                  # view angle (alternative view: 22, -60)

R   = 8.314
X26 = 0.75
DH_FO, DH_FA = 142000.0, 89000.0   # alpha end-member enthalpy of fusion (olivine, Stixrude + Hess)

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

# ── 3D P-X-T prism: isobaric T-X sections stacked along P ───────────────────
melt_colors = {'a':'tab:red', 'b':'darkorange', 'g':'brown'}
P_sections  = np.arange(P3D_MIN, P3D_MAX + 1e-9, P3D_STEP)

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
    three_poly = (not np.isnan(T_abL)) and (not np.isnan(T_bgL)) and (T_abL > T_bgL)

    # stability window (Tlo, Thi) of each phase+L loop; nan = phase not shown
    if three_poly:
        tri_locus['ab'].append((P, *ab))
        tri_locus['bg'].append((P, *bg))
        windows = {'a': (T_abL, np.inf), 'b': (T_bgL, T_abL), 'g': (-np.inf, T_bgL)}
    elif not np.isnan(T_agL):
        tri_locus['ag'].append((P, *ag))
        windows = {'a': (T_agL, np.inf), 'g': (-np.inf, T_agL), 'b': (np.nan, np.nan)}
    else:
        windows = {'a': (-np.inf, np.inf), 'b': (np.nan, np.nan), 'g': (np.nan, np.nan)}

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
        ax.plot(XS, np.full(nT, P), T_grid, '-',  color=c, lw=1.4)               # solidus
        ax.plot(XL, np.full(nT, P), T_grid, '--', color=c, lw=1.0, alpha=0.55)   # liquidus

    # three-phase horizontal line of each section
    for tri, ok in [(ab, three_poly), (bg, three_poly),
                    (ag, (not three_poly) and (not np.isnan(T_agL)))]:
        X1t, X2t, XLt, Tt = tri
        if ok and not np.isnan(Tt):
            ax.plot([XLt, X1t], [P, P], [Tt, Tt], 'k-', lw=1.6, alpha=0.8)

# three-phase loci along P (black)
print("Three-phase loci, high-P end (X1, P, T):")
for key in ['ab', 'bg', 'ag']:
    arr = np.array(tri_locus[key])
    if len(arr) < 2:
        continue
    Pv, X1v, X2v, XLv, Tv = arr.T
    for Xv, lw in [(X1v, 2.5), (X2v, 1.5), (XLv, 2.5)]:
        ax.plot(Xv, Pv, Tv, '-', color='k', lw=lw)
    print(f"  {key}L: X1={X1v[-1]:.3f}  P={Pv[-1]:.1f} GPa  T={Tv[-1]:.0f} K  ({len(Pv)} sections)")

# Pierru anchors
for Pp, (Tsol, phx) in PIERRU.items():
    if P3D_MIN <= Pp <= P3D_MAX:
        ax.scatter([X26], [Pp], [Tsol], marker='s', s=140, color='limegreen',
                   edgecolors='darkgreen', lw=1.5, zorder=10)

# end-member melting curves (the two prism edges): Fo at X=1 (blue), Fa at X=0 (red)
Pv = np.linspace(P3D_MIN, P3D_MAX, 60)
ax.plot(np.ones_like(Pv), Pv, Tm_Fo(Pv), 'b-', lw=2.5)
ax.plot(np.zeros_like(Pv), Pv, [Tm_Fa(p) for p in Pv], 'r-', lw=2.5)

ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=14, labelpad=12)
ax.set_ylabel('P (GPa)', fontsize=14, labelpad=12)
ax.set_zlabel('T (K)', fontsize=14, labelpad=12)
ax.set_xlim(0, 1)
ax.tick_params(labelsize=14)
ax.view_init(elev=ELEV, azim=AZIM)
plt.show()
