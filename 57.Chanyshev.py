# 57.Chanyshev.py
# Fo-Fa P-X loops at several temperatures, overlaid with the Chanyshev et al. (2026, JGR)
# Table 1 experiments. Model: Akaogi 1989 parameters (same as 55c), T as a parameter.
# Phases: a = olivine, b = wadsleyite, g = ringwoodite/ahrensite.
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
R = 8.314
COL = {1530: 'tab:blue', 1740: 'tab:orange', 1950: 'tab:red'}    # model temperatures (K)

# Akaogi 1989: dG(i, phi->psi) = H - T S + V P   (H in J/mol, S in J/mol/K, V in cm3/mol, P in GPa)
PAR = {('a', 'b'): {'Fo': (27130.0, -9.0, -2.805), 'Fa': ( 9620.0, -10.9, -3.083)},
       ('b', 'g'): {'Fo': ( 9080.0, -7.3, -1.072), 'Fa': (-5790.0,  -3.1, -1.002)}}
# regular-solution interaction parameter W (J/mol)
W = {'a': lambda P: 12600.0 + 260.0 * P,
     'b': lambda P: 7800.0,
     'g': lambda P: 7800.0}

# Chanyshev et al. (2026) Table 1 (Mg# = X_Fo): (T_K, P_GPa, X_olivine, X_ahrensite)
DATA_OL_AHR = {
    1530: [(1530, 8.87, 0.62, 0.29), (1534, 10.70, 0.79, 0.50),
           (1537, 11.68, 0.86, 0.64), (1529, 8.27, 0.47, 0.19)],
    1950: [(1956, 11.98, 0.73, 0.46), (1951, 10.61, 0.64, 0.36),
           (1948, 9.54, 0.54, 0.32)],
}
DATA_WD_AHR = (1958, 12.88, 0.66, 0.56)   # BT614: wadsleyite + ahrensite


# dG for a->g is dG(a->b) + dG(b->g)
def dG(i, phi, psi, P, T):
    if (phi, psi) in PAR:
        H, S, V = PAR[(phi, psi)][i]
        return H - T * S + V * 1000.0 * P
    return dG(i, 'a', 'b', P, T) + dG(i, 'b', 'g', P, T)


# Two-phase equilibrium between phi (X_p) and psi (X_q), X = X_Fo:
#   2 ln(Xq/Xp)         + [Wq (1-Xq)^2 - Wp (1-Xp)^2] / RT + dG_Fo / RT = 0
#   2 ln((1-Xq)/(1-Xp)) + [Wq Xq^2     - Wp Xp^2    ] / RT + dG_Fa / RT = 0
def eqs(X, phi, psi, P, T):
    Xp, Xq = X
    Wp, Wq = W[phi](P), W[psi](P)
    eFo = 2 * np.log(Xq / Xp) + (Wq * (1 - Xq)**2 - Wp * (1 - Xp)**2) / (R * T) + dG('Fo', phi, psi, P, T) / (R * T)
    eFa = 2 * np.log((1 - Xq) / (1 - Xp)) + (Wq * Xq**2 - Wp * Xp**2) / (R * T) + dG('Fa', phi, psi, P, T) / (R * T)
    return [eFo, eFa]


# Trace a loop from the end-member transition pressure Pc (dG_i(Pc) = 0, linear in P) to P_stop
def scan(phi, psi, end, P_stop, T, n=300):
    i = 'Fo' if end == 'Mg' else 'Fa'
    A = dG(i, phi, psi, 0.0, T)
    Pc = -A / (dG(i, phi, psi, 1.0, T) - A)
    Pg = np.linspace(Pc + np.sign(P_stop - Pc) * 1e-3, P_stop, n)
    X = np.array([0.999, 0.995]) if end == 'Mg' else np.array([0.005, 0.001])
    out = []
    for P in Pg:
        s, info, ok, _ = fsolve(eqs, X, args=(phi, psi, P, T), full_output=True)
        good = ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < s[1] < s[0] < 1
        out.append(s if good else [np.nan, np.nan])
        if good:
            X = s
    return Pg, np.array(out)


fig, ax = plt.subplots(figsize=(8, 8))
for Tk, c in COL.items():
    # univariant point a + b + g: solve both pairs of equations together for (Xa, Xb, Xg, P3)
    Xa, Xb, Xg, P3 = fsolve(lambda v: eqs([v[0], v[1]], 'a', 'b', v[3], Tk) + eqs([v[1], v[2]], 'b', 'g', v[3], Tk),
                            [0.87, 0.74, 0.55, 13.9])
    print(f"T={Tk} K  univariant P={P3:.2f} GPa  Xa={Xa:.3f} Xb={Xb:.3f} Xg={Xg:.3f}")

    P_ag, ag = scan('a', 'g', 'Fa', P3, Tk)          # main alpha + gamma loop
    P_ab, ab = scan('a', 'b', 'Mg', P3, Tk)          # beta wedge (thin lines)
    P_bg, bg = scan('b', 'g', 'Mg', P3, Tk)
    ax.plot(ag[:, 0], P_ag, '-', color=c, lw=2.2, label=f'{Tk} K')
    ax.plot(ag[:, 1], P_ag, '-', color=c, lw=2.2)
    for L, Pv in [(ab, P_ab), (bg, P_bg)]:
        ax.plot(L[:, 0], Pv, '-', color=c, lw=0.9, alpha=0.6)
        ax.plot(L[:, 1], Pv, '-', color=c, lw=0.9, alpha=0.6)
    ax.plot([Xg, Xa], [P3, P3], '-', color=c, lw=1.2, alpha=0.8)

for Tk, mk_c in [(1530, 'tab:blue'), (1950, 'tab:red')]:
    pts = np.array(DATA_OL_AHR[Tk])               # columns: T, P, X_olivine, X_ahrensite
    ax.plot(pts[:, 2], pts[:, 1], 'D', color=mk_c, ms=9, mec='k', zorder=6, label=f'olivine (exp. {Tk} K)')
    ax.plot(pts[:, 3], pts[:, 1], 'o', color=mk_c, ms=9, mec='k', zorder=6, label=f'ahrensite (exp. {Tk} K)')
_, p, xw, xa = DATA_WD_AHR
ax.plot(xw, p, 's', color='tab:green', ms=10, mec='k', zorder=6, label='wadsleyite (BT614)')
ax.plot(xa, p, 'o', color='tab:green', ms=10, mec='k', zorder=6)

ax.set_xlim(0, 1)
ax.set_ylim(5, 20)
ax.set_xlabel('Mg#  ($X_{Fo}$)', fontsize=14)
ax.set_ylabel('P (GPa)', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=10, loc='lower right', ncol=2)
ax.grid(alpha=0.3)
plt.show()
