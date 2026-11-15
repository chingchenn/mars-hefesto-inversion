#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:33:27 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fo-Fa 多溫度 P-X loop 疊圖 + Chanyshev et al. (2026, JGR) Table 1 實驗點
模型：Akaogi 1989 參數（同 55c），T 改為參數
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

R = 8.314

PAR = {('a','b'): {'Fo': (27130.0,  -9.0, -2.805), 'Fa': ( 9620.0, -10.9, -3.083)},
       ('b','g'): {'Fo': ( 9080.0,  -7.3, -1.072), 'Fa': (-5790.0,  -3.1, -1.002)}}

W = {'a': lambda P: 12600.0 + 260.0*P,
     'b': lambda P: 7800.0,
     'g': lambda P: 7800.0}

def dG(i, phi, psi, P, T):
    if (phi, psi) in PAR:
        H, S, V = PAR[(phi, psi)][i]
        return H - T*S + V*1000.0*P
    return dG(i, 'a', 'b', P, T) + dG(i, 'b', 'g', P, T)

def eqs(X, phi, psi, P, T):
    Xp, Xq = X
    Wp, Wq = W[phi](P), W[psi](P)
    eFo = 2*np.log(Xq/Xp)         + (Wq*(1-Xq)**2 - Wp*(1-Xp)**2)/(R*T) + dG('Fo',phi,psi,P,T)/(R*T)
    eFa = 2*np.log((1-Xq)/(1-Xp)) + (Wq*Xq**2     - Wp*Xp**2    )/(R*T) + dG('Fa',phi,psi,P,T)/(R*T)
    return [eFo, eFa]

def P_closure(i, phi, psi, T):
    A = dG(i, phi, psi, 0.0, T)
    return -A / (dG(i, phi, psi, 1.0, T) - A)

def scan(phi, psi, end, P_stop, T, n=300):
    i  = 'Fo' if end == 'Mg' else 'Fa'
    Pc = P_closure(i, phi, psi, T)
    Pg = np.linspace(Pc + np.sign(P_stop - Pc)*1e-3, P_stop, n)
    X  = np.array([0.999, 0.995]) if end == 'Mg' else np.array([0.005, 0.001])
    out = []
    for P in Pg:
        s, info, ok, _ = fsolve(eqs, X, args=(phi, psi, P, T), full_output=True)
        good = ok == 1 and max(abs(info['fvec'])) < 1e-8 and 0 < s[1] < s[0] < 1
        out.append(s if good else [np.nan, np.nan])
        if good:
            X = s
    return Pg, np.array(out)

def univariant(T):
    def f(v):
        Xa, Xb, Xg, P = v
        return eqs([Xa, Xb], 'a', 'b', P, T) + eqs([Xb, Xg], 'b', 'g', P, T)
    return fsolve(f, [0.87, 0.74, 0.55, 13.9])

# ---- Chanyshev et al. (2026) Table 1（Mg# = X_Fo）----
# (T_K, P_GPa, X_olivine, X_ahrensite)；BT614 為 wadsleyite+ahrensite
DATA_OL_AHR = {
    1530: [(1530, 8.87, 0.62, 0.29), (1534, 10.70, 0.79, 0.50),
           (1537, 11.68, 0.86, 0.64), (1529, 8.27, 0.47, 0.19)],
    1950: [(1956, 11.98, 0.73, 0.46), (1951, 10.61, 0.64, 0.36),
           (1948, 9.54, 0.54, 0.32)],
}
DATA_WD_AHR = (1958, 12.88, 0.66, 0.56)   # BT614

# ---- 疊圖 ----
COL = {1530: 'tab:blue', 1740: 'tab:orange', 1950: 'tab:red'}
fig, ax = plt.subplots(figsize=(8, 8))

for Tk, c in COL.items():
    Xa, Xb, Xg, P3 = univariant(Tk)
    P_ag, ag = scan('a', 'g', 'Fa', P3, Tk)          # 主角：alpha+gamma loop
    P_ab, ab = scan('a', 'b', 'Mg', P3, Tk)          # beta 楔形（細線）
    P_bg, bg = scan('b', 'g', 'Mg', P3, Tk)
    ax.plot(ag[:, 0], P_ag, '-',  color=c, lw=2.2, label=f'{Tk} K')
    ax.plot(ag[:, 1], P_ag, '-',  color=c, lw=2.2)
    for L, Pv in [(ab, P_ab), (bg, P_bg)]:
        ax.plot(L[:, 0], Pv, '-', color=c, lw=0.9, alpha=0.6)
        ax.plot(L[:, 1], Pv, '-', color=c, lw=0.9, alpha=0.6)
    ax.plot([Xg, Xa], [P3, P3], '-', color=c, lw=1.2, alpha=0.8)

for Tk, mk_c in [(1530, 'tab:blue'), (1950, 'tab:red')]:
    pts = np.array([(p, xo, xa) for _, p, xo, xa in DATA_OL_AHR[Tk]])
    ax.plot(pts[:, 1], pts[:, 0], 'D', color=mk_c, ms=9, mec='k', zorder=6,
            label=f'olivine (exp. {Tk} K)')
    ax.plot(pts[:, 2], pts[:, 0], 'o', color=mk_c, ms=9, mec='k', zorder=6,
            label=f'ahrensite (exp. {Tk} K)')
_, p, xw, xa = DATA_WD_AHR
ax.plot(xw, p, 's', color='tab:green', ms=10, mec='k', zorder=6, label='wadsleyite (BT614)')
ax.plot(xa, p, 'o', color='tab:green', ms=10, mec='k', zorder=6)

ax.set_xlim(0, 1); ax.set_ylim(5, 20)
ax.set_xlabel('Mg#  ($X_{Fo}$)', fontsize=14)
ax.set_ylabel('P (GPa)', fontsize=14)
ax.set_title('Akaogi-1989 model vs Chanyshev et al. (2026) data', fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=10, loc='upper left', ncol=2)
ax.grid(alpha=0.3)
fig.savefig('56_chanyshev_overlay.png', dpi=150, bbox_inches='tight')

for Tk in COL:
    Xa, Xb, Xg, P3 = univariant(Tk)
    print(f"T={Tk} K  univariant P={P3:.2f} GPa  Xa={Xa:.3f} Xb={Xb:.3f} Xg={Xg:.3f}")
print("saved")