# 48.melting_Mg#_data.py
# Fo-Fa melting loop, "Mars effective": fit W(P) and Tm_Fa(P) to Pierru 2026 Table 4 tie lines.
# Only the P > 13 GPa points (Wad, Rw) are used in the fit, because they are self-consistent in
# the pure Fo-Fa framework (eq1 and eq2 satisfied together). The olivine points (8, 9.6 GPa)
# are shown for reference only.
#   eq1: 2 ln(XL/XS)         + W/(RT) (1-XS)^2 = (DH_Fo/R)(1/Tm_Fo - 1/T)
#   eq2: 2 ln((1-XL)/(1-XS)) + W/(RT) XS^2     = (DH_Fa/R)(1/Tm_Fa - 1/T)
# Fit: W = WH + P WV. At each high-P point eq1 and eq2 are solved together for W and Tm_Fa.
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
R = 8.314

# Pierru 2026 Table 4: (P GPa, T K, mineral, XS, XL)
pierru_all = [
    (8.0,  1890, 'Ol (cold)', 0.80, 0.53),
    (8.0,  1890, 'Ol (warm)', 0.85, 0.58),
    (9.6,  1960, 'Ol',        0.91, 0.67),
    (14.1, 2090, 'Wad',       0.80, 0.51),   # used in the fit
    (17.7, 2200, 'Rw',        0.74, 0.50),   # used in the fit
]
pierru_fit = [p for p in pierru_all if p[0] > 13]
target_P   = [8.0, 9.6, 14.1, 17.7]
phase_name = {8.0: 'Olivine (ref only)', 9.6: 'Olivine (ref only)',
              14.1: 'Wadsleyite (fit)',  17.7: 'Ringwoodite (fit)'}

# enthalpies of fusion (DH_Fo, DH_Fa), J/mol: P < 14 | 14 <= P < 20 | P >= 20 GPa
def get_DH(P):
    if P < 14.0:   return 142000.0, 89000.0
    elif P < 20.0: return 116000.0, 73000.0
    else:          return 105000.0, 74000.0

# Fo melting point: Tm_Fo = 2163 (P/65.67 + 1)^0.7809
def Tm_Fo(P):
    return 2163.0 * (P/65.67 + 1.0)**0.7809

# ── Step 1: at each high-P point solve eq1 = eq2 = 0 for (W, Tm_Fa) ──────────
# 5 starting guesses; keep the converged one (residual < 1e-8, Tm_Fa > 500 K) with the smallest residual
print("="*65)
print("Step 1: solve W and Tm_Fa at each high-P point")
print("="*65)
print(f"{'P':<6} {'Mineral':<10} {'Tm_Fa(K)':<12} {'W(kJ/mol)':<12} {'eq1 res':<12} {'eq2 res'}")
print("-"*65)
P_solved, W_solved, TmFa_solved = [], [], []
for (P, T, mineral, XS, XL) in pierru_fit:
    DH_Fo, DH_Fa = get_DH(P)
    TmFo = Tm_Fo(P)
    def eqs(v):
        W, TmFa = v
        e1 = 2*np.log(XL/XS) + W/(R*T)*(1-XS)**2 - (DH_Fo/R)*(1/TmFo - 1/T)
        e2 = 2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2 - (DH_Fa/R)*(1/TmFa - 1/T)
        return [e1, e2]
    best = None
    for g in [[0, 1500], [-50000, 1500], [50000, 1500], [-100000, 2000], [100000, 2000]]:
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        resid = np.max(np.abs(info['fvec']))
        if ier == 1 and resid < 1e-8 and sol[1] > 500:
            if best is None or resid < best[2]:
                best = (sol[0], sol[1], resid)
    if best:
        W_sol, TmFa_sol, _ = best
        e1, e2 = eqs([W_sol, TmFa_sol])
        print(f"{P:<6.1f} {mineral:<10} {TmFa_sol:<12.0f} {W_sol/1000:<12.2f} {e1:.2e}    {e2:.2e}")
        P_solved.append(P); W_solved.append(W_sol); TmFa_solved.append(TmFa_sol)
P_arr, W_arr, TmFa_arr = np.array(P_solved), np.array(W_solved), np.array(TmFa_solved)

# ── Step 2: linear fit W(P) = WH + P WV ──────────────────────────────────────
WV_fit, WH_fit = np.polyfit(P_arr, W_arr, 1)
print()
print("="*65)
print("Step 2: W(P) = WH + P*WV fit")
print("="*65)
print(f"  WH = {WH_fit/1000:.2f} kJ/mol")
print(f"  WV = {WV_fit/1000:.3f} kJ/mol/GPa")
for P in target_P:
    print(f"  W({P:.1f} GPa) = {(WH_fit + WV_fit*P)/1000:.2f} kJ/mol")

# ── Step 3: Simon-Glatzel fit of Tm_Fa(P): T = T0 (P/a + 1)^(1/c) ─────────────
# high-P Tm_Fa from Step 1 plus the 1 atm anchor 1478 K (Ohtani 1979)
P_sg = np.concatenate([[0.0001], P_arr])
T_sg = np.concatenate([[1478.0], TmFa_arr])
popt, _ = curve_fit(lambda P, T0, a, c: T0 * (P/a + 1.0)**(1.0/c), P_sg, T_sg, p0=[1478, 10, 1.5], maxfev=10000)
print(f"\nTm_Fa Simon-Glatzel: T0={popt[0]:.1f}, a={popt[1]:.3f}, c={popt[2]:.4f}")

# ── phase loop on a T grid from 0.97 Tm_Fa to 1.02 Tm_Fo ─────────────────────
# T >= Tm_Fo: pure Fo (X = 1); T <= Tm_Fa: pure Fa (X = 0); otherwise solve eq1, eq2 for (XS, XL)
# with 5 guesses, accepting residual < 1e-6 and 0 < XL < XS < 1. Unsolved points are dropped.
def build_curves(P, TmFa, W):
    DH_Fo, DH_Fa = get_DH(P)
    TmFo = Tm_Fo(P)
    T_arr  = np.linspace(TmFa*0.97, TmFo*1.02, 400)
    XS_arr = np.full(len(T_arr), np.nan)
    XL_arr = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        if T >= TmFo:
            XS_arr[i] = 1.0; XL_arr[i] = 1.0
        elif T <= TmFa:
            XS_arr[i] = 0.0; XL_arr[i] = 0.0
        else:
            def eqs(v):
                XS, XL = np.clip(v, 1e-9, 1-1e-9)
                e1 = 2*np.log(XL/XS) + W/(R*T)*(1-XS)**2 - (DH_Fo/R)*(1/TmFo - 1/T)
                e2 = 2*np.log((1-XL)/(1-XS)) + W/(R*T)*XS**2 - (DH_Fa/R)*(1/TmFa - 1/T)
                return [e1, e2]
            for x0 in [[0.85,0.65], [0.80,0.55], [0.70,0.45], [0.90,0.70], [0.60,0.40]]:
                sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
                if (ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6
                        and 0 < sol[0] < 1 and 0 < sol[1] < 1 and sol[0] > sol[1]):
                    XS_arr[i], XL_arr[i] = sol
                    break
    valid = ~np.isnan(XS_arr)
    return T_arr[valid], XS_arr[valid], XL_arr[valid]

# ── plot: one panel per pressure ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, P in zip(axes.flatten(), target_P):
    TmFo   = Tm_Fo(P)
    TmFa   = popt[0] * (P/popt[1] + 1.0)**(1.0/popt[2])
    W      = WH_fit + WV_fit*P
    DH_Fo, DH_Fa = get_DH(P)
    is_fit = P > 13
    col_main = 'steelblue' if is_fit else 'gray'
    print(f"\nP = {P} GPa  {phase_name[P]}:  Tm_Fo = {TmFo:.0f} K  Tm_Fa = {TmFa:.0f} K  "
          f"W = {W/1000:.1f} kJ/mol  DH_Fo = {DH_Fo/1000:.0f}  DH_Fa = {DH_Fa/1000:.0f} kJ/mol")

    # loop with the fitted W(P)
    T_v, XS_v, XL_v = build_curves(P, TmFa, W)
    ax.plot(XS_v, T_v, '-',  color=col_main, lw=2.5 if is_fit else 1.5,
            label=f'W(P)={W/1000:.1f} kJ/mol (fit)  $T_m^{{Fa}}$={TmFa:.0f}K')
    ax.plot(XL_v, T_v, '--', color=col_main, lw=2.5 if is_fit else 1.5)
    ax.fill_betweenx(T_v, XS_v, XL_v, alpha=0.10, color=col_main)

    # W = 0 reference (fit points only): eq2 with W = 0 gives Tm_Fa analytically,
    #   1/Tm_Fa = 1/T + (R/DH_Fa) 2 ln((1-XL)/(1-XS)),  averaged over the points within 0.5 GPa
    if is_fit:
        ref_pts = [(P2, T2, XS2, XL2) for (P2, T2, _, XS2, XL2) in pierru_fit if abs(P2-P) < 0.5]
        if ref_pts:
            TmFa0 = np.mean([1.0 / (1.0/t + (R/get_DH(p)[1])*2*np.log((1-xl)/(1-xs)))
                             for p, t, xs, xl in ref_pts])
            T_v0, XS_v0, XL_v0 = build_curves(P, TmFa0, 0)
            ax.plot(XS_v0, T_v0, '-',  color='tomato', lw=1.5, alpha=0.7, label=f'W=0  $T_m^{{Fa}}$={TmFa0:.0f}K')
            ax.plot(XL_v0, T_v0, '--', color='tomato', lw=1.5, alpha=0.7)

    # end-member melting points
    ax.plot(1.0, TmFo, 'b^', ms=10, zorder=6)
    ax.plot(0.0, TmFa, 'v',  color=col_main, ms=10, zorder=6)

    # Pierru Table 4 tie lines (green = fit, orange = reference only)
    for (Pp, Tp, mineral, XSp, XLp) in pierru_all:
        if abs(Pp - P) < 1.0:
            mcol = 'limegreen' if P > 13 else 'orange'
            ecol = 'darkgreen' if P > 13 else 'darkorange'
            ax.scatter([XSp], [Tp], marker='o', s=200, color=mcol, edgecolors=ecol, zorder=10, lw=2)
            ax.scatter([XLp], [Tp], marker='D', s=200, color=mcol, edgecolors=ecol, zorder=10, lw=2)
            ax.plot([XSp, XLp], [Tp, Tp], color=ecol, lw=2, zorder=9)
            print(f"  tie line {mineral}: T = {Tp} K  XS = {XSp} -> XL = {XLp}")

    # Mars Mg# 0.75 (orange dashed) and 0.89 (grey dotted)
    ax.axvline(0.75, color='darkorange', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(0.89, color='gray',       ls=':',  lw=1.0, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1000, 3200)
    ax.set_xlabel('Mg# ($X_{Fo}$)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'P = {P} GPa  |  {phase_name[P]}', fontsize=12)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
