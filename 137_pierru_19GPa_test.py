import os
import types
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '26fall_main_mcmc.py')
P19     = 19.0       # GPa
MG_BULK = 0.75       # Pierru starting Mg#
T_ERR   = 50.0       # K, 19 GPa temperatures are nominal (no gradient correction)
CASES   = [0.0, -5000.0]   # change of Akaogi dH(Fo, alpha->gamma), J/mol; 0 = a60 baseline
# Pierru 2026 SI, 19 GPa ex situ runs, not used in the calibration
#   name, nominal T (K), Rw Mg#, melt Mg#
RUNS = [('S8293', 2303.0, 0.822, 0.572),
        ('S8295', 2323.0, 0.809, 0.550)]
H6050_T, H6050_XL = 2403.0, 0.682      # garnet-only residue, melt fraction 70-74%

src = open(FORWARD_FILE).read()
i0 = src.index('# ── physics: solidus')
i1 = src.index('# ── composition')

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for k, dH in enumerate(CASES):
    # fresh copy of the phase-diagram block (a60 calibration done at load)
    m = types.ModuleType('pd')
    m.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
    exec(src[i0:i1], m.__dict__)
    if dH != 0.0:
        m._DH_Fo_ag += dH
        m._SCAN_CACHE.clear()
        m._fit_fa_slope()
        m._DELTA_SHIFT, m._RAW_SOL_REF, m._XL_TIE_MODEL = m._compute_shift()
        m._SCAN_CACHE.clear()
    title = 'a60 baseline' if dH == 0.0 else f'Akaogi dH(Fo, a->g) {dH/1000:+.0f} kJ'
    print(f"\n==== {title}: Fa slope {m._FA_SLOPE:.2f} K/GPa  shift {m._DELTA_SHIFT:.1f} K")

    # invariant beta + gamma + L; T3 carries the same shift as bml_phase_diagram
    TmFo = m._Tm_Fo(P19)
    TmFa = m._Tm_Fa(P19)
    tp = m._three_phase_bg(P19, TmFo, TmFa)
    T3 = tp['T'] + m._DELTA_SHIFT
    print(f"{P19} GPa invariant: T3 {T3:.1f} K  X_L3 {tp['XL']:.3f}  "
          f"X_gamma {tp['Xg']:.3f}  X_beta {tp['Xb']:.3f}")

    Tg, XSg, XLg = m._scan(m._gamma_XS_XL, P19, TmFo, TmFa)
    Tb, XSb, XLb = m._scan(m._beta_XS_XL,  P19, TmFo, TmFa)
    og = np.argsort(XSg)
    ob = np.argsort(XSb)

    # tie-line test: at X_S = observed Rw Mg#, read model T and X_L
    #   K_D = [(1-X_S)/X_S] / [(1-X_L)/X_L]
    for name, T_obs, XS_obs, XL_obs in RUNS:
        KD_obs = ((1 - XS_obs) / XS_obs) / ((1 - XL_obs) / XL_obs)
        print(f"  {name}: obs T {T_obs:.0f} K  X_S {XS_obs:.3f}  X_L {XL_obs:.3f}  K_D {KD_obs:.3f}")
        for lname, T_l, XS_l, XL_l, o in [('gamma', Tg, XSg, XLg, og), ('beta', Tb, XSb, XLb, ob)]:
            T_mod  = np.interp(XS_obs, XS_l[o], T_l[o])
            XL_mod = np.interp(XS_obs, XS_l[o], XL_l[o])
            KD_mod = ((1 - XS_obs) / XS_obs) / ((1 - XL_mod) / XL_mod)
            print(f"    {lname:5s}: T {T_mod:7.1f} K (dT {T_mod - T_obs:+6.1f})  X_L {XL_mod:.3f}  "
                  f"K_D {KD_mod:.3f}  {'above T3' if T_mod > T3 else 'below T3'}")

    # H6050: binary model at bulk Mg# 0.75
    for T in [H6050_T - T_ERR, H6050_T, H6050_T + T_ERR]:
        r = m.bml_phase_diagram(T, P19, MG_BULK)
        print(f"  H6050 T {T:.0f} K: branch {r['phase']}  T_sol {r['T_solidus']:.1f}  "
              f"T_liq {r['T_liquidus']:.1f}  melt fraction {r['f_liquid']:.2f} (obs 0.70-0.74)")

    # panel
    a = ax[k]
    a.plot(XSg, Tg, '-',  color='C0', lw=1.8, label=r'$\gamma$ solidus')
    a.plot(XLg, Tg, '-',  color='C3', lw=1.8, label=r'$\gamma$ liquidus')
    a.plot(XSb, Tb, '--', color='C0', lw=1.4, label=r'$\beta$ solidus')
    a.plot(XLb, Tb, '--', color='C3', lw=1.4, label=r'$\beta$ liquidus')
    a.axhline(T3, color='k', ls=':', lw=1, label=r'$T_3$ ($\beta+\gamma$+L)')
    a.axvline(tp['Xg'], color='C2', ls=':', lw=1)
    a.axvline(tp['Xb'], color='C4', ls=':', lw=1)
    for name, T_obs, XS_obs, XL_obs in RUNS:
        a.errorbar(XS_obs, T_obs, yerr=T_ERR, fmt='s', color='C0', ms=7, capsize=3)
        a.errorbar(XL_obs, T_obs, yerr=T_ERR, fmt='o', color='C3', mfc='none', ms=7, capsize=3)
        a.plot([XL_obs, XS_obs], [T_obs, T_obs], color='0.5', lw=0.8)
        a.annotate(name, (XS_obs, T_obs), textcoords='offset points', xytext=(6, 4), fontsize=11)
    a.errorbar(H6050_XL, H6050_T, yerr=T_ERR, fmt='^', color='C3', mfc='none', ms=8, capsize=3)
    a.annotate('H6050 melt', (H6050_XL, H6050_T), textcoords='offset points', xytext=(6, 4), fontsize=11)
    a.set_xlim(0.3, 1.0)
    a.set_ylim(2000, 2700)
    a.set_xlabel('Mg#', fontsize=14)
    a.set_title(f'{P19} GPa, {title}', fontsize=14)
    a.tick_params(labelsize=14)

ax[0].set_ylabel('T (K)', fontsize=14)
ax[0].legend(loc='lower right', fontsize=10)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '137_pierru_19GPa_compare.png'), dpi=200)
print("\nsaved 137_pierru_19GPa_compare.png")