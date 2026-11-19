import os
import types
import numpy as np
from scipy.optimize import brentq, fsolve

# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '26fall_main_mcmc.py')
# Chanyshev 2026 BT614: wadsleyite + ahrensite at 12.88 GPa, 1958 K
P_BT614, T_BT614, XB_OBS, XG_OBS = 12.88, 1958.0, 0.66, 0.56
# Pierru 2026 19 GPa runs: name, nominal T (K), Rw Mg#, melt Mg#
P19  = 19.0
RUNS = [('S8293', 2303.0, 0.822, 0.572), ('S8295', 2323.0, 0.809, 0.550)]
P_LIST = [19.0, 21.0, 23.0]
GUESS_BG = [[0.70, 0.55], [0.80, 0.65], [0.60, 0.45], [0.90, 0.80], [0.50, 0.35]]

# ── load the phase-diagram block (baseline a60) ───────────────────────────────
src = open(FORWARD_FILE).read()
i0 = src.index('# ── physics: solidus')
i1 = src.index('# ── composition')
m = types.ModuleType('pd')
m.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
exec(src[i0:i1], m.__dict__)
R = m._R_GAS

# ── part 1: beta-gamma solid-solid equilibrium vs Chanyshev BT614 ─────────────
#   mu_Fo:  dG_Fo(b->g) + 2RT ln(Xg/Xb)         + Wg(1-Xg)^2 - Wb(1-Xb)^2 = 0
#   mu_Fa:  dG_Fa(b->g) + 2RT ln((1-Xg)/(1-Xb)) + Wg Xg^2    - Wb Xb^2    = 0
#   dG(b->g) = dG(a->g) - dG(a->b)
print(f"\npart 1: beta + gamma at {P_BT614} GPa, {T_BT614:.0f} K  (obs X_beta {XB_OBS}, X_gamma {XG_OBS})")
for label, Te in [('as in MCMC, raw T = T - shift', T_BT614 - m._DELTA_SHIFT),
                  ('physical T', T_BT614)]:
    eqs = lambda v: [
        m._dG_Fo_ag(P_BT614, Te) - m._dG_Fo_ab(P_BT614, Te) + 2*R*Te*np.log(v[1]/v[0])
        + m._W_GAMMA*(1-v[1])**2 - m._W_BETA*(1-v[0])**2,
        m._dG_Fa_ag(P_BT614, Te) - m._dG_Fa_ab(P_BT614, Te) + 2*R*Te*np.log((1-v[1])/(1-v[0]))
        + m._W_GAMMA*v[1]**2 - m._W_BETA*v[0]**2]
    found = False
    for g in GUESS_BG:
        s, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier == 1 and np.max(np.abs(info['fvec'])) < 1e-6 and 0 < s[1] < s[0] < 1:
            print(f"  {label:32s} (T_eval {Te:.0f} K): X_beta {s[0]:.3f}  X_gamma {s[1]:.3f}")
            found = True
            break
    if not found:
        print(f"  {label:32s} (T_eval {Te:.0f} K): no beta+gamma loop at this P-T")

# ── part 2: shift applied to melting only (solid-solid dG at physical T) ──────
#   loops are evaluated at raw T = T_phys - shift, so T_phys = raw + shift
f_Fo_ab, f_Fa_ab, f_Fo_ag, f_Fa_ag = m._dG_Fo_ab, m._dG_Fa_ab, m._dG_Fo_ag, m._dG_Fa_ag
m._dG_Fo_ab = lambda P, T: f_Fo_ab(P, T + m._DELTA_SHIFT)
m._dG_Fa_ab = lambda P, T: f_Fa_ab(P, T + m._DELTA_SHIFT)
m._dG_Fo_ag = lambda P, T: f_Fo_ag(P, T + m._DELTA_SHIFT)
m._dG_Fa_ag = lambda P, T: f_Fa_ag(P, T + m._DELTA_SHIFT)

# recalibrate at the 17.7 GPa tie-line; the shift now enters the solid dG -> fixed-point iteration
print("\npart 2: recalibration with solid-solid dG at physical T")
for it in range(8):
    m._SCAN_CACHE.clear()
    m._fit_fa_slope()
    new_shift, raw_sol, xl_tie = m._compute_shift()
    print(f"  iter {it+1}: Fa slope {m._FA_SLOPE:.2f} K/GPa  shift {new_shift:.1f} K  X_L(tie) {xl_tie:.3f}")
    done = abs(new_shift - m._DELTA_SHIFT) < 0.5
    m._DELTA_SHIFT = new_shift
    if done:
        break
m._SCAN_CACHE.clear()

for P in P_LIST:
    tp = m._three_phase_bg(P, m._Tm_Fo(P), m._Tm_Fa(P))
    if tp is None:
        print(f"  {P:.1f} GPa: no beta+gamma+L invariant (gamma only)")
    else:
        print(f"  {P:.1f} GPa: T3 {tp['T'] + m._DELTA_SHIFT:.1f} K  X_L3 {tp['XL']:.3f}  "
              f"X_gamma {tp['Xg']:.3f}  X_beta {tp['Xb']:.3f}")

print(f"\n19 GPa tie-line test (shift on melting only):")
TmFo, TmFa = m._Tm_Fo(P19), m._Tm_Fa(P19)
Tg, XSg, XLg = m._scan(m._gamma_XS_XL, P19, TmFo, TmFa)
og = np.argsort(XSg)
for name, T_obs, XS_obs, XL_obs in RUNS:
    T_mod  = np.interp(XS_obs, XSg[og], Tg[og])
    XL_mod = np.interp(XS_obs, XSg[og], XLg[og])
    KD_obs = ((1-XS_obs)/XS_obs) / ((1-XL_obs)/XL_obs)
    KD_mod = ((1-XS_obs)/XS_obs) / ((1-XL_mod)/XL_mod)
    r = m.bml_phase_diagram(T_obs, P19, XS_obs)
    print(f"  {name}: gamma T {T_mod:.1f} vs obs {T_obs:.0f} K (dT {T_mod - T_obs:+.1f})  "
          f"X_L {XL_mod:.3f} vs {XL_obs:.3f}  K_D {KD_mod:.3f} vs {KD_obs:.3f}  "
          f"model branch at X={XS_obs}: {r['phase'] if r else 'fail'}")

r = m.bml_phase_diagram(2403.0, P19, 0.75)
print(f"  H6050 (2403 K, bulk 0.75): branch {r['phase']}  T_sol {r['T_solidus']:.1f}  "
      f"T_liq {r['T_liquidus']:.1f}  melt fraction {r['f_liquid']:.2f} (obs 0.70-0.74)")