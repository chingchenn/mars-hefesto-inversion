"""
134_phase_param_sensitivity.py

Sensitivity of the BML Fo-Fa phase diagram (26fall_main_mcmc.py, a60) to its
thermodynamic parameters at BML pressures.

Each case: perturb one parameter -> redo the a60 two-parameter calibration at the
Pierru 17.7 GPa tie-line (Fa slope from X_L = 0.50, rigid shift from T_sol = 2200 K)
-> evaluate T_solidus, T_liquidus, phase branch on a (P, Mg#) grid -> compare
with the baseline.

The phase-diagram block is cut out of the forward model (same markers as 132),
so the code is identical to what the MCMC uses. Needs only numpy/scipy.
"""
import types
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
import os
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
FORWARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '26fall_main_mcmc.py')
P_GRID  = [19.5, 21.0, 22.0, 23.0]                 # GPa, BML range
MG_GRID = np.round(np.arange(0.20, 0.801, 0.05), 2)  # Mg#_bulk_bml prior range
P_PLOT  = 21.0                                     # pressure shown in the figure
T_LOW   = 1000.0   # below every solidus: bml_phase_diagram then returns
                   # T_solidus / T_liquidus without solving the loop at T

SRC = open(FORWARD_FILE).read()
I0, I1 = SRC.index('# ── physics: solidus'), SRC.index('# ── composition')

def load_block():
    """fresh copy of the phase-diagram block (baseline parameters)"""
    m = types.ModuleType('pd')
    m.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
    exec(SRC[I0:I1], m.__dict__)
    return m

# ── perturbations (each acts on a fresh module) ───────────────────────────────
def p_dhfus_fo(m):
    m._DH_FO_FUS = 122000.0

def p_tm0_fo(m):
    f = m._Tm_Fo                                  # 2163 K is hard-coded inside
    m._Tm_Fo = lambda P: f(P) * 2174.0 / 2163.0

def p_wg(d):
    def f(m): m._W_GAMMA = 6200.0 + d
    return f

def p_wb(d):
    def f(m): m._W_BETA = 12900.0 + d
    return f

def p_fa_minus35(m):
    m._FA_T0 = m._FA_T0 - 35.0
    m._FA_ALPHA_T = tuple(t - 35.0 for t in m._FA_ALPHA_T)

def p_akaogi(d):
    def f(m): m._DH_Fo_ag = m._DH_Fo_ag + d       # used by _dG_Fo_ag
    return f

CASES = [
    ('baseline',            None),
    ('dHfus Fo 122 kJ',     p_dhfus_fo),
    ('Tm0 Fo 2174 K',       p_tm0_fo),
    ('W_gamma +1.7 kJ',     p_wg(+1700.0)),
    ('W_gamma -1.7 kJ',     p_wg(-1700.0)),
    ('W_beta +2.5 kJ',      p_wb(+2500.0)),
    ('W_beta -2.5 kJ',      p_wb(-2500.0)),
    ('Fa -35 K',            p_fa_minus35),
    ('Akaogi dH Fo ag +5',  p_akaogi(+5000.0)),
    ('Akaogi dH Fo ag -5',  p_akaogi(-5000.0)),
]

# ── calibration + evaluation ──────────────────────────────────────────────────
def recalibrate(m):
    """a60 calibration: Fa slope (X_L) then rigid shift (T_sol) at 17.7 GPa"""
    m._SCAN_CACHE.clear()
    m._fit_fa_slope()
    m._DELTA_SHIFT, m._RAW_SOL_REF, m._XL_TIE_MODEL = m._compute_shift()
    m._SCAN_CACHE.clear()          # _scan results depend on _DELTA_SHIFT

def evaluate(m):
    shape = (len(P_GRID), len(MG_GRID))
    Ts, Tl = np.full(shape, np.nan), np.full(shape, np.nan)
    ph = np.full(shape, 'fail', dtype=object)
    for i, P in enumerate(P_GRID):
        for j, X in enumerate(MG_GRID):
            r = m.bml_phase_diagram(T_LOW, P, X)
            if r is not None:
                Ts[i, j], Tl[i, j], ph[i, j] = r['T_solidus'], r['T_liquidus'], r['phase']
    return Ts, Tl, ph

results = {}
for label, patch in CASES:
    m = load_block()
    if patch is not None:
        patch(m)
    try:
        recalibrate(m)
    except ValueError as e:        # brentq bracket on the Fa slope
        print(f"{label:22s} calibration failed: {e}")
        continue
    Ts, Tl, ph = evaluate(m)
    results[label] = dict(slope=m._FA_SLOPE, shift=m._DELTA_SHIFT,
                          XL=m._XL_TIE_MODEL, Ts=Ts, Tl=Tl, ph=ph)
    print(f"{label:22s} Fa slope={m._FA_SLOPE:7.2f} K/GPa  "
          f"shift={m._DELTA_SHIFT:7.1f} K  X_L(tie)={m._XL_TIE_MODEL:.3f}")

# ── summary table ─────────────────────────────────────────────────────────────
base = results['baseline']
print(f"\nP = {P_GRID} GPa,  Mg# = {MG_GRID.min():.2f}-{MG_GRID.max():.2f}")
print(f"{'case':22s}{'max|dT_sol|':>13s}{'at (P, Mg#)':>16s}"
      f"{'max|dT_liq|':>13s}{'at (P, Mg#)':>16s}{'branch chg':>12s}{'fails':>7s}")

def argmax_loc(a):
    if not np.isfinite(a).any():
        return np.nan, '      -'
    k = np.unravel_index(np.nanargmax(a), a.shape)
    return a[k], f"({P_GRID[k[0]]:.1f}, {MG_GRID[k[1]]:.2f})"

for label, r in results.items():
    if label == 'baseline':
        continue
    dS = np.abs(r['Ts'] - base['Ts'])
    dL = np.abs(r['Tl'] - base['Tl'])
    vS, locS = argmax_loc(dS)
    vL, locL = argmax_loc(dL)
    n_chg  = int(np.sum(r['ph'] != base['ph']))
    n_fail = int(np.sum(r['ph'] == 'fail'))
    print(f"{label:22s}{vS:13.1f}{locS:>16s}{vL:13.1f}{locL:>16s}{n_chg:12d}{n_fail:7d}")

# ── figure: signed change at P_PLOT ───────────────────────────────────────────
ip = P_GRID.index(P_PLOT)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for k, (label, r) in enumerate(results.items()):
    if label == 'baseline':
        continue
    c = f"C{k % 10}"
    ax[0].plot(MG_GRID, r['Ts'][ip] - base['Ts'][ip], '-o', ms=3, color=c, label=label)
    ax[1].plot(MG_GRID, r['Tl'][ip] - base['Tl'][ip], '-o', ms=3, color=c)
for a, name in zip(ax, [r'$\Delta T_{sol}$ (K)', r'$\Delta T_{liq}$ (K)']):
    a.axhline(0, color='k', lw=0.8)
    a.set_xlabel(r'Mg$\#_{BML}$', fontsize=14)
    a.set_ylabel(name, fontsize=14)
    a.set_title(f'P = {P_PLOT} GPa', fontsize=14)
    a.tick_params(labelsize=14)
ax[0].legend(fontsize=9, loc='best')
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '134_phase_param_sensitivity.png'), dpi=200)
print("\nsaved 134_phase_param_sensitivity.png")


# baseline branch map: which Mg# sits on which branch at each P
print("\nbaseline branch map (gamm = gamma, beta = beta, thre = three-phase -> beta):")
for i, P in enumerate(P_GRID):
    print(f"  {P:5.1f} GPa  " + '  '.join(f"{X:.2f}:{str(b)[:4]}" for X, b in zip(MG_GRID, base['ph'][i])))

# split the sensitivity: grid points on the same branch vs points that switched branch
print(f"\n{'case':22s}{'same: max|dT_sol|':>19s}{'same: max|dT_liq|':>19s}"
      f"{'switched: max|dT_liq|':>23s}{'n_switched':>12s}")
for label, r in results.items():
    if label == 'baseline':
        continue
    same = r['ph'] == base['ph']
    dS = np.abs(r['Ts'] - base['Ts'])
    dL = np.abs(r['Tl'] - base['Tl'])
    s_sol = np.nanmax(np.where(same, dS, np.nan))
    s_liq = np.nanmax(np.where(same, dL, np.nan))
    w_liq = np.nanmax(np.where(~same, dL, np.nan)) if (~same).any() else np.nan
    print(f"{label:22s}{s_sol:19.1f}{s_liq:19.1f}{w_liq:23.1f}{int((~same).sum()):12d}")