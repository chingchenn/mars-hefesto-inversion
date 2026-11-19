import os
import glob
import json
import types
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
FORWARD_FILE   = f'{HERE}/26fall_main_mcmc.py'
COMP           = 'TA'       # a60 composition: 'YM20', 'TA' or 'EH45'
CHAIN_GLOB     = f'{MCMC_DIR}/chain_a60_{COMP}_*/chain.jsonl'
# the phase diagram is loaded from FORWARD_FILE (a60 calibration); only check a60 chains with it
EXCLUDE        = ['chain_a60_TA']     # frozen chain
BURN_FRAC      = 0.40
PROFILE_STRIDE = 5        # part 2 uses every Nth unique accepted profile
P_ROUND        = 0.02     # GPa, phase-diagram cache resolution (~1.5 km)
X_ROUND        = 0.002    # Mg#, phase-diagram cache resolution (~1-2 K in T_liq)
T_DUMMY        = 2000.0   # T argument for bml_phase_diagram; T_solidus/T_liquidus do not depend on it
COLORS         = {'solid': 'C0', 'layered': 'C2', 'molten': 'C3'}

# Question: inside the liquid BML, T follows the liquid adiabat (dT/dP = gamma T / K_S),
# but the phase diagram is only evaluated at the solid/liquid interface (or P_top if molten).
# If the liquidus of X_L is steeper than the adiabat, T - T_liq decreases with depth and the
# bottom of the liquid can be undercooled (T < T_liq), i.e. not self-consistent.
#   liquid check:  dT_liq = min over liquid nodes of  T(z) - T_liq(X_L, P(z))   (want >= 0)
#   solid check:   dT_sol = max over solid BML nodes of T(z) - T_sol(X_S, P(z)) (want <= 0)

# ── load ONLY the Fo-Fa phase diagram + liquid EoS block of the forward model ─
# Importing the whole file needs nGibbs, torch, TauP and the Samuel data on beno3.
# The block between these two markers needs only numpy and scipy, so it runs on the Mac
# and stays identical to what the MCMC used (no copied code).
src = open(FORWARD_FILE).read()
i0 = src.index('# ── physics: solidus')
i1 = src.index('# ── composition')
fwd = types.ModuleType('fwd')
fwd.__dict__.update(np=np, brentq=brentq, fsolve=fsolve)
exec(src[i0:i1], fwd.__dict__)
print(f"loaded phase-diagram block: {src[i0:i1].count(chr(10))} lines, "
      f"_DELTA_SHIFT = {fwd._DELTA_SHIFT:.1f} K")

# ── read chains: keep ALL records after burn-in (rejected steps repeat the state) ─
rows = []    # (chain_dir, record, step of last accepted state)
for path in sorted(glob.glob(CHAIN_GLOB)):
    chain_dir = os.path.dirname(path)
    if os.path.basename(chain_dir) in EXCLUDE:
        continue
    lines = [json.loads(l) for l in open(path) if l.strip()]
    n_burn = int(BURN_FRAC * len(lines))
    last_acc = None
    for i, rec in enumerate(lines):
        if rec['accepted']:
            last_acc = rec['step']
        if i >= n_burn:
            rows.append((chain_dir, rec, last_acc))
print(f"records after burn-in: {len(rows)}")

# three-state classification from the BML sublayer thicknesses
states = []
for chain_dir, rec, last_acc in rows:
    if rec['h_liquid_km'] < 1.0:
        states.append('solid')
    elif rec['h_solid_km'] < 1.0:
        states.append('molten')
    else:
        states.append('layered')
states = np.array(states)

# ── part 1: bottom of the liquid, from chain fields only ──────────────────────
# T at the base of the BML = T_core (T_core is defined at the top of the metal core)
#   dT_bot = T_core - T_liq(X_L, P_cmb)
cache = {}
dT_bot = np.full(len(rows), np.nan)
for i, (chain_dir, rec, last_acc) in enumerate(rows):
    if states[i] == 'solid':
        continue
    P = rec['P_cmb']
    X = rec['Mg_liquid']
    key = (int(round(P / P_ROUND)), int(round(X / X_ROUND)))
    if key not in cache:
        pd = fwd.bml_phase_diagram(T_DUMMY, key[0] * P_ROUND, key[1] * X_ROUND)
        cache[key] = (pd['T_solidus'], pd['T_liquidus']) if pd is not None else (np.nan, np.nan)
    dT_bot[i] = rec['params']['T_core'] - cache[key][1]

print("\n== part 1: base of liquid layer, T_core - T_liq(X_L, P_cmb) ==")
for s in ['layered', 'molten']:
    m = states == s
    v = dT_bot[m]
    ok = np.isfinite(v)
    if ok.sum() == 0:
        print(f"  {s:8s}: no samples")
        continue
    print(f"  {s:8s}: n={m.sum()}  phase-diagram failures={np.sum(~ok)}  "
          f"undercooled (dT<0) = {np.mean(v[ok] < 0)*100:.1f}%  "
          f"min={v[ok].min():+.0f} K  p1={np.percentile(v[ok], 1):+.0f}  "
          f"p5={np.percentile(v[ok], 5):+.0f}  median={np.median(v[ok]):+.0f} K")

# slope comparison at the median posterior state
#   liquidus slope: centred difference over +-0.5 GPa
#   adiabat slope:  gamma T / K_S from the Thomas 2012/2013 liquid EoS
print("\n== slopes at the posterior median CMB state ==")
for s in ['layered', 'molten']:
    m = states == s
    if m.sum() == 0:
        continue
    P0 = np.median([rows[i][1]['P_cmb'] for i in np.flatnonzero(m)])
    X0 = np.median([rows[i][1]['Mg_liquid'] for i in np.flatnonzero(m)])
    T0 = np.median([rows[i][1]['params']['T_core'] for i in np.flatnonzero(m)])
    Tl_hi = fwd.bml_phase_diagram(T_DUMMY, P0 + 0.5, X0)['T_liquidus']
    Tl_lo = fwd.bml_phase_diagram(T_DUMMY, P0 - 0.5, X0)['T_liquidus']
    slope_liq = (Tl_hi - Tl_lo) / 1.0
    slope_ad = fwd.liquid_adiabat_dTdP(P0, T0, X0)
    print(f"  {s:8s}: P={P0:.2f} GPa  X_L={X0:.3f}  T_core={T0:.0f} K   "
          f"dT_liq/dP={slope_liq:.1f} K/GPa   adiabat={slope_ad:.1f} K/GPa   "
          f"{'liquidus steeper -> bottom is the critical point' if slope_liq > slope_ad else 'adiabat steeper -> interface is the critical point'}")

# ── part 2: full depth, from saved accepted profiles ──────────────────────────
# weight of each unique state = number of post-burn-in records that sit on it
weights = {}
info = {}
for i, (chain_dir, rec, last_acc) in enumerate(rows):
    if last_acc is None:
        continue
    k = (chain_dir, last_acc)
    weights[k] = weights.get(k, 0) + 1
    info[k] = (states[i], rec['Mg_liquid'], rec['Mg_solid'], str(rec.get('phase_branch', '')), rec['h_liquid_km'])
keys = sorted(weights)[::PROFILE_STRIDE]
print(f"\n== part 2: full depth, {len(keys)} of {len(weights)} unique profiles (stride {PROFILE_STRIDE}) ==")

out_state, out_w, out_liq, out_sol, out_br, out_hl = [], [], [], [], [], []
n_missing = 0
for k in keys:
    chain_dir, step = k
    s, XL, XS, br, hl = info[k]
    f = os.path.join(chain_dir, f"profile_s{step:05d}.npz")
    if not os.path.exists(f):
        n_missing += 1
        continue
    d = np.load(f)
    m_bml = d['layer'] == 2
    P_b, T_b, Vs_b = d['P_GPa'][m_bml], d['T_K'][m_bml], d['Vs'][m_bml]
    liq = Vs_b <= 1e-6
    sol = ~liq

    dl = np.nan
    if liq.any():
        dts = []
        for P, T in zip(P_b[liq], T_b[liq]):
            key = (int(round(P / P_ROUND)), int(round(XL / X_ROUND)))
            if key not in cache:
                pd = fwd.bml_phase_diagram(T_DUMMY, key[0] * P_ROUND, key[1] * X_ROUND)
                cache[key] = (pd['T_solidus'], pd['T_liquidus']) if pd is not None else (np.nan, np.nan)
            dts.append(T - cache[key][1])
        dl = np.nanmin(dts)

    ds = np.nan
    if sol.any():
        dts = []
        for P, T in zip(P_b[sol], T_b[sol]):
            key = (int(round(P / P_ROUND)), int(round(XS / X_ROUND)))
            if key not in cache:
                pd = fwd.bml_phase_diagram(T_DUMMY, key[0] * P_ROUND, key[1] * X_ROUND)
                cache[key] = (pd['T_solidus'], pd['T_liquidus']) if pd is not None else (np.nan, np.nan)
            dts.append(T - cache[key][0])
        ds = np.nanmax(dts)

    out_state.append(s); out_w.append(weights[k]); out_liq.append(dl); out_sol.append(ds)
    out_br.append(br); out_hl.append(hl)

out_state = np.array(out_state); out_w = np.array(out_w, float)
out_liq = np.array(out_liq); out_sol = np.array(out_sol)
out_br = np.array(out_br); out_hl = np.array(out_hl, float)
RUN_TAG = os.path.basename(os.path.dirname(CHAIN_GLOB)).rstrip('*').rstrip('_')
print(f"  missing profile files: {n_missing}   phase-diagram cache entries: {len(cache)}")

print("  liquid layer, min over depth of T - T_liq(X_L, P)   (want >= 0):")
for s in ['layered', 'molten']:
    m = (out_state == s) & np.isfinite(out_liq)
    if m.sum() == 0:
        continue
    v, w = out_liq[m], out_w[m]
    vr = np.repeat(v, w.astype(int))
    print(f"    {s:8s}: n_profiles={m.sum()}  undercooled = {np.sum(w[v < 0])/w.sum()*100:.1f}% (weighted)  "
          f"min={v.min():+.0f} K  p5={np.percentile(vr, 5):+.0f}  median={np.median(vr):+.0f} K")

print("  solid sublayer, max over depth of T - T_sol(X_S, P)   (want <= 0):")
for s in ['solid', 'layered']:
    m = (out_state == s) & np.isfinite(out_sol)
    if m.sum() == 0:
        continue
    v, w = out_sol[m], out_w[m]
    vr = np.repeat(v, w.astype(int))
    print(f"    {s:8s}: n_profiles={m.sum()}  above solidus = {np.sum(w[v > 0])/w.sum()*100:.1f}% (weighted)  "
          f"max={v.max():+.0f} K  p95={np.percentile(vr, 95):+.0f}  median={np.median(vr):+.0f} K")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for s in ['layered', 'molten']:
    m = (out_state == s) & np.isfinite(out_liq)
    if m.sum():
        ax[0].hist(out_liq[m], bins=40, weights=out_w[m] / out_w[m].sum(),
                   histtype='step', lw=1.8, color=COLORS[s], label=s)
ax[0].axvline(0, color='k', ls='--', lw=1)
ax[0].set_xlabel(r'min$_z$ [$T - T_{liq}(X_L, P)$] in liquid BML (K)', fontsize=14)
ax[0].set_ylabel('posterior fraction', fontsize=14)
ax[0].legend(loc='lower right', fontsize=12)

for s in ['solid', 'layered']:
    m = (out_state == s) & np.isfinite(out_sol)
    if m.sum():
        ax[1].hist(out_sol[m], bins=40, weights=out_w[m] / out_w[m].sum(),
                   histtype='step', lw=1.8, color=COLORS[s], label=s)
ax[1].axvline(0, color='k', ls='--', lw=1)
ax[1].set_xlabel(r'max$_z$ [$T - T_{sol}(X_S, P)$] in solid BML (K)', fontsize=14)
ax[1].legend(loc='lower right', fontsize=12)

for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(f'132_{RUN_TAG}_liquidus_adiabat_check.png', dpi=200)
print(f"\nsaved 132_{RUN_TAG}_liquidus_adiabat_check.png")

# ── diagnostics: (1) undercooling vs liquid thickness, (2) solid check by phase branch ──
# (1) if the liquidus is steeper than the liquid adiabat, the base of the liquid is the coldest
#     point relative to its liquidus:  min dT ~ -(dT_liq/dz - dT_ad/dz) * h_liquid
# (2) a positive max(T - T_sol(X_S)) in the solid sublayer can come from the check itself:
#     for 'three-phase(bg)->beta' the MCMC solidus is the invariant T3, but recomputing with
#     Mg_bulk = X_S may pick another branch
print("\n== diagnostics ==")
for s in ['layered', 'molten']:
    m = (out_state == s) & np.isfinite(out_liq) & (out_hl > 1.0)
    if m.sum() > 2:
        a1, a0 = np.polyfit(out_hl[m], out_liq[m], 1)
        print(f"  {s:8s}: min dT_liq = {a0:+.1f} {a1:+.3f} * h_liquid (km)   (n = {m.sum()})")
print("  layered, solid sublayer by phase branch:")
for br in np.unique(out_br[out_state == 'layered']):
    m = (out_state == 'layered') & (out_br == br) & np.isfinite(out_sol)
    if m.sum():
        v, w = out_sol[m], out_w[m]
        print(f"    {br:24s}: n={m.sum():5d}  above solidus {np.sum(w[v > 0])/w.sum()*100:5.1f}%  "
              f"max {v.max():+.0f} K  median {np.median(np.repeat(v, w.astype(int))):+.0f} K")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for s in ['layered', 'molten']:
    m = (out_state == s) & np.isfinite(out_liq)
    ax[0].scatter(out_hl[m], out_liq[m], s=6, color=COLORS[s], alpha=0.5, label=s)
ax[0].axhline(0, color='k', ls='--', lw=1)
ax[0].set_xlabel(r'$h_{liquid}$ (km)', fontsize=14)
ax[0].set_ylabel(r'min$_z$ [$T - T_{liq}(X_L, P)$] (K)', fontsize=14)
ax[0].legend(loc='lower right', fontsize=12)
for j, br in enumerate(np.unique(out_br[out_state == 'layered'])):
    m = (out_state == 'layered') & (out_br == br) & np.isfinite(out_sol)
    if m.sum():
        ax[1].hist(out_sol[m], bins=40, weights=out_w[m] / out_w[m].sum(), histtype='step',
                   lw=1.8, color=f'C{j}', label=f'{br} (n={m.sum()})')
ax[1].axvline(0, color='k', ls='--', lw=1)
ax[1].set_xlabel(r'layered: max$_z$ [$T - T_{sol}(X_S, P)$] (K)', fontsize=14)
ax[1].legend(loc='lower right', fontsize=10)
for a in ax:
    a.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(f'132_{RUN_TAG}_liquidus_diagnostics.png', dpi=200)
print(f"saved 132_{RUN_TAG}_liquidus_diagnostics.png")