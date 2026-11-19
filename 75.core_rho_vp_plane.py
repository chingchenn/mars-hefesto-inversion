#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
75: core density vs core-top Vp, posterior coloured by BML state
    (replaces the old 113_rho_vp_plane.png, which was drawn from a52B)

Panel (a)  density at the top of the metal core vs Vp at the top of the core
           - posterior samples (rho_top recomputed with core_fes at P_cmb, T_core, w_S;
             the chain stores Vp_core_cmb but not rho_core_top)
           - Fe-FeS reference curves (w_S = 0 ... 0.365) at fixed P, T
           - Khan 2023 ensemble and Samuel 2023 core-top values from their profiles
           - grey lines: constant K_S = rho Vp^2
Panel (b)  mean core density vs Vp at the top of the core
           - both quantities are stored in the chain, no recomputation
           - literature boxes from Khan et al. RoG (submitted) Table 11

Notes
  * Irving 2023 reports a mean core density and Vp at the top of the core, not the
    core-top density, so Irving appears only in panel (b).
  * For molten/layered states the metal core lies BELOW the reflector Irving used.
    Compare molten/layered samples with Khan/Samuel, not with Irving.
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

import mars_config as cfg
import core_fes as CF
from mars_config import DATA_BASE, KHAN_MODEL_DIR

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR   = cfg.MCMC_DIR
PREFIX      = 'chain_a66_YM20'      # <-- check the actual run name
EXCLUDE     = []
BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 100                 # a63 is still early (~300 steps per chain)
H_MIN_BML   = 1.0                   # km, below this the state is 'none'
N_PLOT      = 4000                  # posterior points per panel (subsampled)
SEED        = 0
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
OUT_DIR     = cfg.FIG_DIR
TAG         = '75_' + PREFIX

# literature profiles (same paths as 71.core_EoS_results.py)
SAMUEL_BASE    = f'{DATA_BASE}/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1'

# Fe-FeS reference curves in panel (a)
REF_P   = None                      # GPa; None = posterior median P_cmb
REF_T   = (2000.0, 2400.0, 2800.0)  # K
WS_GRID = np.linspace(0.0, 0.36, 37)

# literature (mean core density g/cm3, Vp at top of core km/s); RoG Table 11
LIT_BOX = {
    'Irving 2023': dict(rho=(6.20, 6.30), vp=(4.90, 5.00), color='k'),
    'Samuel 2023': dict(rho=(6.40, 6.60), vp=(5.00, 5.20), color='#708090'),
    'Khan 2023':   dict(rho=(6.55, 6.75), vp=(5.20, 5.40), color='#4198B9'),
}

STATES      = ['none', 'solid', 'layered', 'molten']
STATE_COLOR = {'none': '#999999', 'solid': '#4E6E8E',
               'layered': '#7B9E87', 'molten': '#CD5C5C'}

FS = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
R_MARS = 3389.5


# ══════════════════════════════════════════════════════════════════════════
# loading
# ══════════════════════════════════════════════════════════════════════════
def load_jsonl(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


def load_run(prefix):
    chains, route = [], None
    for d in sorted(glob.glob(os.path.join(CHAIN_DIR, f'{prefix}_*'))):
        name = os.path.basename(d)
        path = os.path.join(d, 'chain.jsonl')
        if name in EXCLUDE or not os.path.exists(path):
            continue
        recs = load_jsonl(path)[:-DROP_LAST or None]
        if len(recs) < MIN_STEPS:
            print(f'  {name:<24s} skipped ({len(recs)} steps)')
            continue
        chains.append(recs)
        cfile = os.path.join(d, 'config.json')
        if route is None and os.path.exists(cfile):
            with open(cfile) as fh:
                route = json.load(fh).get('route')
    if not chains:
        raise SystemExit(f'no usable chains for {prefix} in {CHAIN_DIR}')
    print(f'  {prefix}: {len(chains)} chains, core EoS route = {route}')
    return chains, route


def get(rec, key):
    v = rec.get('params', {}).get(key, rec.get(key))
    return np.nan if v is None else float(v)


KEYS = ['w_S', 'T_core', 'R_cmb', 'BML_thickness',
        'P_cmb', 'Vp_core_cmb', 'rho_core_mean', 'h_solid_km', 'h_liquid_km']


def to_post(chains):
    recs = []
    for ch in chains:
        recs += ch[int(len(ch) * BURNIN_FRAC):]
    post = {k: np.array([get(r, k) for r in recs]) for k in KEYS}
    hs, hl, H = post['h_solid_km'], post['h_liquid_km'], post['BML_thickness']
    st = np.full(len(H), 'fail', dtype='<U8')
    st[(hs >= 1) & (hl >= 1)] = 'layered'
    st[(hs < 1) & (hl >= 1)] = 'molten'
    st[(hs >= 1) & (hl < 1)] = 'solid'
    st[H < H_MIN_BML] = 'none'
    post['state'] = st
    return post


print('loading chains')
chains, route = load_run(PREFIX)
post = to_post(chains)
CF.configure(route=route or 'B', verbose=False)
for s in STATES:
    print(f'  {s:<8s} {np.sum(post["state"] == s):7d}')

rng = np.random.default_rng(SEED)
ok = np.isfinite(post['P_cmb']) & np.isfinite(post['Vp_core_cmb']) & (post['state'] != 'fail')
idx = rng.choice(np.flatnonzero(ok), size=min(N_PLOT, ok.sum()), replace=False)

# core-top density and Vp recomputed from the EoS (sanity check against the chain)
rho_top = np.full(len(idx), np.nan)
vp_top = np.full(len(idx), np.nan)
for j, i in enumerate(idx):
    pr = CF.core_properties(post['P_cmb'][i], post['T_core'][i], post['w_S'][i])
    rho_top[j], vp_top[j] = float(pr['rho']), float(pr['Vp'])
dvp = vp_top - post['Vp_core_cmb'][idx]
print(f'  Vp check (EoS - chain): median {np.nanmedian(dvp):+.4f}, '
      f'max |d| {np.nanmax(np.abs(dvp)):.4f} km/s')


# ══════════════════════════════════════════════════════════════════════════
# literature core-top values from profiles (panel a)
# ══════════════════════════════════════════════════════════════════════════
def core_top(depth, rho, vp, zmin=1200.0, jump=1.0):
    """first density jump > `jump` g/cm3 below zmin = top of the metal core"""
    o = np.argsort(depth)
    depth, rho, vp = depth[o], rho[o], vp[o]
    m = depth > zmin
    d, r, v = depth[m], rho[m], vp[m]
    k = np.flatnonzero(np.diff(r) > jump)
    if k.size == 0:
        return np.nan, np.nan
    return r[k[0] + 1], v[k[0] + 1]


khan_top = []
for f in sorted(glob.glob(os.path.join(KHAN_MODEL_DIR, 'Model_*.txt'))):
    try:
        d = np.loadtxt(f, comments='#')
    except Exception:
        continue
    rho_k, vp_k = d[:, 3], d[:, 1]
    if np.nanmax(rho_k) > 100: rho_k = rho_k / 1e3
    if np.nanmax(vp_k) > 100:  vp_k = vp_k / 1e3
    khan_top.append(core_top(d[:, 0], rho_k, vp_k))
khan_top = np.array(khan_top) if khan_top else np.empty((0, 2))
print(f'  Khan 2023 ensemble: {len(khan_top)} models')

sam_top = (np.nan, np.nan)
try:
    s_rho, s_r1 = np.loadtxt(os.path.join(SAMUEL_BASE, 'PANEL_J', 'rho_profile.dat')).T
    s_vp, s_r2 = np.loadtxt(os.path.join(SAMUEL_BASE, 'PANEL_K', 'vp_profile.dat')).T
    # core top from the density jump; Vp read on the Vp file's own grid, 10 km below
    # the jump (interpolating Vp onto the density grid picked the BML side, ~5.6 km/s)
    d_rho = R_MARS - s_r1
    rho_top_s, _ = core_top(d_rho, s_rho / 1e3, np.zeros_like(s_rho))
    o = np.argsort(d_rho)
    jump = np.flatnonzero((np.diff(s_rho[o] / 1e3) > 1.0) & (d_rho[o][1:] > 1200))
    z_top = d_rho[o][jump[0] + 1]
    d_vp = R_MARS - s_r2
    ov = np.argsort(d_vp)
    vp_top_s = np.interp(z_top + 10.0, d_vp[ov], s_vp[ov] / 1e3)
    sam_top = (rho_top_s, vp_top_s)
    print(f'  Samuel 2023 core top: rho {sam_top[0]:.2f}, Vp {sam_top[1]:.2f}')
except Exception as e:
    print(f'  Samuel 2023 not loaded ({e})')


# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
st = post['state'][idx]

# (a) core-top density vs core-top Vp
a = ax[0]
for s in STATES:
    m = st == s
    if m.any():
        a.scatter(rho_top[m], vp_top[m], s=6, alpha=0.4, color=STATE_COLOR[s],
                  label=f'{s} ({m.sum()})', rasterized=True)

P_ref = REF_P if REF_P is not None else float(np.nanmedian(post['P_cmb']))
for k, T in enumerate(REF_T):
    r_ = np.array([float(CF.core_properties(P_ref, T, w)['rho']) for w in WS_GRID])
    v_ = np.array([float(CF.core_properties(P_ref, T, w)['Vp']) for w in WS_GRID])
    a.plot(r_, v_, color='k', lw=1.2, ls=['-', '--', ':'][k % 3],
           label=f'Fe–FeS, {P_ref:.0f} GPa, {T:.0f} K')
    a.plot(r_[0], v_[0], 'k*', ms=12)          # pure Fe end of the curve
    for w in (0.1, 0.2, 0.3):
        i = np.argmin(np.abs(WS_GRID - w))
        if k == 0:
            a.annotate(f'$w_S$={w:.1f}', (r_[i], v_[i]), textcoords='offset points',
                       xytext=(4, -12), fontsize=10)

if len(khan_top):
    a.scatter(khan_top[:, 0], khan_top[:, 1], s=14, marker='s', facecolor='none',
              edgecolor=LIT_BOX['Khan 2023']['color'], lw=0.8, label='Khan 2023 ensemble')
if np.isfinite(sam_top[0]):
    a.plot(*sam_top, 'D', ms=9, color=LIT_BOX['Samuel 2023']['color'], label='Samuel 2023')

a.set_xlabel(r'density at top of metal core (g/cm$^3$)', fontsize=FS)
a.set_ylabel('Vp at top of metal core (km/s)', fontsize=FS)
a.set_title('(a) core top', fontsize=FS)

# (b) mean core density vs core-top Vp
b = ax[1]
rm = post['rho_core_mean'][idx]
if np.nanmedian(rm) > 100:
    rm = rm / 1e3
for s in STATES:
    m = st == s
    if m.any():
        b.scatter(rm[m], post['Vp_core_cmb'][idx][m], s=6, alpha=0.4,
                  color=STATE_COLOR[s], rasterized=True)
for name, box in LIT_BOX.items():
    (r0, r1), (v0, v1) = box['rho'], box['vp']
    b.add_patch(plt.Rectangle((r0, v0), r1 - r0, v1 - v0, fill=False,
                              ec=box['color'], lw=2))
    b.annotate(name, (r1, v1), textcoords='offset points', xytext=(3, 3),
               fontsize=11, color=box['color'])
b.set_xlabel(r'mean core density (g/cm$^3$)', fontsize=FS)
b.set_title('(b) whole core', fontsize=FS)

# constant K_S = rho Vp^2 guide lines (GPa) on both panels
for p in ax[:2]:
    xl, yl = p.get_xlim(), p.get_ylim()
    xx = np.linspace(*xl, 50)
    for K in range(100, 260, 20):
        yy = np.sqrt(K / xx)
        p.plot(xx, yy, color='0.8', lw=0.8, zorder=0)
        if yl[0] < yy[-1] < yl[1]:
            p.text(xl[1], yy[-1], f' {K}', color='0.6', fontsize=10, va='center')
    p.set_xlim(xl); p.set_ylim(yl)
    p.tick_params(labelsize=14)
    p.grid(alpha=0.3)


# (c) w_S vs R_cmb: does the molten state hit the pure-Fe limit (w_S = 0)?
c = ax[2]
for s_ in STATES:
    m = st == s_
    if m.any():
        c.scatter(post['R_cmb'][idx][m], post['w_S'][idx][m], s=6, alpha=0.4,
                  color=STATE_COLOR[s_], rasterized=True)
c.axhline(0.0, color='k', lw=1.2, ls='--')
c.text(c.get_xlim()[0], 0.005, '  pure Fe (no Ni)', fontsize=11, va='bottom')
c.set_xlabel(r'$R_{CMB}$ (km)', fontsize=FS)
c.set_ylabel(r'$w_S$', fontsize=FS)
c.set_title(r'(c) light-element content vs core radius', fontsize=FS)
c.tick_params(labelsize=14)
c.grid(alpha=0.3)

print('  fraction with w_S < 0.02, by state (full posterior, not subsample):')
for s_ in STATES:
    m = post['state'] == s_
    if m.any():
        low = post['w_S'][m] < 0.02
        rlow = post['R_cmb'][m][low]
        extra = f', R_cmb of those: {np.median(rlow):.0f} [{rlow.min():.0f}, {rlow.max():.0f}] km' if low.any() else ''
        print(f'    {s_:<8s} {low.mean():6.3f}  (n={m.sum()}){extra}')

print('  prior-wall check, by state (fraction of samples):')
for s_ in STATES:
    m = post['state'] == s_
    if m.any():
        r, w = post['R_cmb'][m], post['w_S'][m]
        print(f'    {s_:<8s} R_cmb<1420: {np.mean(r < 1420):6.3f}   w_S<0.02: {np.mean(w < 0.02):6.3f}   '
              f'both: {np.mean((r < 1420) & (w < 0.02)):6.3f}')

ax[0].legend(fontsize=10, loc='upper left', markerscale=2)
ax[1].set_xlim(left=min(ax[1].get_xlim()[0], 6.1))
fig.suptitle(f'{PREFIX}   (core EoS route {route};  grey: constant $K_S=\\rho V_P^2$, GPa)',
             fontsize=FS)
fig.tight_layout()
if SAVE_FIG:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, TAG + '_rho_vp_plane.png')
    fig.savefig(out, dpi=200)
    print('saved', out)
plt.show()
