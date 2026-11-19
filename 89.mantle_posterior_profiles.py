#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
89: posterior density of the interior profiles (paper Fig. 3, Bodin et al. 2012 style)

FULL_DEPTH = True  (paper Fig. 3)
  (a)-(d) T, Vp, Vs, rho from the surface to the centre (MERGE_VEL = True puts Vp and Vs
          on one velocity axis, as in Khan et al. 2023 Fig. 2a). Each depth row is a histogram of
          the posterior samples, normalised to its own maximum. White line = median.
  (e)     interface depths: BML top, ScS reflector (top of the liquid = CMB - h_liquid),
          and CMB, as horizontal histograms on the same depth axis. For solid models the
          ScS reflector is the CMB.
  Below about 1300 km a depth row can contain mantle, BML and core values at the same time,
  because the interface depths differ between models. This is the uncertainty in the
  layering, not a plotting artefact. In deep rows the colour also depends on how many
  samples are in each state, which is not reliable while the chains mix poorly; say so in
  the caption.

FULL_DEPTH = False  (SI version)
  Mantle nodes only (layer == 1), from SPLICE_DEPTH down to where COVER_MIN of the samples
  still have mantle. The grey dashed line marks where the coverage drops below 95%.

Lines: black = median of all samples (mantle rows); in the BML (dashed) and core (solid) one
median per state (solid / layered / molten), taken over the samples of that state whose node
at that depth is in that layer. The BML is split into its solid and liquid sublayers (by Vs),
so a layered BML gets two dashed lines. A line is drawn only where at least STATE_MIN_N
samples and STATE_MIN_FRAC of that state's samples are in that layer. The per-state medians are conditional on the state, so
they do not depend on the state weights, which are unreliable while the chains mix poorly.
LIT_OVERLAY adds Samuel 2023 and the Khan 2023 median as reference lines (off by default).

The black dashed line on the T panel is the mantle solidus (Duncan 2018 + 6 K per 1% Fe,
at the median Mg#), the same solidus used in the likelihood penalty, drawn only over the
mantle depth range.

Posterior weighting
  chain.jsonl stores the chain STATE at every step; a rejected step repeats the previous
  state. Profiles are saved only for accepted steps (profile_sNNNNN.npz). A posterior
  sample at step i uses the profile of the last accepted step <= i, so drawing N_DRAW
  records uniformly from all post burn-in records gives correct posterior weights.

Reading follows 66.plot_result_profile.py (a60 npz: z_km, layer, P_GPa, T_K, rho, Vp, Vs,
z_bml_top, z_cmb, h_liquid_km; layer 0 = fixed shell, 1 = mantle, 2 = BML, 3 = core).
Each layer is interpolated on its own, and a layer is split further wherever Vs or Vp jumps
between two nodes (the solid/liquid boundary inside the BML), so values are never smoothed
across an interface.
The white median line is drawn only where one layer holds >= 95% of the samples.
"""

import json, glob, os
import numpy as np
import warnings
warnings.filterwarnings('ignore', 'All-NaN slice')      # rows outside a sample's layer are NaN by design
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import mars_config as cfg

# ══════════════════════════════════════════════════════════════════════════
# settings
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = cfg.MCMC_DIR          # figures are saved next to this script
PREFIX    = 'chain_a64_YM20'      # main result (YM20, Hd, gravity penalty); was chain_a60_YM20
EXCLUDE   = []                    # a60 used ['chain_a60_EH45_00', 'chain_a60_TA_15']

FULL_DEPTH  = True                # True = surface to centre + interface panel; False = mantle only
MERGE_VEL   = True                # True = Vp and Vs in one panel (Khan 2023 Fig. 2a style); False = separate

BURNIN_FRAC = 0.4                 # same as 63v2 and 113
DROP_LAST   = 1
MIN_STEPS   = 100                 # a64 is still running; raise to 400 when finished
N_DRAW      = 3000                # posterior samples drawn (profiles loaded, cached by step)
SEED        = 0
SAVE        = False

MARS_RADIUS  = 3389.5             # km
SPLICE_DEPTH = 100.0              # km, bottom of the fixed layer (Samuel 2023)
DZ           = 2.0 if not FULL_DEPTH else 5.0     # km, depth grid spacing
NBINS        = 150                # value bins per panel
NBINS_IF     = 120                # bins for the interface histograms
COVER_MIN    = 0.30               # mantle-only mode: stop where < 30% of samples still have mantle

CMAP          = 'Greys'           # white background; coloured state lines sit on top.
                                  # 'white_turbo' = turbo with a white low end (built below)
DENSITY_GAMMA = 0.5
MED_COLOR     = 'k'               # all-sample median (mantle rows)

STATE_LINES   = True              # per-state median lines in the BML and core rows
STATES        = ['none', 'solid', 'layered', 'molten']
STATE_COLOR   = {'none': '#999999', 'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
STATE_MIN_N   = 30                # need this many samples of a state in a row to draw its median
STATE_MIN_FRAC = 0.2              # ... and at least this fraction of that state's samples
                                  # (cuts the top of each layer where only a few extreme models reach)
LAY_BML_LIQ   = 4                 # BML nodes with Vs ~ 0 (liquid sublayer), split from solid BML (2)
H_STATE_TOL   = 1.0               # km, same state labels as 63v2 and 113

LIT_OVERLAY   = True             # Samuel 2023 and Khan 2023 median as reference lines
FS            = 16

PANELS = [('T',   'T (K)'),
          ('Vp',  r'$V_P$ (km/s)'),
          ('Vs',  r'$V_S$ (km/s)'),
          ('rho', r'$\rho$ (g/cm$^3$)')]
# panels actually drawn. 'vel' = Vp and Vs together on one velocity axis
PLOT_PANELS = ([('T', 'T (K)'), ('vel', 'Velocity (km/s)'), ('rho', r'$\rho$ (g/cm$^3$)')]
               if MERGE_VEL else PANELS)
NPZ_KEY = {'T': 'T_K', 'Vp': 'Vp', 'Vs': 'Vs', 'rho': 'rho', 'P': 'P_GPa'}

# interface histograms: (key, label, colour, line style). Neutral colours, not the state colours.
INTERFACES = [('z_bml_top', 'BML top',              '0.55', '-'),
              ('z_scs',     'ScS reflector',        'k',    '-'),
              ('z_cmb',     'CMB (top of metal)',   '0.25', '--')]

JUMP_VS = 0.5                     # km/s; a larger change between two nodes is a discontinuity
JUMP_VP = 1.0                     # km/s
MG_DUNCAN_REF = 0.75              # Duncan 2018 composition; -6 K per 1% Fe relative to it
LAY_MANTLE    = 1
LAYERS        = [0, 1, 2, 3] if FULL_DEPTH else [LAY_MANTLE]

plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════════════════════
# solidus used in the likelihood (Duncan 2018, K), Fe correction as in Methods 2.2
# ══════════════════════════════════════════════════════════════════════════
def solidus_duncan2018(P):
    P = np.asarray(P, float)
    T_C = np.where(P <= 10.0, -4.877 * P**2 + 120.2 * P + 1088.0,
          np.where(P <= 23.0, -1.323 * (P - 10)**2 + 38.18 * (P - 10) + 1802.0,
                   77.75 * (P - 23) + 2075.0))
    return T_C + 273.15


# ══════════════════════════════════════════════════════════════════════════
# load chains: keep (chain_dir, step of the profile to use, Mg#) for every post burn-in record
# ══════════════════════════════════════════════════════════════════════════
records = []
n_chain = 0
for d in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '_*'))):
    name = os.path.basename(d)
    path = os.path.join(d, 'chain.jsonl')
    if name in EXCLUDE or not os.path.exists(path):
        continue
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    recs = recs[:-DROP_LAST or None]
    if len(recs) < MIN_STEPS:
        continue
    n_chain += 1
    b = int(len(recs) * BURNIN_FRAC)
    last_acc = None
    for i, r in enumerate(recs):
        if r.get('accepted'):
            last_acc = r['step']          # profile exists only for accepted steps
        if i >= b and last_acc is not None:
            records.append((d, last_acc, float(r['params']['Mg#'])))
print(f'{PREFIX}: {n_chain} chains, {len(records)} post burn-in records')

rng = np.random.default_rng(SEED)
draw = rng.choice(len(records), min(N_DRAW, len(records)), replace=False)


# ══════════════════════════════════════════════════════════════════════════
# load the drawn profiles; keep each layer as its own sorted segment
# ══════════════════════════════════════════════════════════════════════════
cache, profs, mg = {}, [], []
for k in draw:
    d, step, mgk = records[k]
    key = (d, step)
    if key not in cache:
        f = os.path.join(d, f'profile_s{step:05d}.npz')
        if not os.path.exists(f):
            cache[key] = None
        else:
            npz = np.load(f)
            lay = np.asarray(npz['layer'])
            z = np.asarray(npz['z_km'])
            segs = []
            for L in LAYERS:
                m = lay == L
                if not m.any():
                    continue
                o = np.argsort(z[m])
                zz = z[m][o]
                vv = {p: np.asarray(npz[NPZ_KEY[p]])[m][o] for p in NPZ_KEY}
                # split a layer where a velocity jumps between two nodes (the solid/liquid
                # boundary inside the BML), so the jump is never interpolated
                cut = np.where((np.abs(np.diff(vv['Vs'])) > JUMP_VS) |
                               (np.abs(np.diff(vv['Vp'])) > JUMP_VP))[0] + 1
                for a, b in zip(np.r_[0, cut], np.r_[cut, len(zz)]):
                    segs.append({'L': L, 'z': zz[a:b], **{p: vv[p][a:b] for p in vv}})
            scal = lambda s: float(np.asarray(npz[s]).ravel()[0]) if s in npz.files else np.nan
            z_cmb = scal('z_cmb')
            h_liq = scal('h_liquid_km')
            h_sol = scal('h_solid_km')
            h_sol = 0.0 if not np.isfinite(h_sol) else h_sol
            h_lq = 0.0 if not np.isfinite(h_liq) else h_liq
            # a62+: H_BML < 1 km is the 'none' state (mantle connects to the core);
            # without this it would be labelled 'fail' (or 'solid' in older scripts)
            h_bml = z_cmb - scal('z_bml_top')
            state = ('none' if (np.isfinite(h_bml) and h_bml < H_STATE_TOL) else
                     'layered' if (h_sol >= H_STATE_TOL and h_lq >= H_STATE_TOL) else
                     'molten' if h_lq >= H_STATE_TOL else
                     'solid' if h_sol >= H_STATE_TOL else 'fail')
            cache[key] = {'segs': segs, 'state': state,
                          'z_bml_top': scal('z_bml_top'), 'z_cmb': z_cmb,
                          'z_scs': z_cmb - (h_liq if np.isfinite(h_liq) else 0.0)}
    if cache[key] is not None and cache[key]['segs']:
        profs.append(cache[key])
        mg.append(mgk)
print(f'  {len(profs)} profiles used ({len(cache)} distinct npz files)')
if not profs:
    raise SystemExit('no profile npz found; check profile_sNNNNN.npz in the chain folders')


# ══════════════════════════════════════════════════════════════════════════
# common depth grid. Each layer fills from its own top down to the top of the next layer,
# so values are never interpolated across an interface.
# ══════════════════════════════════════════════════════════════════════════
z0 = 0.0 if FULL_DEPTH else SPLICE_DEPTH
z1 = MARS_RADIUS if FULL_DEPTH else max(p['segs'][-1]['z'][-1] for p in profs)
zg = np.arange(z0, z1 + DZ / 2, DZ)

grid = {p: np.full((len(profs), len(zg)), np.nan) for p in NPZ_KEY}
layer_g = np.full((len(profs), len(zg)), -1)
for i, pr in enumerate(profs):
    segs = pr['segs']
    for s, seg in enumerate(segs):
        nxt = segs[s + 1] if s + 1 < len(segs) else None
        same_layer = nxt is not None and nxt['L'] == seg['L']
        bottom = nxt['z'][0] if (nxt is not None and (FULL_DEPTH or same_layer)) else seg['z'][-1]
        inside = (zg >= seg['z'][0]) & ((zg < bottom) if bottom > seg['z'][-1] else (zg <= bottom))
        for p in NPZ_KEY:
            grid[p][i, inside] = np.interp(zg[inside], seg['z'], seg[p])
        layer_g[i, inside] = seg['L']

# split the BML into its solid (2) and liquid (4) sublayers by Vs, so a layered BML is not
# averaged across the solid/liquid boundary
layer_g[(layer_g == 2) & (grid['Vs'] < 0.1)] = LAY_BML_LIQ
cover = (layer_g == LAY_MANTLE).mean(axis=0)         # fraction of samples that are mantle at each depth
# median only where one layer holds >= 95% of the samples; in mixed rows the median of
# mantle, BML and core values together has no meaning
dominant = np.max([(layer_g == L).mean(axis=0) for L in LAYERS + [LAY_BML_LIQ]], axis=0)
# all-sample median: rows above the mantle bottom where one layer holds >= 95% of the samples.
# Below that (BML and core) the per-state medians are drawn instead.
upper = (layer_g <= LAY_MANTLE).mean(axis=0)          # fixed shell or mantle
shallow = np.zeros(len(zg), bool)
shallow[:np.argmax(upper < 0.95) if (upper < 0.95).any() else len(zg)] = True
med_rows = (dominant >= 0.95) & (shallow if (FULL_DEPTH and STATE_LINES) else True)

# per-state medians. They are conditional on the state, so they do not depend on how many
# samples each state has (unlike the colours of the heat map in the deep rows).
# One line per (state, layer) for the BML and the core: at each depth, the median over the
# samples of that state whose node at that depth lies in that layer. Rows with fewer than
# STATE_MIN_N such samples are left blank.
state_of = np.array([p['state'] for p in profs])
state_masks = {}                                     # (state, layer) -> sample-by-depth mask
for st in STATES:
    n_st = max(int((state_of == st).sum()), 1)
    for L in (2, LAY_BML_LIQ, 3):                    # solid BML, liquid BML, core
        m = (state_of == st)[:, None] & (layer_g == L)
        cnt = m.sum(axis=0)
        ok_rows = (cnt >= STATE_MIN_N) & (cnt / n_st >= STATE_MIN_FRAC)
        if ok_rows.any():
            state_masks[(st, L)] = (m, ok_rows)
print('  state counts in the draw:', {st: int((state_of == st).sum()) for st in STATES})
z_full = zg[np.where(cover >= 0.95)[0][-1]]         # deepest depth where >= 95% are mantle
if FULL_DEPTH:
    keep = np.ones(len(zg), bool)
    z_stop = zg[-1]
else:
    z_stop = zg[np.where(cover >= COVER_MIN)[0][-1]]
    keep = zg <= z_stop
print(f'  depth window {z0:.0f}-{z_stop:.0f} km; >= 95% of samples are mantle down to {z_full:.0f} km')

P_med = np.nanmedian(grid['P'], axis=0)
mg_med = np.median(mg)
T_sol = solidus_duncan2018(P_med) - 6.0 * (100 * (1 - mg_med) - 100 * (1 - MG_DUNCAN_REF))
sol_rows = (zg >= SPLICE_DEPTH) & (zg <= z_full)     # solidus only where the row is mantle

if FULL_DEPTH:
    zi = {k: np.array([p[k] for p in profs]) for k, _, _, _ in INTERFACES}
    for k, lab, _, _ in INTERFACES:
        q = np.nanpercentile(zi[k], [5, 50, 95])
        print(f'  {lab:<20s} depth median {q[1]:6.0f} km  [5-95%: {q[0]:6.0f} - {q[2]:6.0f}]'
              f'   radius median {MARS_RADIUS - q[1]:6.0f} km')


# ══════════════════════════════════════════════════════════════════════════
# literature reference profiles (only if LIT_OVERLAY)
#   Samuel 2023 Fig. 1 source data: value vs radius (km); Vp, Vs in m/s, rho in kg/m3
#   Khan 2023 LSL models: columns depth, Vp, Vs, rho, T; median over the ensemble
# ══════════════════════════════════════════════════════════════════════════
LIT_STYLE = {'Samuel 2023': '--', 'Khan 2023 median': ':'}
lit = {}
if LIT_OVERLAY:
    sam_dir = os.path.dirname(os.path.dirname(cfg.SAMUEL_RHO_PROFILE_PATH))
    for k, f, scale in [('Vp', 'PANEL_K/vp_profile.dat', 1e-3), ('Vs', 'PANEL_K/vs_profile.dat', 1e-3),
                        ('rho', 'PANEL_J/rho_profile.dat', 1e-3), ('T', 'PANEL_I/Tprofile.dat', 1.0)]:
        v, r = np.loadtxt(os.path.join(sam_dir, f)).T
        lit.setdefault(k, {})['Samuel 2023'] = (v * scale, MARS_RADIUS - r)
    kh = {k: [] for k in ['Vp', 'Vs', 'rho', 'T']}
    for f in sorted(glob.glob(os.path.join(cfg.KHAN_MODEL_DIR, 'Model_*.txt'))):
        d = np.loadtxt(f, comments='#')
        for c, k in zip([1, 2, 3, 4], ['Vp', 'Vs', 'rho', 'T']):
            kh[k].append(np.interp(zg, d[:, 0], d[:, c], left=np.nan, right=np.nan))
    for k in kh:
        if kh[k]:
            lit[k]['Khan 2023 median'] = (np.nanmedian(kh[k], axis=0), zg)
    print(f'  literature: Samuel 2023 + {len(kh["Vp"])} Khan 2023 models')


# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
if CMAP == 'Greys':                                  # stop at mid grey so the black median stays visible
    from matplotlib.colors import ListedColormap
    CMAP = ListedColormap(plt.get_cmap('Greys')(np.linspace(0.0, 0.65, 256)))
elif CMAP == 'white_turbo':                          # turbo whose lowest values fade to white
    from matplotlib.colors import ListedColormap
    t = plt.get_cmap('turbo')(np.linspace(0.1, 1.0, 230))
    fade = np.linspace(0, 1, 26)[:, None]
    head = (1 - fade) * np.ones((26, 4)) + fade * t[0]
    CMAP = ListedColormap(np.vstack([head, t]))
norm = PowerNorm(gamma=DENSITY_GAMMA, vmin=0, vmax=1)
n_heat = len(PLOT_PANELS)
if FULL_DEPTH:
    widths = [1.4 if p == 'vel' else 1 for p, _ in PLOT_PANELS] + [0.6]
    fig, axes = plt.subplots(1, n_heat + 1, figsize=(4.4 * sum(widths), 10), sharey=True,
                             gridspec_kw={'width_ratios': widths})
else:
    fig, axes = plt.subplots(1, n_heat, figsize=(4.6 * n_heat, 9), sharey=True)

def row_hist(vals, vb):
    """per-depth histogram, each row normalised to its own maximum"""
    H = np.zeros((vals.shape[1], len(vb) - 1))
    for j in range(vals.shape[1]):
        col = vals[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            h, _ = np.histogram(col, bins=vb)
            H[j] = h / max(h.max(), 1)
    return H


for ax, (p, lab) in zip(axes[:n_heat], PLOT_PANELS):
    keys = ['Vp', 'Vs'] if p == 'vel' else [p]
    allv = np.concatenate([grid[k][:, keep].ravel() for k in keys])
    lo, hi = np.nanpercentile(allv, [0.5, 99.5])
    if p == 'vel':
        lo = 0.0                                      # liquid Vs = 0
    pad = 0.05 * (hi - lo)
    vb = np.linspace(lo - pad, hi + pad, NBINS + 1)

    # Vp and Vs are normalised separately, then combined by the larger value per cell.
    # They overlap only near 4.3-5.5 km/s at 1500-1800 km depth.
    H = np.max([row_hist(grid[k][:, keep], vb) for k in keys], axis=0)
    ax.imshow(H, origin='upper', aspect='auto', cmap=CMAP, norm=norm,
              extent=[vb[0], vb[-1], zg[keep][-1] + DZ / 2, zg[keep][0] - DZ / 2])
    for k in keys:
        med = np.nanmedian(grid[k][:, keep], axis=0)
        med[~med_rows[keep]] = np.nan                # gap in the mixed-layer depth range
        ax.plot(med, zg[keep], color=MED_COLOR, lw=1.5)
        if FULL_DEPTH and STATE_LINES:
            for (st, L), (m, ok_rows) in state_masks.items():
                ms = np.nanmedian(np.where(m, grid[k], np.nan)[:, keep], axis=0)
                ms[~ok_rows[keep]] = np.nan
                ax.plot(ms, zg[keep], color=STATE_COLOR[st], lw=2.2,
                        ls='-' if L == 3 else (0, (4, 1.5)))   # dashed in the BML (both sublayers), solid in the core
        if LIT_OVERLAY and k in lit:
            for name, (lv, lz) in lit[k].items():
                ax.plot(lv, lz, color='0.25', ls=LIT_STYLE[name], lw=1.5)
        if p == 'vel':                               # label each curve at 400 km depth
            jz = np.argmin(np.abs(zg[keep] - 400.0))
            ax.text(med[jz] - 0.25, zg[keep][jz], r'$V_P$' if k == 'Vp' else r'$V_S$',
                    color=MED_COLOR, fontsize=FS, ha='right', va='center')
    if p == 'T':
        ax.plot(T_sol[sol_rows], zg[sol_rows], color='k', ls='--', lw=1.5)
    if not FULL_DEPTH and z_full < z_stop:
        ax.axhline(z_full, color='0.6', ls='--', lw=1)

    ax.set_xlabel(lab, fontsize=FS)
    ax.set_xlim(vb[0], vb[-1])
    ax.tick_params(labelsize=14)
handles = [plt.Line2D([], [], color=MED_COLOR, lw=1.5, label='median (all samples)')]
if FULL_DEPTH and STATE_LINES:
    handles += [plt.Line2D([], [], color=STATE_COLOR[st], lw=2.2, label=f'median, {st}')
                for st in STATES if any(k[0] == st for k in state_masks)]
    handles += [plt.Line2D([], [], color='0.3', lw=2.2, ls=(0, (4, 1.5)), label='(dashed: BML)')]
if LIT_OVERLAY:
    handles += [plt.Line2D([], [], color='0.25', ls=LIT_STYLE[n], lw=1.5, label=n) for n in LIT_STYLE]
axes[0].legend(handles=handles, fontsize=12, loc='lower left', frameon=False)
axes[0].set_ylabel('Depth (km)', fontsize=FS)
axes[0].set_ylim(z_stop, z0)

# (e) interface depths
if FULL_DEPTH:
    axi = axes[-1]
    zb = np.arange(z0, z_stop + DZ, (z_stop - z0) / NBINS_IF)
    for k, lab, c, ls in INTERFACES:
        v = zi[k][np.isfinite(zi[k])]
        h, e = np.histogram(v, bins=zb, density=True)
        axi.step(h / max(h.max(), 1e-12), 0.5 * (e[1:] + e[:-1]), where='mid',
                 color=c, ls=ls, lw=2, label=lab)
    axi.set_xlim(0, 1.05)
    axi.set_xticks([0, 1])
    axi.set_xlabel('interface\nprobability', fontsize=FS)
    axi.tick_params(labelsize=14)
    axi.legend(fontsize=12, loc='lower right', frameon=False)

# pressure on the right axis (median P at each depth)
axr = axes[-1].twinx()
axr.set_ylim(z_stop, z0)
ok = np.isfinite(P_med) & keep
p_step = 5.0 if FULL_DEPTH else 2.0
p_ticks = np.arange(p_step, np.nanmax(P_med[ok]) + 0.1, p_step)
axr.set_yticks(np.interp(p_ticks, P_med[ok], zg[ok]))
axr.set_yticklabels([f'{t:.0f}' for t in p_ticks])
axr.set_ylabel('Pressure (GPa)', fontsize=FS)
axr.tick_params(labelsize=14)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
cax = fig.add_axes([0.25, 0.075, 0.40, 0.018])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm), cax=cax, orientation='horizontal')
cb.set_label('posterior density (normalised per depth)', fontsize=FS)
cb.ax.tick_params(labelsize=14)
if SAVE:
    tag = 'full_depth' if FULL_DEPTH else 'mantle'
    fig.savefig(f'89_{PREFIX}_{tag}.png', dpi=200)
plt.show()
