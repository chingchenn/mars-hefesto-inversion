#!/usr/bin/env python3
"""
Plot top-20 best-fit nGibbs MCMC profiles vs HeFESTo vs Samuel 2023 vs Khan 2023
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MCMC_BASE_DIR  = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
CHAIN_PREFIX   = 'chain_a21'
HEF_PREFIX     = 'chain_a07'
KHAN_MODEL_DIR = '/net/beno3/data1/jcchen2/Mars_Khan_2023/LSL_Models'
SAMUEL_BASE    = ('/net/beno3/data1/jcchen2/Mars_Samuel_2023/'
                  'Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1')

DEPTH_MAX      = 1800
SPLICE_DEPTH   = 200.0
MARS_RADIUS    = 3389.5
TRUE_CMB_DEPTH = 1743.3
PHI_OUTER_CORE = 0.30
MG_DUNCAN_REF  = 0.75
TOP_N          = 20

# ── PIERRU CORRECTION ─────────────────────────────────────────────────────────
def solidus_duncan2018(P):
    P = np.atleast_1d(np.asarray(P, dtype=float))
    T_C = np.where(P <= 10.0,
            -4.877*P**2 + 120.2*P + 1088.0,
            np.where(P <= 23.0,
                -1.323*(P-10)**2 + 38.18*(P-10) + 1802.0,
                77.75*(P-23) + 2075.0))
    return T_C + 273.15

def compute_phi(T_K, P_GPa, Mg_bml):
    dT   = -6.0 * (100*(1-Mg_bml) - 100*(1-MG_DUNCAN_REF))
    Tsol = solidus_duncan2018(P_GPa) + dT
    Tliq = (2160.6 + 64.7109*P_GPa - 3.97463*P_GPa**2 + 0.0957894*P_GPa**3) + dT
    Tliq = np.where(Tliq <= Tsol, Tsol + 200, Tliq)
    return np.clip((T_K - Tsol) / (Tliq - Tsol), 0, 1)

def apply_pierru(Vp, Vs, phi):
    phi_n = [0.0, 0.10, 0.20, 0.30, 1.0]
    dVp_n = [0.0, 0.10, 0.175, 0.20, 0.20]
    dVs_n = [0.0, 0.20, 0.40,  1.00, 1.00]
    dVp = np.interp(phi, phi_n, dVp_n)
    dVs = np.interp(phi, phi_n, dVs_n)
    return Vp*(1-dVp), np.maximum(Vs*(1-dVs), 0.0)

# ── READ FORT.56 ──────────────────────────────────────────────────────────────
def read_fort56(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            f.readline(); cols = f.readline().split()
        df = pd.read_csv(path, sep=r'\s+', skiprows=2, names=cols)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['P(GPa)', 'T(K)', 'rho(g/cm^3)', 'VS(km/s)', 'VP(km/s)'])
        if df.empty: return None
        P   = df['P(GPa)'].values
        rho = df['rho(g/cm^3)'].values
        dP  = np.diff(P) * 1e9
        dz  = dP / ((rho[:-1]+rho[1:])/2 * 1000 * 3.72) / 1000
        depth = np.zeros(len(P)); depth[1:] = np.cumsum(dz)
        return {'depth': depth, 'P': P, 'T': df['T(K)'].values,
                'rho': rho, 'Vp': df['VP(km/s)'].values, 'Vs': df['VS(km/s)'].values}
    except:
        return None

# ── FIND TOP-N nGibbs MODELS ──────────────────────────────────────────────────
def find_top_n(base_dir, prefix, n=20):
    results = []
    for fpath in sorted(glob.glob(f'{base_dir}/{prefix}_*/chain.json')):
        try:
            chain = json.load(open(fpath))
            for s in chain:
                if s.get('accepted'):
                    results.append({
                        'misfit':    s['misfit'],
                        'step':      s['step'],
                        'chain_dir': os.path.dirname(fpath),
                        'params':    s['params'],
                    })
        except:
            continue
    results.sort(key=lambda x: x['misfit'])
    return results[:n]

# ── FIND BEST HeFESTo MODEL ───────────────────────────────────────────────────
def find_best(base_dir, prefix):
    best = {'misfit': 999.0, 'step': None, 'chain_dir': None, 'params': None}
    for fpath in sorted(glob.glob(f'{base_dir}/{prefix}_*/chain.json')):
        try:
            chain = json.load(open(fpath))
            for s in chain:
                if s.get('accepted') and s['misfit'] < best['misfit']:
                    best = {'misfit': s['misfit'], 'step': s['step'],
                            'chain_dir': os.path.dirname(fpath),
                            'params':    s['params']}
        except:
            continue
    return best

# ── LOAD SAMUEL 2023 ──────────────────────────────────────────────────────────
print("Loading Samuel 2023...")
def r2d(r): return MARS_RADIUS - r

sam_vp,  dep_vp  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vp_profile.dat').T
sam_vs,  dep_vs  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vs_profile.dat').T
sam_rho, dep_rho = np.loadtxt(f'{SAMUEL_BASE}/PANEL_J/rho_profile.dat').T
sam_T,   dep_T   = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tprofile.dat').T
sam_sol, dep_sol = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tsol.dat').T
sam_liq, dep_liq = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tliq.dat').T

# ── LOAD KHAN ENSEMBLE ────────────────────────────────────────────────────────
print("Loading Khan models...")
khan_d, khan_vp, khan_vs, khan_rho = [], [], [], []
for fpath in sorted(glob.glob(f'{KHAN_MODEL_DIR}/Model_*.txt')):
    try:
        d = np.loadtxt(fpath, comments='#')
        mask = (d[:,2] > 0.01) & (d[:,0] <= DEPTH_MAX)
        if mask.sum() < 5: continue
        khan_d.append(d[mask,0]); khan_vp.append(d[mask,1])
        khan_vs.append(d[mask,2]); khan_rho.append(d[mask,3])
    except:
        continue
print(f"  {len(khan_d)} Khan models")

Z = np.linspace(0, DEPTH_MAX, 500)
def khan_stats(vals):
    mat = np.full((len(khan_d), len(Z)), np.nan)
    for i, (z, v) in enumerate(zip(khan_d, vals)):
        if z.max() > DEPTH_MAX*0.8:
            mat[i] = np.interp(Z, z, v, left=np.nan, right=np.nan)
    return np.nanmedian(mat,0), np.nanpercentile(mat,16,0), np.nanpercentile(mat,84,0)

vp_med,  vp_p16,  vp_p84  = khan_stats(khan_vp)
vs_med,  vs_p16,  vs_p84  = khan_stats(khan_vs)
rho_med, rho_p16, rho_p84 = khan_stats(khan_rho)

# ── LOAD TOP-20 nGibbs PROFILES ───────────────────────────────────────────────
print(f"\nSearching top {TOP_N} in {CHAIN_PREFIX}...")
top_n = find_top_n(MCMC_BASE_DIR, CHAIN_PREFIX, n=TOP_N)
ng_profiles = []
for r in top_n:
    npz_path = f"{r['chain_dir']}/profile_s{r['step']:05d}.npz"
    if os.path.exists(npz_path):
        ng_profiles.append({
            'npz':       np.load(npz_path),
            'misfit':    r['misfit'],
            'params':    r['params'],
            'chain_dir': r['chain_dir'],   # ← 加這行
            'step':      r['step'],        # ← 加這行
        })
print(f"  Loaded {len(ng_profiles)} nGibbs profiles")
best_ng = ng_profiles[0]
best_npz = best_ng['npz']

# pressure axis reference from best model
ng_P     = best_npz['P_GPa']
ng_depth = best_npz['depth_km']

# ── LOAD HeFESTo BEST ─────────────────────────────────────────────────────────
print(f"\nSearching best in {HEF_PREFIX}...")
hf = find_best(MCMC_BASE_DIR, HEF_PREFIX)
print(f"  {os.path.basename(hf['chain_dir'])}  step={hf['step']}  misfit={hf['misfit']:.4f}")

hf_man     = read_fort56(f"{hf['chain_dir']}/step_{hf['step']:05d}/s3_final/fort.56")
hf_bml_raw = read_fort56(f"{hf['chain_dir']}/step_{hf['step']:05d}/bml/fort.56")
has_hf_bml = hf_bml_raw is not None and hf['params'] is not None

if has_hf_bml:
    bml_thick    = hf['params'].get('BML_thickness', 168.7)
    hf_bml_depth = (hf_bml_raw['depth'] - hf_bml_raw['depth'][0]
                    + TRUE_CMB_DEPTH - bml_thick)
    Mg_bml       = hf['params'].get('Mg#_bml', 0.7)
    phi          = compute_phi(hf_bml_raw['T'], hf_bml_raw['P'], Mg_bml)
    hf_bml_Vp, hf_bml_Vs = apply_pierru(hf_bml_raw['Vp'], hf_bml_raw['Vs'], phi)
    hf_bml_rho   = hf_bml_raw['rho']
    hf_bml_T     = hf_bml_raw['T']
    oc_mask      = phi >= PHI_OUTER_CORE
    hf_oc        = hf_bml_depth[np.argmax(oc_mask)] if oc_mask.any() else hf_bml_depth[-1]

# Khan crust splice for best nGibbs
m_shal = Z <= SPLICE_DEPTH
m_deep = ng_depth > SPLICE_DEPTH

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 10), gridspec_kw={'wspace': 0.45})
fig.patch.set_facecolor('#FAFAFA')

# colours
C_NG  = '#C0392B'   # nGibbs
C_HF  = '#2C3E50'   # HeFESTo
C_SAM = '#E74C3C'   # Samuel
C_VP  = '#4198B9'   # Khan Vp
C_VS  = '#97795D'   # Khan Vs
C_RHO = '#35838D'   # Khan rho

panel_cfg = [
    # (xlabel, khan_vals, khan_med, p16, p84, khan_color, vp_key, bml_vp_key)
    ('Vp (km/s)',  khan_vp,  vp_med,  vp_p16,  vp_p84,  C_VP,  'Vp',  'bml_Vp'),
    ('Vs (km/s)',  khan_vs,  vs_med,  vs_p16,  vs_p84,  C_VS,  'Vs',  'bml_Vs'),
    ('ρ (g/cm³)', khan_rho, rho_med, rho_p16, rho_p84, C_RHO, 'rho', 'bml_rho'),
]
sam_vals = [sam_vp/1000, sam_vs/1000, sam_rho/1000]
sam_deps = [r2d(dep_vp),  r2d(dep_vs),  r2d(dep_rho)]
hf_vals  = ([hf_man['Vp'], hf_man['Vs'], hf_man['rho']] if hf_man else [None]*3)
hf_bml_vals = ([hf_bml_Vp, hf_bml_Vs, hf_bml_rho] if has_hf_bml else [None]*3)

for i, (ax, (xlabel, kvals, med, p16, p84, kc, vkey, bvkey),
        sv, sd, hv, hbv) in enumerate(
            zip(axes[:3], panel_cfg, sam_vals, sam_deps, hf_vals, hf_bml_vals)):

    # Khan ensemble
    for kd, kv in zip(khan_d, kvals):
        ax.plot(kv, kd, color=kc, alpha=0.04, lw=0.5, rasterized=True)
    ax.fill_betweenx(Z, p16, p84, color=kc, alpha=0.20, label='Khan 16–84%')
    ax.plot(med, Z, color=kc, lw=2.0, label='Khan median')

    # Samuel
    ax.plot(sv, sd, color=C_SAM, lw=2.0, label='Samuel 2023')

    # nGibbs top-20 (thin + transparent)
    for p in ng_profiles[1:]:
        npz = p['npz']
        d   = np.concatenate([Z[m_shal], npz['depth_km'][npz['depth_km'] > SPLICE_DEPTH]])
        v   = np.concatenate([
                np.interp(Z[m_shal], ng_depth, best_npz[vkey]),
                npz[vkey][npz['depth_km'] > SPLICE_DEPTH]])
        ax.plot(v, d, color=C_NG, lw=0.8, alpha=0.25, zorder=4)
        if bvkey in npz.files:
            ax.plot(npz[bvkey], npz['bml_depth_km'],
                    color=C_NG, lw=0.8, alpha=0.25, zorder=4)

    # nGibbs best (bold)
    d_best = np.concatenate([Z[m_shal],
                              best_npz['depth_km'][best_npz['depth_km'] > SPLICE_DEPTH]])
    v_best = np.concatenate([
                np.interp(Z[m_shal], ng_depth, best_npz[vkey]),
                best_npz[vkey][best_npz['depth_km'] > SPLICE_DEPTH]])
    ax.plot(v_best, d_best, color=C_NG, lw=2.5, zorder=5,
            label=f'nGibbs best (misfit={best_ng["misfit"]:.2f})')
    if bvkey in best_npz.files:
        ax.plot(best_npz[bvkey], best_npz['bml_depth_km'],
                color=C_NG, lw=2.5, zorder=5)

    # HeFESTo mantle + BML
    if hv is not None:
        ax.plot(hv, hf_man['depth'], color=C_HF, lw=2.0, ls='-.', zorder=6,
                label=f'HeFESTo best (misfit={hf["misfit"]:.2f})')
    if hbv is not None:
        ax.plot(hbv, hf_bml_depth, color=C_HF, lw=1.8, ls=':', zorder=6)

    # BML interface lines (from best nGibbs)
    if 'bml_depth_km' in best_npz.files:
        bml_d = best_npz['bml_depth_km']
        ng_oc = float(best_npz['bml_outer_core_depth_abs'][0])
        ax.axhline(bml_d[0],  color='orange', ls='--', lw=1.0, alpha=0.6)
        ax.axhline(bml_d[-1], color='orange', ls='--', lw=1.0, alpha=0.6)
        ax.axhline(ng_oc,     color='orange', ls=':',  lw=1.0, alpha=0.5)

    ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
    ax.set_ylim(DEPTH_MAX, 0)
    ax.set_ylabel('Depth (km)' if i == 0 else '', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.grid(True, ls='--', alpha=0.3, lw=0.8)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)
    for sp in ax.spines.values(): sp.set_linewidth(2.0)

    # pressure axis
    axr = ax.twinx()
    axr.set_ylim(DEPTH_MAX, 0)
    p_ticks = np.array([0,2,4,6,8,10,12,14,16,18])
    p_ticks = p_ticks[p_ticks <= ng_P.max()*1.05]
    axr.set_yticks(np.interp(p_ticks, ng_P, ng_depth))
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray', fontsize=9)
    axr.tick_params(axis='y', colors='gray', length=4)
    for sp in axr.spines.values(): sp.set_linewidth(2.0)

# ── Panel 4: Temperature ──────────────────────────────────────────────────────
ax = axes[3]
ax.plot(sam_T,   r2d(dep_T),   color='#2980B9', lw=2.0, label='Samuel 2023 T')
ax.plot(sam_sol, r2d(dep_sol), color='orange',   lw=1.5, ls='--', label='Solidus')
ax.plot(sam_liq, r2d(dep_liq), color='brown',    lw=1.5, ls='--', label='Liquidus')

# nGibbs top-20 T (thin)
for p in ng_profiles[1:]:
    npz = p['npz']
    T_full = np.concatenate([npz['T_K'], npz['bml_T_K']]) if 'bml_T_K' in npz.files else npz['T_K']
    d_full = np.concatenate([npz['depth_km'], npz['bml_depth_km']]) if 'bml_depth_km' in npz.files else npz['depth_km']
    ax.plot(T_full, d_full, color='#E67E22', lw=0.8, alpha=0.25, zorder=4)

# nGibbs best T (bold)
T_best = np.concatenate([best_npz['T_K'], best_npz['bml_T_K']]) if 'bml_T_K' in best_npz.files else best_npz['T_K']
d_best = np.concatenate([best_npz['depth_km'], best_npz['bml_depth_km']]) if 'bml_depth_km' in best_npz.files else best_npz['depth_km']
ax.plot(T_best, d_best, color='#E67E22', lw=2.5, zorder=5,
        label=f'nGibbs best T')

# HeFESTo T
if hf_man:
    hf_T_full = np.concatenate([hf_man['T'], hf_bml_T]) if has_hf_bml else hf_man['T']
    hf_d_full = np.concatenate([hf_man['depth'], hf_bml_depth]) if has_hf_bml else hf_man['depth']
    ax.plot(hf_T_full, hf_d_full, color=C_HF, lw=2.0, ls='-.', zorder=6,
            label='HeFESTo T')

# outer core + BML markers
if 'bml_depth_km' in best_npz.files:
    bml_d = best_npz['bml_depth_km']
    ng_oc = float(best_npz['bml_outer_core_depth_abs'][0])
    ax.axhline(ng_oc, color='#E67E22', ls=':', lw=1.5, alpha=0.8)
    ax.text(0.02, ng_oc-20, f'nGibbs outer core\n{ng_oc:.0f} km',
            transform=ax.get_yaxis_transform(), fontsize=7,
            color='#E67E22', va='bottom')
    ax.axhline(bml_d[0],  color='orange', ls='--', lw=1.0, alpha=0.6)
    ax.axhline(bml_d[-1], color='orange', ls='--', lw=1.0, alpha=0.6)

if has_hf_bml:
    ax.axhline(hf_oc, color=C_HF, ls=':', lw=1.5, alpha=0.8)
    ax.text(0.02, hf_oc+10, f'HeFESTo outer core\n{hf_oc:.0f} km',
            transform=ax.get_yaxis_transform(), fontsize=7,
            color=C_HF, va='top')

ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_ylim(DEPTH_MAX, 0)
ax.set_xlabel('T (K)', fontsize=13)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.tick_params(labelsize=13)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
for sp in ax.spines.values(): sp.set_linewidth(2.0)

axr = axes[3].twinx()
axr.set_ylim(DEPTH_MAX, 0)
p_ticks = np.array([0,2,4,6,8,10,12,14,16,18])
p_ticks = p_ticks[p_ticks <= ng_P.max()*1.05]
axr.set_yticks(np.interp(p_ticks, ng_P, ng_depth))
axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray', fontsize=9)
axr.set_ylabel('Pressure (GPa)', fontsize=10, color='gray')
axr.tick_params(axis='y', colors='gray', length=4)
for sp in axr.spines.values(): sp.set_linewidth(2.0)

# ── TITLE ─────────────────────────────────────────────────────────────────────
def fmt(p):
    if not p: return ''
    return (f"T_lit={p.get('T_lit',0):.0f}K  P_lit={p.get('P_lit',0):.2f}GPa  "
            f"Mg#={p.get('Mg#',0):.3f}  T_bml={p.get('T_bml',0):.0f}K  "
            f"Mg#_bml={p.get('Mg#_bml',0):.3f}  BML={p.get('BML_thickness',0):.0f}km")

fig.suptitle(
    f'Mars Mantle+BML: nGibbs (top {len(ng_profiles)}) vs HeFESTo vs Samuel 2023 vs Khan 2023\n'
    f"nGibbs best : {os.path.basename(ng_profiles[0]['chain_dir'])}  "
    f"step {ng_profiles[0]['step']}  misfit={ng_profiles[0]['misfit']:.4f}  "
    f"|  {fmt(ng_profiles[0]['params'])}\n"
    f"HeFESTo best: {os.path.basename(hf['chain_dir'])}  step {hf['step']}  "
    f"misfit={hf['misfit']:.4f}  |  {fmt(hf['params'])}",
    fontsize=9.5, y=1.01,
)

out = f'{MCMC_BASE_DIR}/best_fit_{CHAIN_PREFIX}_top{TOP_N}.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")