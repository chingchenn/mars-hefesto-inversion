#!/usr/bin/env python3
"""
plot_mcmc.py  —  MCMC 結果視覺化 (for 50_mcmc_main.py new version)
用法: python plot_mcmc.py --chain_dir /path/to/mcmc_output --burnin 200
"""

import json, glob, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import fsolve, brentq, curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('--chain_dir', type=str, default='.')
parser.add_argument('--burnin',    type=float, default=0.3,
                    help='burn-in 比例，例如 0.3 = 丟掉前 30%%')
parser.add_argument('--out',       type=str, default='/net/beno3/data1/jcchen2/mars-hefesto-runs/figures')
parser.add_argument('--prefix',    type=str, default=None,
                    help='只讀取符合這個 prefix 的資料夾，例如 chain_a22')
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
TAG = f'{args.prefix}_' if args.prefix else ''
# ── 讀取所有 chain ─────────────────────────────────────────────────────────────
pattern = f'{args.prefix}*/chain.json' if args.prefix else '*/chain.json'
chain_files = sorted(glob.glob(os.path.join(args.chain_dir, pattern)))
if not chain_files:
    chain_files = sorted(glob.glob(os.path.join(args.chain_dir, 'chain.json')))
print(f"找到 {len(chain_files)} 條 chain")

all_chains = []
for f in chain_files:
    with open(f) as fh:
        all_chains.append(json.load(fh))

# burn-in: 每條 chain 各自算前 X% 的步數
def get_burnin(chain):
    if len(chain) == 0:
        return 0
    return int(chain[-1]['step'] * args.burnin)

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)', 'Mg#_bulk_bml', 'BML thickness (km)']
COLORS = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
          'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

# ── 圖1: Trace plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=False)
fig.suptitle('Trace plots', fontsize=13)
for i, (param, label) in enumerate(zip(PARAMS, LABELS)):
    ax = axes[i]
    for ci, chain in enumerate(all_chains):
        steps  = [s['step'] for s in chain]
        values = [s['params'][param] for s in chain]
        ax.plot(steps, values, lw=0.5, alpha=0.8, color=COLORS[ci % len(COLORS)], label=f'chain {ci}')
        ax.axvline(get_burnin(chain), color='k', ls='--', lw=0.8,
                   label=f'burn-in ({args.burnin*100:.0f}%)' if (i==0 and ci==0) else '')
    ax.set_ylabel(label, fontsize=9)
    if i == 0:
        ax.legend(fontsize=7, ncol=min(len(all_chains), 5))
axes[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '01_trace.png'), dpi=150)
plt.close()
print("saved 01_trace.png")

# ── 圖2: Misfit history ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
fig.suptitle('Misfit history', fontsize=13)
for ax, key, label in zip(axes,
    ['misfit', 'misfit_tt', 'misfit_solidus'],
    ['Total misfit', 'TT misfit', 'Solidus penalty']):
    for ci, chain in enumerate(all_chains):
        steps  = [s['step'] for s in chain]
        values = [s.get(key, 999) for s in chain]
        ax.plot(steps, values, lw=0.5, alpha=0.8, color=COLORS[ci % len(COLORS)])
        ax.axvline(get_burnin(chain), color='k', ls='--', lw=0.8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(0, 20)
axes[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '02_misfit_history.png'), dpi=150)
plt.close()
print("saved 02_misfit_history.png")

# ── 圖3: Acceptance rate ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for ci, chain in enumerate(all_chains):
    steps = [s['step'] for s in chain]
    rates = [s['accept_rate'] for s in chain]
    ax.plot(steps, rates, lw=0.8, color=COLORS[ci % len(COLORS)], label=f'chain {ci}')
for chain in all_chains:
    ax.axvline(get_burnin(chain), color='k', ls='--', lw=0.8)
ax.axvline(get_burnin(all_chains[0]), color='k', ls='--', lw=0.8, label=f'burn-in ({args.burnin*100:.0f}%)')
ax.axhline(20, color='gray', ls=':', lw=1)
ax.axhline(30, color='gray', ls=':', lw=1)
ax.set_xlabel('Step'); ax.set_ylabel('Acceptance rate (%)'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '03_accept_rate.png'), dpi=150)
plt.close()
print("saved 03_accept_rate.png")

# ── 合併 post-burnin ───────────────────────────────────────────────────────────
samples = [s for chain in all_chains for s in chain if s['step'] > get_burnin(chain)]
print(f"Post burn-in samples: {len(samples)}")
if not samples:
    print("沒有 post-burnin samples，請減少 --burnin"); exit()

post = {p: np.array([s['params'][p] for s in samples]) for p in PARAMS}
post['misfit']         = np.array([s['misfit'] for s in samples])
post['upper_contrast'] = np.array([s['upper_contrast'] if s['upper_contrast'] is not None else np.nan for s in samples])
post['lower_contrast'] = np.array([s['lower_contrast'] if s['lower_contrast'] is not None else np.nan for s in samples])
post['Ra']             = np.array([s['Ra']             if s.get('Ra')             is not None else np.nan for s in samples])
post['h_solid_km']     = np.array([s['h_solid_km']     if s.get('h_solid_km')     is not None else np.nan for s in samples])
post['h_liquid_km']    = np.array([s['h_liquid_km']    if s.get('h_liquid_km')    is not None else np.nan for s in samples])
post['Mg_solid']       = np.array([s['Mg_solid']       if s.get('Mg_solid')       is not None else np.nan for s in samples])
post['Mg_liquid']      = np.array([s['Mg_liquid']      if s.get('Mg_liquid')      is not None else np.nan for s in samples])
post['T_interface']    = np.array([s['T_interface']    if s.get('T_interface')    is not None else np.nan for s in samples])
post['thermal_state']  = np.array([s.get('thermal_state', '') for s in samples])

# ── 圖4: 1D Marginals ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('Posterior marginals (post burn-in)', fontsize=13)
for ax, param, label in zip(axes.flatten(), PARAMS, LABELS):
    ax.hist(post[param], bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(np.median(post[param]),          color='red',    ls='--', lw=1.5, label='median')
    ax.axvline(np.percentile(post[param],  16), color='orange', ls=':',  lw=1)
    ax.axvline(np.percentile(post[param],  84), color='orange', ls=':',  lw=1, label='16/84%')
    ax.set_xlabel(label, fontsize=9); ax.set_ylabel('count', fontsize=9); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '04_marginals.png'), dpi=150)
plt.close()
print("saved 04_marginals.png")

# ── 圖5: BML thickness vs Mg#_bulk_bml ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(post['Mg#_bulk_bml'], post['BML_thickness'],
                c=post['misfit'], cmap='viridis_r', s=4, alpha=0.5, vmax=10)
plt.colorbar(sc, ax=ax, label='total misfit')
ax.set_xlabel('Mg#_bulk_bml', fontsize=11)
ax.set_ylabel('BML thickness (km)', fontsize=11)
ax.set_title('BML bulk composition vs. thickness', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '05_bml_thickness_vs_mgnum.png'), dpi=150)
plt.close()
print("saved 05_bml_thickness_vs_mgnum.png")

# ── 圖6: T_core vs Mg#_bulk_bml 疊 phase diagram ─────────────────────────────
# 重建 Pierru 2026 solidus/liquidus (同 main code)
_W   = 8000.0; _R = 8.314
_PIERRU_SOL = {7.95: 1890, 9.6: 1960, 14.1: 2090, 17.7: 2200}

def _get_DH(P):
    if P < 14: return 142000., 89000.
    elif P < 20: return 116000., 73000.
    else: return 105000., 74000.

def _Tm_Fo(P): return 2163.0 * (P/65.67 + 1.0)**(1.0/0.7809)

def _solve_XS_XL(T, P, TmFa):
    DH_Fo, DH_Fa = _get_DH(P); TmFo = _Tm_Fo(P)
    def eqs(v):
        XS, XL = np.clip(v, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS) + _W/(_R*T)*(1-XS)**2 - (DH_Fo/_R)*(1/TmFo - 1/T)
        e2 = 2*np.log((1-XL)/(1-XS)) + _W/(_R*T)*XS**2 - (DH_Fa/_R)*(1/TmFa - 1/T)
        return [e1, e2]
    for x0 in [[0.85,0.65],[0.80,0.55],[0.70,0.45],[0.90,0.70]]:
        sol, info, ier, _ = fsolve(eqs, x0, full_output=True)
        XS, XL = sol
        if ier==1 and np.max(np.abs(info['fvec']))<1e-6 and 0<XS<1 and 0<XL<1 and XS>XL:
            return float(np.clip(XS,0,1)), float(np.clip(XL,0,1))
    return np.nan, np.nan

def _get_TmFa_anchor(P_anchor):
    Tsol = _PIERRU_SOL[P_anchor]; TmFo_P = _Tm_Fo(P_anchor); DH_Fo, DH_Fa = _get_DH(P_anchor); XS=0.75
    def eqs(v):
        TmFa, XL = v; XL = np.clip(XL, 1e-9, 1-1e-9)
        e1 = 2*np.log(XL/XS) + _W/(_R*Tsol)*(1-XS)**2 - (DH_Fo/_R)*(1/TmFo_P - 1/Tsol)
        e2 = 2*np.log((1-XL)/(1-XS)) + _W/(_R*Tsol)*XS**2 - (DH_Fa/_R)*(1/TmFa - 1/Tsol)
        return [e1, e2]
    for g in [[1400,0.40],[1600,0.35],[1200,0.45]]:
        sol, info, ier, _ = fsolve(eqs, g, full_output=True)
        if ier==1 and np.max(np.abs(info['fvec']))<1e-8: return float(sol[0])
    return np.nan

_sg_P = np.array([0.0001] + list(_PIERRU_SOL.keys()))
_sg_T = np.array([1478.0] + [_get_TmFa_anchor(P) for P in _PIERRU_SOL.keys()])
_sg_popt, _ = curve_fit(lambda P, T0, a, c: T0*(P/a+1)**(1/c), _sg_P, _sg_T,
                         p0=[1478,10,1.5], bounds=([1000,0.1,0.1],[3000,200,10]))
def Tm_Fa(P): T0,a,c = _sg_popt; return float(T0*(P/a+1)**(1/c))

# 在代表壓力 P_ref 畫 solidus/liquidus vs Mg#
P_ref  = 10.0
mg_arr = np.linspace(0.40, 0.80, 120)
TmFa_ref = Tm_Fa(P_ref)
T_sol_line, T_liq_line = [], []
for mg in mg_arr:
    def sol_res(T): XS, XL = _solve_XS_XL(T, P_ref, TmFa_ref); return (XS-mg) if not np.isnan(XS) else 999
    def liq_res(T): XS, XL = _solve_XS_XL(T, P_ref, TmFa_ref); return (XL-mg) if not np.isnan(XL) else 999
    try: Tsol = brentq(sol_res, 1200, 3200, xtol=1.0)
    except: Tsol = np.nan
    try: Tliq = brentq(liq_res, Tsol if not np.isnan(Tsol) else 1200, 4000, xtol=1.0)
    except: Tliq = np.nan
    T_sol_line.append(Tsol); T_liq_line.append(Tliq)
T_sol_line = np.array(T_sol_line); T_liq_line = np.array(T_liq_line)

# 色碼：conductive=藍, convective=紅, 其他=灰
c_arr = np.where(post['thermal_state']=='conductive', 'steelblue',
         np.where(post['thermal_state']=='convective', 'tomato', 'gray'))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(post['Mg#_bulk_bml'], post['T_core'], c=c_arr, s=4, alpha=0.4)
ax.plot(mg_arr, T_sol_line, 'b-',  lw=2, label=f'solidus @ {P_ref} GPa')
ax.plot(mg_arr, T_liq_line, 'r-',  lw=2, label=f'liquidus @ {P_ref} GPa')
ax.fill_between(mg_arr, T_sol_line, T_liq_line, alpha=0.1, color='green', label='partial melt')
legend_handles = [
    mpatches.Patch(color='steelblue', label='conductive'),
    mpatches.Patch(color='tomato',    label='convective'),
    plt.Line2D([0],[0], color='b', lw=2, label=f'solidus @ {P_ref} GPa'),
    plt.Line2D([0],[0], color='r', lw=2, label=f'liquidus @ {P_ref} GPa'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_xlabel('Mg#_bulk_bml', fontsize=11)
ax.set_ylabel('T_core (K)', fontsize=11)
ax.set_title(f'T_core vs. Mg#_bulk_bml  (phase diagram @ P_ref={P_ref} GPa)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(args.out, TAG + '06_Tcore_vs_mgnum_phase.png'), dpi=150)
plt.close()
print("saved 06_Tcore_vs_mgnum_phase.png")

# ── 圖7: h_solid vs h_liquid ──────────────────────────────────────────────────
mask7 = np.isfinite(post['h_solid_km']) & np.isfinite(post['h_liquid_km'])
if mask7.sum() > 0:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(post['h_solid_km'][mask7], post['h_liquid_km'][mask7],
                    c=post['misfit'][mask7], cmap='viridis_r', s=4, alpha=0.5, vmax=10)
    plt.colorbar(sc, ax=ax, label='total misfit')
    # 線：h_solid + h_liquid = BML_thickness (diagonal)
    total = post['BML_thickness'][mask7]
    x_line = np.linspace(0, total.max(), 100)
    ax.plot(x_line, np.median(total) - x_line, 'k--', lw=1, label=f'median total = {np.median(total):.0f} km')
    ax.set_xlabel('h_solid (km)', fontsize=11)
    ax.set_ylabel('h_liquid (km)', fontsize=11)
    ax.set_title('Solid vs. liquid BML thickness', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, TAG + '07_solid_vs_liquid_thickness.png'), dpi=150)
    plt.close()
    print("saved 07_solid_vs_liquid_thickness.png")

# ── 圖8: Rayleigh number distribution ─────────────────────────────────────────
Ra_valid = post['Ra'][np.isfinite(post['Ra']) & (post['Ra'] > 0)]
if len(Ra_valid) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(np.log10(Ra_valid), bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    axes[0].axvline(np.log10(1000), color='r', ls='--', lw=1.5, label='Ra_c = 1000')
    axes[0].set_xlabel('log10(Ra)', fontsize=11); axes[0].set_ylabel('count'); axes[0].legend()
    axes[0].set_title('Rayleigh number distribution')

    # pie: conductive vs convective
    n_cond = np.sum(post['thermal_state'] == 'conductive')
    n_conv = np.sum(post['thermal_state'] == 'convective')
    n_other = len(post['thermal_state']) - n_cond - n_conv
    labels = [f'conductive\n({n_cond})', f'convective\n({n_conv})']
    sizes  = [n_cond, n_conv]
    if n_other > 0:
        labels.append(f'other\n({n_other})'); sizes.append(n_other)
    axes[1].pie(sizes, labels=labels, colors=['steelblue','tomato','gray'],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Thermal state fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, TAG + '08_rayleigh.png'), dpi=150)
    plt.close()
    print("saved 08_rayleigh.png")

# ── 圖9: Density contrast (gravitational stability) ────────────────────────────
uc = post['upper_contrast']; lc = post['lower_contrast']
mask9 = np.isfinite(uc) & np.isfinite(lc)
if mask9.sum() > 0:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(uc[mask9], lc[mask9], c=post['misfit'][mask9],
                    cmap='viridis_r', s=5, alpha=0.5, vmax=10)
    plt.colorbar(sc, ax=ax, label='total misfit')
    ax.axvline(0, color='k', lw=1.5, ls='--'); ax.axhline(0, color='k', lw=1.5, ls='--')
    ax.set_xlabel('upper contrast: ρ_BML_top − ρ_mantle_bot (g/cc)', fontsize=10)
    ax.set_ylabel('lower contrast: ρ_core_top − ρ_BML_bot (g/cc)', fontsize=10)
    ax.set_title('Gravitational stability  (stable = both > 0)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, TAG + '09_density_contrast.png'), dpi=150)
    plt.close()
    print("saved 09_density_contrast.png")

# ── 圖10: Velocity profiles from .npz ─────────────────────────────────────────
npz_files = sorted(glob.glob(os.path.join(args.chain_dir, '*/profile_s*.npz')))
# npz: 用所有 chain 的最大 burnin step 做簡單過濾
max_burnin = max(get_burnin(c) for c in all_chains) if all_chains else 0
npz_post  = [f for f in npz_files
             if int(os.path.basename(f).replace('profile_s','').replace('.npz','')) > max_burnin]
print(f"Post burn-in .npz profiles: {len(npz_post)}")

if npz_post:
    depth_grid = np.linspace(0, 2000, 500)
    Vp_all=[]; Vs_all=[]; rho_all=[]; T_all=[]
    # BML 剖面
    bml_Vp_all=[]; bml_Vs_all=[]; bml_rho_all=[]; bml_phi_all=[]; bml_T_all=[]
    bml_depth_grids=[]

    for f in npz_post:
        d = np.load(f, allow_pickle=True)
        depth = d['depth_km']
        Vp_all.append(np.interp(depth_grid, depth, d['Vp'],   left=np.nan, right=np.nan))
        Vs_all.append(np.interp(depth_grid, depth, d['Vs'],   left=np.nan, right=np.nan))
        rho_all.append(np.interp(depth_grid, depth, d['rho'], left=np.nan, right=np.nan))
        T_all.append(np.interp(depth_grid,   depth, d['T_K'], left=np.nan, right=np.nan))
        if 'bml_depth_km' in d:
            bml_d = d['bml_depth_km']
            bml_grid = np.linspace(bml_d[0], bml_d[-1], 50)
            bml_depth_grids.append(bml_grid)
            bml_Vp_all.append(np.interp(bml_grid, bml_d, d['bml_Vp']))
            bml_Vs_all.append(np.interp(bml_grid, bml_d, d['bml_Vs']))
            bml_rho_all.append(np.interp(bml_grid, bml_d, d['bml_rho']))
            bml_phi_all.append(np.interp(bml_grid, bml_d, d['bml_phi']))
            bml_T_all.append(np.interp(bml_grid,   bml_d, d['bml_T_K']))

    # 圖10a: Vp/Vs/rho 剖面
    fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharey=True)
    fig.suptitle('Mantle velocity/density profiles (post burn-in ensemble)', fontsize=12)
    for ax, arr, xlabel in zip(axes, [Vp_all, Vs_all, rho_all], ['Vp (km/s)', 'Vs (km/s)', 'ρ (g/cc)']):
        arr = np.array(arr)
        med = np.nanmedian(arr, axis=0)
        p16 = np.nanpercentile(arr, 16, axis=0)
        p84 = np.nanpercentile(arr, 84, axis=0)
        ax.plot(med, depth_grid, 'b-', lw=2, label='median')
        ax.fill_betweenx(depth_grid, p16, p84, alpha=0.3, color='steelblue', label='16–84%')
        ax.set_xlabel(xlabel, fontsize=10); ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel('Depth (km)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, TAG + '10a_mantle_profiles.png'), dpi=150)
    plt.close()
    print("saved 10a_mantle_profiles.png")

    # 圖10b: BML phi / Vs / rho 剖面 (depth relative to BML top)
    if bml_phi_all:
        fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
        fig.suptitle('BML profiles (solid + liquid, post burn-in)', fontsize=12)
        # 用 relative depth (0 = BML top)
        for ax, arr_list, xlabel in zip(axes,
            [bml_phi_all, bml_Vs_all, bml_rho_all],
            ['melt fraction φ', 'Vs (km/s)', 'ρ (g/cc)']):
            # 統一 grid (relative depth)
            rel_grids = [g - g[0] for g in bml_depth_grids]
            max_rel   = np.median([g[-1] for g in rel_grids])
            rel_grid  = np.linspace(0, max_rel, 60)
            interp_arr = []
            for vals, rg in zip(arr_list, rel_grids):
                interp_arr.append(np.interp(rel_grid, rg, vals, left=np.nan, right=np.nan))
            interp_arr = np.array(interp_arr)
            med = np.nanmedian(interp_arr, axis=0)
            p16 = np.nanpercentile(interp_arr, 16, axis=0)
            p84 = np.nanpercentile(interp_arr, 84, axis=0)
            ax.plot(med, rel_grid, 'r-', lw=2, label='median')
            ax.fill_betweenx(rel_grid, p16, p84, alpha=0.3, color='salmon', label='16–84%')
            ax.set_xlabel(xlabel, fontsize=10); ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel('Depth below BML top (km)', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, TAG + '10b_bml_profiles.png'), dpi=150)
        plt.close()
        print("saved 10b_bml_profiles.png")

print(f"\n完成！所有圖存在: {args.out}/")