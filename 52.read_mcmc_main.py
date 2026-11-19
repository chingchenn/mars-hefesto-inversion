#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:42:41 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 52.mcmc_analysis.py  —  chain_a31 analysis

import json, glob, os
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import RegularGridInterpolator
from mars_config import MCMC_DIR, RUN_BASE, FIG_DIR

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR   = MCMC_DIR
OUT_DIR     = FIG_DIR
PREFIX      = 'chain_a31'
PHASE_TABLE = f'{RUN_BASE}/mcmcfigures/bml_phase_table_prior_a31.npz'

plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS = 400          # 從 200 提高,自動剔除 chain 14
RA_C        = 1.0e5          # 跟 59 的 _RA_C 一致
CHENG_TC    = (2200.0, 2400.0)
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
SEC_YR, G_MARS = 3.156e7, 3.72

RHO_S_CONST = 4.097      # g/cc, nGibbs at posterior XS, [3.98, 4.24]
fig_I_segregation = 1


PRIOR_FRAC  = {'solid': 0.284, 'layered': 0.132, 'molten': 0.584}
PRIOR_FRAC_IS_STALE = True

CUR_STEP = {'T_lit': 50.0, 'P_lit': 0.7, 'Mg#': 0.025,
            'T_core': 100.0, 'Mg#_bulk_bml': 0.06, 'BML_thickness': 25.0}
ESS_TARGET = 400

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.86),
    'T_core':        (1800.0, 3200.0),
    'Mg#_bulk_bml':  (  0.30,   0.80),
    'BML_thickness': (  50.0,  400.0),
}
UNITS = {'T_lit':'K', 'P_lit':'GPa', 'Mg#':'', 'T_core':'K',
         'Mg#_bulk_bml':'', 'BML_thickness':'km'}


A_GRAIN, C_KC, MU_MELT, ETA_MTX, N_RZ = 1e-3, 1000.0, 1.0, 1e19, 4.65
RCMF = (0.25, 0.40)

# molten 態的 TBL（沒有固態次層可用，改由邊際穩定估厚度）
ALPHA_TH, RHO_TH, KAPPA_TH, RA_C_TBL, ETA_TBL = 2e-5, 4000.0, 1e-6, 1000.0, 1e20
Q_CMB_MAX = 40.0        # mW/m2, 穩態上限（Q_CMB_RANGE 是文獻估計 5-20）

# 熱流檢查用
MARS_RADIUS     = 3389.5     # km
TRUE_CMB_DEPTH  = 1743.3     # km
K_SOLID         = 4.0        # W/m/K
Q_CMB_RANGE     = (5.0, 20.0)   # mW/m2, 文獻估計

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)']
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
bwith, fontsize = 1.5, 12

trace_plot        = 0
misfit_plot       = 1
acc_plot          = 0
marginal          = 0
corner_misfit     = 0
corner_state      = 0
bml_thickness     = 0
bml_analysis      = 0
hs_hl_thickness   = 0
Ra_plot           = 0
density_check     = 0


fig_A_misfit      = 0      # misfit vs 三種狀態
fig_B_bayes       = 0      # prior -> posterior + BF + Cheng reweighting
fig_C_grav        = 0      # 重力對比 vs Mg#
fig_G_density  = 0
fig_D_drainage = 0
fig_E_region      = 0      # 區域圖 + f_solid vs xrel  (需要 PHASE_TABLE)
code_vs_table     = 0      # code 分類 vs 相圖表邊界  (需要 PHASE_TABLE)
misfit_decomp     = 0      # misfit 分量拆解
heat_flux         = 0      # 固體 BML 傳導熱流 -> 回應 Duncan
migration_diag    = 0
fsolid_hist       = 0
convergence_diag  = 0
prior_post = 0

 
misfit_vs_thermal = 1   # B1
tt_miss_hist      = 1   # B2

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''



def liquid_bml_properties(P_GPa, T_K, Mg_liquid):
    rho0_Fo, KS0_Fo, Ksp_Fo, gam0_Fo, q_Fo, Cv_Fo, T0_Fo = 2.806, 17.5, 5.3, 0.62, -0.8, 900.0, 2273.0
    rho0_Fa, KS0_Fa, Ksp_Fa, gam0_Fa, q_Fa, Cv_Fa, T0_Fa = 3.699, 21.99, 7.28, 0.412, -0.95, 670.0, 1573.0
    Mg = float(np.clip(Mg_liquid, 0.0, 1.0))
    rho0 = Mg*rho0_Fo + (1-Mg)*rho0_Fa
    KS0  = Mg*KS0_Fo  + (1-Mg)*KS0_Fa
    Ksp  = Mg*Ksp_Fo  + (1-Mg)*Ksp_Fa
    gam0 = Mg*gam0_Fo + (1-Mg)*gam0_Fa
    q    = Mg*q_Fo    + (1-Mg)*q_Fa
    Cv   = Mg*Cv_Fo   + (1-Mg)*Cv_Fa
    T0   = Mg*T0_Fo   + (1-Mg)*T0_Fa
    P_Pa = P_GPa * 1e9
    def P_of_rho(rho):
        f  = 0.5 * ((rho/rho0)**(2/3) - 1)
        PS = 3*KS0*f*(1+2*f)**2.5 * (1 + 1.5*(Ksp-4)*f) * 1e9
        Pth = gam0 * (rho0/rho)**q * rho*1e3 * Cv * (T_K - T0)
        return PS + Pth - P_Pa
    try:
        return float(brentq(P_of_rho, rho0*0.5, rho0*3.0, xtol=1e-6))
    except Exception:
        return np.nan

def w0_seg(phi, drho, a=1e-3, K=1000.0, mu=1.0, g=G_MARS):
    k = phi**3 * a**2 / (K * (1 - phi)**2)
    return k / mu * (1 - phi) / phi * drho * g

# ── 穩健的 chain 載入 ────────────────────────────────────────────────────────
def load_chain_json(path):
    if not os.path.exists(path):
        return None, 'missing'
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return None, 'empty'
    try:
        return json.loads(text), 'ok'
    except json.JSONDecodeError:
        pass
    dec, i, recs = json.JSONDecoder(), text.find('[') + 1, []
    while True:
        while i < len(text) and text[i] in ' \t\r\n,':
            i += 1
        if i >= len(text) or text[i] == ']':
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        recs.append(obj)
    return (recs, 'salvaged') if recs else (None, 'unparseable')


chain_dirs = sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*')))
all_chains, chain_names, load_report = [], [], []

for dd in chain_dirs:
    name = os.path.basename(dd)
    ch, status = load_chain_json(os.path.join(dd, 'chain.json'))
    n_raw = 0 if ch is None else len(ch)
    if ch is not None and DROP_LAST > 0:
        ch = ch[:-DROP_LAST] if len(ch) > DROP_LAST else []
    if ch is None or len(ch) < MIN_STEPS:
        load_report.append((name, status, n_raw, 'DROPPED'))
        continue
    all_chains.append(ch)
    chain_names.append(name)
    load_report.append((name, status, n_raw, f'kept {len(ch)}'))

print(f"{'chain':<20s} {'status':<12s} {'n_raw':>7s}  action")
for name, status, n_raw, action in load_report:
    print(f"{name:<20s} {status:<12s} {n_raw:>7d}  {action}")
print(f"\nusable chains: {len(all_chains)}")
if not all_chains:
    raise SystemExit("no usable chains")


# ── 後驗(逐鏈套 burn-in,長度不必相同) ───────────────────────────────────────
samples, chain_id = [], []
for c, ch in enumerate(all_chains):
    b = int(len(ch) * BURNIN_FRAC)
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"post burn-in samples: {len(samples)}")


def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])


post = {p: np.array([s['params'][p] for s in samples], dtype=float) for p in PARAMS}
for k in ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
          'upper_contrast', 'lower_contrast', 'Ra', 'h_solid_km', 'h_liquid_km',
          'Mg_solid', 'Mg_liquid', 'T_interface', 'T_mantle_bottom', 'moi_pred',
          'tt_n_ph', 'tt_n_miss']:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

h_sol, h_liq = post['h_solid_km'], post['h_liquid_km']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= 1.0) & (h_liq >= 1.0)] = 'layered'
state[(h_sol <  1.0) & (h_liq >= 1.0)] = 'molten'
state[(h_sol >= 1.0) & (h_liq <  1.0)] = 'solid'
post['state'] = state
f_solid = h_sol / np.maximum(h_sol + h_liq, 1e-9)
print("state counts:", {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})






P_ax = np.linspace(10, 22, 61)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

ax = axes[0]
for mg, c in zip([0.30, 0.42, 0.50, 0.60],
                 ['#E45756', '#F58518', '#54A24B', '#4C78A8']):
    ax.plot(P_ax, [liquid_bml_properties(p, 2400.0, mg) for p in P_ax],
            lw=2.6, color=c, label=f'Mg#$_L$ = {mg:.2f}')
ax.axhspan(3.90, 4.40, color='#79706E', alpha=.20)
ax.axhspan(3.98, 4.24, color='#79706E', alpha=.20)
ax.text(10.3, 4.42, f'solid $\\rho_S$ = {RHO_S_CONST} [3.98, 4.24] (nGibbs at $X_S$)',
        fontsize=10, color='#79706E')
ax.axhline(4.65, color='k', ls='-.', lw=1.4)
ax.text(10.3, 4.67, 'Kono 2025 upper limit', fontsize=10)
ax.set_xlabel('P (GPa)'); ax.set_ylabel(r'$\rho_L$ (g cm$^{-3}$)')
ax.set_title('(a) liquid BML density, Thomas EoS, T = 2400 K', fontsize=11.5)
ax.tick_params(labelsize=14); ax.grid(alpha=.3); ax.legend(fontsize=10)

ax = axes[1]
rho_s_grid = np.linspace(3.8, 4.6, 200)
mg_grid    = np.linspace(0.30, 0.65, 200)
rho_L = np.array([liquid_bml_properties(18.0, 2400.0, m) for m in mg_grid])
DR = (rho_L[:, None] - rho_s_grid[None, :]) * 1000.0
cs = ax.contourf(rho_s_grid, mg_grid, DR, levels=np.arange(-800, 900, 100),
                 cmap='RdBu_r', extend='both')
ax.contour(rho_s_grid, mg_grid, DR, levels=[0], colors='k', linewidths=2.4)
plt.colorbar(cs, ax=ax, label=r'$\Delta\rho=\rho_L-\rho_S$ (kg m$^{-3}$)')

mgl = post['Mg_liquid'][(state == 'layered') & (post['Mg_liquid'] > 0)]
if mgl.size:
    ax.axhline(np.median(mgl), color='k', ls=':', lw=2)
    ax.text(3.83, np.median(mgl) + 0.008,
            f'posterior median Mg#$_L$ = {np.median(mgl):.3f}', fontsize=10)
    print(f"Mg_liquid (layered): median={np.median(mgl):.3f}  "
          f"5/95% = {np.percentile(mgl,5):.3f} / {np.percentile(mgl,95):.3f}")

ax.set_xlabel(r'solid BML $\rho_S$ (g cm$^{-3}$)')
ax.set_ylabel(r'liquid Mg#$_L$')
ax.set_title('(b) density inversion threshold @ 18 GPa, 2400 K', fontsize=11.5)
ax.tick_params(labelsize=14)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'G_density.png'), dpi=150)
plt.show()
    

phi = np.logspace(-5, -0.5, 300)
fig, ax = plt.subplots(figsize=(8.5, 6))
for d, c in zip([100, 300, 500, 700],
                ['#E45756', '#F58518', '#54A24B', '#4C78A8']):
    for D, ls in zip([5e4, 4e5], ['--', '-']):
        ax.loglog(phi, D / w0_seg(phi, d) / SEC_YR, lw=2.2, color=c, ls=ls,
                  label=(fr'$\Delta\rho$ = {d} kg m$^{{-3}}$' if D == 4e5 else None))
ax.axhline(4.5e9, color='k', ls='-.', lw=1.8)
ax.text(1.3e-5, 6e9, 'age of Mars', fontsize=11)
ax.axvspan(0.25, 0.4, color='#79706E', alpha=.25)
ax.text(0.27, 1e-3, 'RCMF', fontsize=10, rotation=90)
ax.set_xlabel(r'melt fraction $\phi$'); ax.set_ylabel(r'drainage time $D/w_0$ (yr)')
ax.set_title('solid = D 400 km, dashed = D 50 km   '
             '(no matrix viscosity needed)', fontsize=11.5)
ax.set_ylim(1e-4, 1e16)
ax.tick_params(labelsize=14); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=10)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'D_drainage.png'), dpi=150)
plt.show()

def phi_crit(D, drho, t=4.5e9):
    pp = np.logspace(-8, -0.5, 6000)
    return float(np.interp(D / (t * SEC_YR), w0_seg(pp, drho), pp))

print("\ntrapped melt fraction after 4.5 Gyr, and equivalent mush thickness")
for D in [5e4, 1e5, 2e5, 4e5]:
    for d in [200, 500, 700]:
        pc = phi_crit(D, d)
        print(f"   D={D/1e3:5.0f} km  drho={d:4.0f}  phi_crit={pc:.2e}  "
              f"-> residual melt column = {pc*D/1e3:.4f} km")


m = (state == 'layered') & np.isfinite(post['Mg_liquid']) & (post['Mg_liquid'] > 0)
D_km  = post['BML_thickness'][m]
phi_l = h_liq[m] / (h_sol[m] + h_liq[m])
T_int = post['T_interface'][m]
XL    = post['Mg_liquid'][m]
P_int = 20.9 - 0.0123 * (D_km - h_sol[m])          # 線性近似,±0.2 GPa

rL   = np.array([liquid_bml_properties(p, t, x) for p, t, x in zip(P_int, T_int, XL)])
drho = (rL - RHO_S_CONST) * 1000.0
A, K, MU, ETA, NRZ = 1e-3, 1000.0, 1.0, 1e19, 4.65

k_phi = A**2 * phi_l**3 / (K * (1 - phi_l)**2)
w_d   = k_phi/MU * (1-phi_l)/phi_l * drho * G_MARS
dc    = np.sqrt(k_phi * ETA * (1/phi_l + 4/3) / MU)
v_stk = 2*drho*G_MARS*A**2/(9*MU) * phi_l**NRZ
t_stk = D_km*1e3 / np.maximum(v_stk, 1e-30) / SEC_YR


Re = rL * 1e3 * v_stk * A_GRAIN / MU_MELT
   
 
# ---- B4: 兩套理論在瓦解門檻附近接得上（這張最該進論文）
pg   = np.logspace(-4, np.log10(0.95), 300)
dm   = np.nanmedian(drho)
kg   = A_GRAIN**2 * pg**3 / (C_KC * (1 - pg)**2)
v_dg = kg / MU_MELT * (1 - pg) / pg * dm * G_MARS
v_sg = 2 * dm * G_MARS * A_GRAIN**2 / (9 * MU_MELT) * pg**N_RZ
 
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(pg, v_dg * SEC_YR, lw=2.6, color=COLORS[3], label='Darcy (McKenzie B7)')
ax.loglog(pg, v_sg * SEC_YR, lw=2.6, color=COLORS[0], label='hindered Stokes')
ax.axvspan(*RCMF, color='k', alpha=.15)
ax.text(np.mean(RCMF), v_sg.min()*3, 'disaggregation', fontsize=10,
        rotation=90, ha='center')
ax.axvspan(np.nanpercentile(phi_l, 5), np.nanpercentile(phi_l, 95),
           color=COLORS[8], alpha=.25)
ax.text(np.nanmedian(phi_l), v_sg.min()*3, 'posterior 5-95%', fontsize=10,
        rotation=90, ha='center', color='#B45309')
ax.set_xlabel(r'melt fraction $\phi$', fontsize=fontsize)
ax.set_ylabel('separation velocity (m/yr)', fontsize=fontsize)
ax.set_title(f'two frameworks agree near the threshold '
             f'($\\Delta\\rho$={dm:.0f} kg m$^{{-3}}$, a={A_GRAIN*1e3:.0f} mm)',
             fontsize=fontsize)
ax.tick_params(labelsize=14); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=11)
for sp in ['top', 'bottom', 'left', 'right']:
    ax.spines[sp].set_linewidth(bwith)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'K2_darcy_vs_stokes.png'), dpi=150)
plt.show()
 
i02 = np.argmin(np.abs(pg - 0.2)); i03 = np.argmin(np.abs(pg - 0.3))
print(f'  Stokes/Darcy 比值：phi=0.2 -> {v_sg[i02]/v_dg[i02]:.1f}x，'
      f'phi=0.3 -> {v_sg[i03]/v_dg[i03]:.1f}x  '
      f'(門檻附近同量級 -> 結論不依賴框架選擇)')
 

def q_(x, name, u=''):
    a5, a50, a95 = np.nanpercentile(x, [5, 50, 95])
    print(f'  {name:20s}{a50:12.4g}  [{a5:.4g}, {a95:.4g}] {u}')
q_(Re, 'Reynolds')
print(f'  P(Re > 0.1)           = {100*np.nanmean(Re > 0.1):.2f}%   '
      f'(Re ~ 1 要改用 Schiller-Naumann 修正)')

print(f'\nSEGREGATION (layered, n={int(m.sum())}, rho_S={RHO_S_CONST})')
q_(phi_l, 'phi = f_liquid'); q_(rL, 'rho_L', 'g/cc'); q_(drho, 'drho', 'kg/m3')
q_(dc, 'delta_c', 'm'); q_(v_stk*SEC_YR, 'v_settle', 'm/yr'); q_(t_stk, 't_settle', 'yr')
print(f'  P(phi > 0.40)         = {100*np.nanmean(phi_l > 0.40):.1f}%  (suspension -> Stokes)')
print(f'  P(phi < 0.25)         = {100*np.nanmean(phi_l < 0.25):.1f}%  (matrix -> Darcy)')
print(f'  P(D/delta_c < 1)      = {100*np.nanmean(D_km*1e3/dc < 1):.1f}%')
print(f'  P(t_settle > 4.5 Gyr) = {100*np.nanmean(t_stk > 4.5e9):.2f}%')
print(f'  P(drho <= 0)          = {100*np.nanmean(drho <= 0):.2f}%')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(phi_l, bins=40, color=STATE_COLOR['layered'])
axes[0].axvspan(0.25, 0.40, color='k', alpha=.15)
axes[0].set_xlabel(r'$\phi=f_{liquid}$'); axes[0].set_ylabel('count')
axes[1].hist(drho[np.isfinite(drho)], bins=40, color='#35838D')
axes[1].axvline(0, color='k', lw=2)
axes[1].set_xlabel(r'$\Delta\rho$ (kg/m$^3$)  (>0 = melt sinks)')
for a in axes: a.tick_params(labelsize=14)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'K_segregation.png'), dpi=150)
plt.show()


nm, npz_ = post['tt_n_miss'], post['tt_n_ph']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ok_ = np.isfinite(nm)
vals = np.unique(nm[ok_]).astype(int)
for s, c in STATE_COLOR.items():
    m_ = (state == s) & ok_
    if m_.sum():
        axes[0].hist(nm[m_], bins=np.arange(vals.min()-.5, vals.max()+1.5),
                     histtype='step', lw=2, color=c, label=f'{s} (n={int(m_.sum())})')
axes[0].axvline(2, color='k', ls='--', lw=2, label='healthy = 2')
axes[0].set_xlabel('tt_n_miss  (phases the model could not predict)')
axes[0].set_ylabel('count'); axes[0].set_yscale('log'); axes[0].legend(fontsize=10)
 
m_ = ok_ & np.isfinite(post['misfit_tt'])
sc = axes[1].scatter(nm[m_], post['misfit_tt'][m_], s=6, alpha=.3,
                     c=post['P_lit'][m_], cmap='viridis')
plt.colorbar(sc, ax=axes[1], label='P_lit (GPa)')
axes[1].set_xlabel('tt_n_miss'); axes[1].set_ylabel('TT misfit')
for a_ in axes:
    a_.tick_params(labelsize=14)
    for sp in ['top', 'bottom', 'left', 'right']:
        a_.spines[sp].set_linewidth(bwith)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'M_tt_miss.png'), dpi=150)
plt.show()
 
print(f'\ntt_n_miss: median={np.nanmedian(nm):.0f}  '
      f'P(=2)={100*np.nanmean(nm[ok_]==2):.1f}%  max={np.nanmax(nm[ok_]):.0f}')
print(f'tt_n_ph  : median={np.nanmedian(npz_):.0f}  (資料總數 72)')
print('   缺相集中在薄熱岩石圈（LVZ 陰影帶）-> 看右圖 P_lit 的顏色分佈')
 