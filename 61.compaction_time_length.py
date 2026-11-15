#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:06:50 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mckenzie_compaction.py  —  McKenzie (1984) 壓實計算

  k_phi   = a^2 phi^n / (C (1-phi)^2)              (4.4) Blake-Kozeny-Carman
  w0      = k/mu * (1-phi)/phi * drho * g          (3.7) minimum fluidization velocity
  delta_c = sqrt( k (zeta + 4 eta/3) / mu )        (3.3) compaction length
  tau_0   = delta_c / (w0 (1-phi))                 (B11) compaction timescale
  t_drain = H / w0                                 排空厚 H 的層
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ── 設定 ───────────────────────────────────────────────────────────────────────
OUT_DIR = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'

A_GRAIN = 1e-3          # m,     晶粒半徑
C_KC    = 1000.0        # Kozeny-Carman 常數（McKenzie 用 1000；Bear 教科書值 45）
N_PERM  = 3.0           # 滲透率指數（2 = 管狀孔道, 3 = 薄膜）
MU_MELT = 1.0           # Pa s,  熔體黏滯度
DRHO    = 700.0         # kg/m3, |rho_melt - rho_solid|
G_MARS  = 3.72          # m/s2
H_LAYER = 1e5           # m,     層厚（100 km）

ETA_LIST = [1e13, 1e15, 1e19, 1e21]   # Pa s，有效黏滯度 (zeta + 4 eta/3)
ZETA_DIV = False         # True: zeta = eta/phi 發散  /  False: ETA_LIST 直接當 (zeta+4eta/3)

PHI_TABLE = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3]

SEC_YR, AGE = 3.156e7, 4.5e9
bwith, fontsize = 1.5, 12
COLORS = ['#CD5C5C', '#35838D', '#414F67', '#7B9E87', '#C47F3E']

make_figure = 1
phi_crit_on = 1         # 額外算「臨界熔體分率」：低於此 4.5 Gyr 排不掉 → 永久殘留
# ──────────────────────────────────────────────────────────────────────────────


def k_phi(phi, a=A_GRAIN, C=C_KC, n=N_PERM):
    return a**2 * phi**n / (C * (1 - phi)**2)


def w0(phi, drho=DRHO, mu=MU_MELT, **kw):
    """(3.7) minimum fluidization velocity, m/s"""
    return k_phi(phi, **kw) / mu * (1 - phi) / phi * drho * G_MARS


def delta_c(phi, eta, mu=MU_MELT, **kw):
    """(3.3) compaction length, m.  ETA_LIST 是 eta 還是 (zeta+4eta/3) 由 ZETA_DIV 決定"""
    eta_b = eta * (1.0 / phi + 4.0 / 3.0) if ZETA_DIV else eta
    return np.sqrt(k_phi(phi, **kw) * eta_b / mu)


def tau0(phi, eta, **kw):
    """(B11) compaction timescale, yr"""
    return delta_c(phi, eta, **kw) / (w0(phi, **kw) * (1 - phi)) / SEC_YR


# ── 表 ────────────────────────────────────────────────────────────────────────
print(f'a = {A_GRAIN*1e3:.2f} mm   C = {C_KC:.0f}   n = {N_PERM:.0f}   '
      f'mu = {MU_MELT} Pa s   drho = {DRHO:.0f} kg/m3   H = {H_LAYER/1e3:.0f} km')
print(f'zeta = eta/phi (發散)' if ZETA_DIV else 'ETA_LIST 直接當 (zeta + 4eta/3)')

print('\n' + '=' * 92)
print(f"{'phi':>8}{'k (m2)':>12}{'w0 (m/yr)':>13}{'t_drain (yr)':>15}"
      + ''.join(f"{f'dc[{e:.0e}] (m)':>18}" for e in ETA_LIST))
print('=' * 92)
for p in PHI_TABLE:
    v = w0(p)
    row = ''.join(f'{delta_c(p, e):18.4g}' for e in ETA_LIST)
    print(f'{p:8.0e}{k_phi(p):12.3e}{v*SEC_YR:13.4g}{H_LAYER/v/SEC_YR:15.4g}' + row)

print('\n' + '=' * 92)
print('compaction timescale  tau_0 = delta_c / [w0 (1-phi)]   (yr)')
print('=' * 92)
print(f"{'phi':>8}" + ''.join(f"{f'eta={e:.0e}':>18}" for e in ETA_LIST))
for p in PHI_TABLE:
    print(f'{p:8.0e}' + ''.join(f'{tau0(p, e):18.4g}' for e in ETA_LIST))

print('\n判讀：')
print('  H / delta_c >> 1  ->  基質黏滯度不是瓶頸，排液由 Darcy 控制，用 t_drain')
print('  H / delta_c << 1  ->  整層都在壓實邊界層內，由基質控制')
print('  phi > ~0.2-0.4    ->  晶粒瓦解，本理論失效')
print(f'\n{"phi":>8}{"H/delta_c":>14}   (eta = {ETA_LIST[-1]:.0e})')
for p in PHI_TABLE:
    print(f'{p:8.0e}{H_LAYER/delta_c(p, ETA_LIST[-1]):14.4g}')


# ── 臨界熔體分率 ──────────────────────────────────────────────────────────────
def phi_crit(H=H_LAYER, drho=DRHO, mu=MU_MELT, t=AGE, **kw):
    """低於此熔體分率，熔體 4.5 Gyr 走不完 H → 永久殘留在基質裡。"""
    pp = np.logspace(-8, np.log10(0.45), 4000)
    return float(np.interp(H / (t * SEC_YR), w0(pp, drho=drho, mu=mu, **kw), pp))


if phi_crit_on:
    print('\n' + '=' * 92)
    print(f'臨界熔體分率 phi_crit   (H = {H_LAYER/1e3:.0f} km, t = 4.5 Gyr, C = {C_KC:.0f})')
    print('  低於此 -> 熔體永遠排不出去；這一部分就是地函能長期保留的殘留熔體')
    print('=' * 92)
    print(f"{'a (mm)':>9}" + ''.join(f"{f'mu={m} Pa s':>16}" for m in [0.05, 0.1, 1.0]))
    for a_ in [3e-4, 1e-3, 3e-3]:
        print(f'{a_*1e3:9.1f}' + ''.join(f'{phi_crit(a=a_, mu=m):16.2e}'
                                        for m in [0.05, 0.1, 1.0]))
    print(f"\n{'a (mm)':>9}{'drho=200':>16}{'drho=700':>16}   (mu = 1 Pa s, 近密度反轉點)")
    for a_ in [3e-4, 1e-3, 3e-3]:
        print(f'{a_*1e3:9.1f}' + ''.join(f'{phi_crit(a=a_, drho=d):16.2e}'
                                        for d in [200., 700.]))

# ── 圖 ────────────────────────────────────────────────────────────────────────
if make_figure:
    os.makedirs(OUT_DIR, exist_ok=True)
    phi = np.logspace(-6, np.log10(0.45), 400)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    for e, c in zip(ETA_LIST, COLORS):
        ax.loglog(phi, delta_c(phi, e), lw=2.4, color=c, label=fr'$\eta$={e:.0e} Pa s')
    ax.axhline(H_LAYER, color='k', ls='--', lw=1.8)
    ax.text(1.5e-6, H_LAYER*1.3, f'H = {H_LAYER/1e3:.0f} km', fontsize=10)
    ax.set_ylabel(r'compaction length $\delta_c$ (m)', fontsize=fontsize)
    ax.set_title('(3.3) compaction length', fontsize=fontsize)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.loglog(phi, w0(phi)*SEC_YR, lw=2.6, color=COLORS[0])
    ax.set_ylabel(r'$w_0$ (m/yr)', fontsize=fontsize)
    ax.set_title('(3.7) minimum fluidization velocity', fontsize=fontsize)

    ax = axes[2]
    for e, c in zip(ETA_LIST, COLORS):
        ax.loglog(phi, tau0(phi, e), lw=2.4, color=c, label=fr'$\tau_0$, $\eta$={e:.0e}')
    ax.loglog(phi, H_LAYER/w0(phi)/SEC_YR, lw=2.6, ls='--', color='k',
              label=f'$H/w_0$, H={H_LAYER/1e3:.0f} km')
    ax.axhline(AGE, color='grey', ls='-.', lw=1.8)
    ax.text(1.5e-6, AGE*1.6, 'age of Mars', fontsize=10)
    ax.set_ylabel('timescale (yr)', fontsize=fontsize)
    ax.set_title('(B11) compaction time vs drainage time', fontsize=fontsize)
    ax.legend(fontsize=8, loc='upper right')

    for ax in axes:
        ax.axvspan(0.2, 0.45, color='k', alpha=.12)
        ax.set_xlabel(r'melt fraction $\phi$', fontsize=fontsize)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=.3, which='both')
        for a in ['top', 'bottom', 'left', 'right']:
            ax.spines[a].set_linewidth(bwith)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mckenzie_compaction.png'), dpi=150)
    print('\nsaved mckenzie_compaction.png')

    if phi_crit_on:
        fig, ax = plt.subplots(figsize=(7, 5.6))
        a_arr = np.logspace(-4, -2, 60)
        for mu_, c in zip([0.05, 0.1, 1.0], COLORS):
            ax.loglog(a_arr*1e3, [phi_crit(a=a_, mu=mu_) for a_ in a_arr],
                      lw=2.4, color=c, label=fr'$\mu$={mu_} Pa s')
        ax.loglog(a_arr*1e3, [phi_crit(a=a_, drho=200.) for a_ in a_arr],
                  lw=2, ls='--', color='k',
                  label=r'$\mu$=1, $\Delta\rho$=200 kg m$^{-3}$')
        ax.set_xlabel('grain size $a$ (mm)', fontsize=fontsize)
        ax.set_ylabel(r'$\phi_{\rm crit}$   (trapped over 4.5 Gyr)', fontsize=fontsize)
        ax.set_title(f'residual melt that can never be expelled  '
                     f'(H = {H_LAYER/1e3:.0f} km)', fontsize=fontsize)
        ax.tick_params(labelsize=14); ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=10)
        for a in ['top', 'bottom', 'left', 'right']:
            ax.spines[a].set_linewidth(bwith)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'mckenzie_phi_crit.png'), dpi=150)
        print('saved mckenzie_phi_crit.png')


# ── 圖D: compaction 時標 ────────────────────────────────────────────────────

phi = np.logspace(-6, -0.3, 60)
a_gr, K_perm, mu = 1e-3, 1000.0, 1.0
drho, g_m, sec_yr = abs(4500.0 - 3800.0), 3.72, 3.156e7
fig, ax = plt.subplots(figsize=(8, 6))
for eta in [1e13, 1e15, 1e17, 1e19]:
    k = phi**3 * a_gr**2 / (K_perm * (1 - phi)**2)
    w0 = k / mu * (1 - phi) / phi * drho * g_m
    tau = np.sqrt(eta / mu * k) / (w0 * (1 - phi)) / sec_yr
    ax.loglog(phi, tau, lw=2, label=f'$\\eta_{{eff}}$={eta:.0e} Pa s')
    print(f"eta={eta:.0e}  tau(1e-6)={tau[0]:.2e}  tau(0.3)={tau[-1]:.2e} yr")
ax.axhline(4.5e9, color='k', ls='--', lw=1.8, label='age of Mars')
ax.set_xlabel('melt fraction $\\phi$'); ax.set_ylabel('compaction timescale (yr)')
ax.tick_params(labelsize=14); ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
   