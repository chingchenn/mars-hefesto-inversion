# 61.compaction_time_length.py
# McKenzie (1984) compaction scales for melt in a Mars mantle / BML layer.
#   k_phi   = a^2 phi^n / (C (1-phi)^2)              (4.4) Blake-Kozeny-Carman permeability
#   w0      = k/mu * (1-phi)/phi * drho * g          (3.7) minimum fluidization velocity
#   delta_c = sqrt( k (zeta + 4 eta/3) / mu )        (3.3) compaction length
#   tau_0   = delta_c / (w0 (1-phi))                 (B11) compaction timescale
#   t_drain = H / w0                                 time to drain a layer of thickness H
#   phi_crit: w0(phi_crit) = H / age  -> below it, melt cannot leave H in 4.5 Gyr (trapped)
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

A_GRAIN = 1e-3          # m,     grain radius
C_KC    = 1000.0        # Kozeny-Carman constant (McKenzie uses 1000; Bear textbook value 45)
N_PERM  = 3.0           # permeability exponent (2 = tubes, 3 = films)
MU_MELT = 1.0           # Pa s,  melt viscosity
DRHO    = 700.0         # kg/m3, |rho_melt - rho_solid|
G_MARS  = 3.72          # m/s2
H_LAYER = 1e5           # m,     layer thickness (100 km)

ETA_LIST = [1e13, 1e15, 1e19, 1e21]   # Pa s, effective matrix viscosity (zeta + 4 eta/3)
ZETA_DIV = False        # True: zeta = eta/phi (diverges)  /  False: ETA_LIST used directly as (zeta + 4eta/3)

PHI_TABLE = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3]
A_LIST    = [3e-4, 1e-3, 3e-3]        # m, grain radii for the phi_crit table
MU_LIST   = [0.05, 0.1, 1.0]          # Pa s, melt viscosities for phi_crit

SEC_YR, AGE = 3.156e7, 4.5e9
bwith, fontsize = 1.5, 12
COLORS = ['#CD5C5C', '#35838D', '#414F67', '#7B9E87', '#C47F3E']


def k_phi(phi, a=A_GRAIN):
    return a**2 * phi**N_PERM / (C_KC * (1 - phi)**2)


def w0(phi, a=A_GRAIN, mu=MU_MELT, drho=DRHO):        # m/s
    return k_phi(phi, a) / mu * (1 - phi) / phi * drho * G_MARS


def delta_c(phi, eta):                                 # m
    eta_b = eta * (1.0 / phi + 4.0 / 3.0) if ZETA_DIV else eta
    return np.sqrt(k_phi(phi) * eta_b / MU_MELT)


def tau0(phi, eta):                                    # yr
    return delta_c(phi, eta) / (w0(phi) * (1 - phi)) / SEC_YR


def phi_crit(a=A_GRAIN, mu=MU_MELT, drho=DRHO):
    # invert w0(phi) = H / age on a log grid
    pp = np.logspace(-8, np.log10(0.45), 4000)
    return float(np.interp(H_LAYER / (AGE * SEC_YR), w0(pp, a, mu, drho), pp))


# ── tables ───────────────────────────────────────────────────────────────────
print(f'a = {A_GRAIN*1e3:.2f} mm   C = {C_KC:.0f}   n = {N_PERM:.0f}   '
      f'mu = {MU_MELT} Pa s   drho = {DRHO:.0f} kg/m3   H = {H_LAYER/1e3:.0f} km')
print('zeta = eta/phi (diverges)' if ZETA_DIV else 'ETA_LIST used directly as (zeta + 4eta/3)')

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

print('\nInterpretation:')
print('  H / delta_c >> 1  ->  matrix viscosity is not the bottleneck; Darcy flow controls, use t_drain')
print('  H / delta_c << 1  ->  whole layer is inside the compaction boundary layer; matrix controls')
print('  phi > ~0.2-0.4    ->  grains disaggregate, theory no longer valid')
print(f'\n{"phi":>8}{"H/delta_c":>14}   (eta = {ETA_LIST[-1]:.0e})')
for p in PHI_TABLE:
    print(f'{p:8.0e}{H_LAYER/delta_c(p, ETA_LIST[-1]):14.4g}')

print('\n' + '=' * 92)
print(f'critical melt fraction phi_crit   (H = {H_LAYER/1e3:.0f} km, t = 4.5 Gyr, C = {C_KC:.0f})')
print('  below it -> melt is never expelled; this is the residual melt the mantle can keep long term')
print('=' * 92)
print(f"{'a (mm)':>9}" + ''.join(f"{f'mu={m} Pa s':>16}" for m in MU_LIST))
for a_ in A_LIST:
    print(f'{a_*1e3:9.1f}' + ''.join(f'{phi_crit(a=a_, mu=m):16.2e}' for m in MU_LIST))
print(f"\n{'a (mm)':>9}{'drho=200':>16}{'drho=700':>16}   (mu = 1 Pa s, near density crossover)")
for a_ in A_LIST:
    print(f'{a_*1e3:9.1f}' + ''.join(f'{phi_crit(a=a_, drho=d):16.2e}' for d in [200., 700.]))

# ── Fig A: delta_c, w0, tau_0 vs phi ─────────────────────────────────────────
phi = np.logspace(-6, np.log10(0.45), 400)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

ax = axes[0]
for e, c in zip(ETA_LIST, COLORS):
    ax.loglog(phi, delta_c(phi, e), lw=2.4, color=c, label=fr'$\eta$={e:.0e} Pa s')
ax.axhline(H_LAYER, color='k', ls='--', lw=1.8, label=f'H = {H_LAYER/1e3:.0f} km')
ax.set_ylabel(r'compaction length $\delta_c$ (m)', fontsize=fontsize)
ax.set_title('(3.3) compaction length', fontsize=fontsize)
ax.legend(fontsize=9, loc='lower right')

ax = axes[1]
ax.loglog(phi, w0(phi) * SEC_YR, lw=2.6, color=COLORS[0])
ax.set_ylabel(r'$w_0$ (m/yr)', fontsize=fontsize)
ax.set_title('(3.7) minimum fluidization velocity', fontsize=fontsize)

ax = axes[2]
for e, c in zip(ETA_LIST, COLORS):
    ax.loglog(phi, tau0(phi, e), lw=2.4, color=c, label=fr'$\tau_0$, $\eta$={e:.0e}')
ax.loglog(phi, H_LAYER / w0(phi) / SEC_YR, lw=2.6, ls='--', color='k',
          label=f'$H/w_0$, H={H_LAYER/1e3:.0f} km')
ax.axhline(AGE, color='grey', ls='-.', lw=1.8, label='age of Mars')
ax.set_ylabel('timescale (yr)', fontsize=fontsize)
ax.set_title('(B11) compaction time vs drainage time', fontsize=fontsize)
ax.legend(fontsize=8, loc='upper right')

for ax in axes:
    ax.axvspan(0.2, 0.45, color='k', alpha=.12)     # phi > 0.2: theory breaks down
    ax.set_xlabel(r'melt fraction $\phi$', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.3, which='both')
    for a in ['top', 'bottom', 'left', 'right']:
        ax.spines[a].set_linewidth(bwith)
plt.tight_layout()

# ── Fig B: phi_crit vs grain size ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.6))
a_arr = np.logspace(-4, -2, 60)
for mu_, c in zip(MU_LIST, COLORS):
    ax.loglog(a_arr * 1e3, [phi_crit(a=a_, mu=mu_) for a_ in a_arr],
              lw=2.4, color=c, label=fr'$\mu$={mu_} Pa s')
ax.loglog(a_arr * 1e3, [phi_crit(a=a_, drho=200.) for a_ in a_arr],
          lw=2, ls='--', color='k', label=r'$\mu$=1, $\Delta\rho$=200 kg m$^{-3}$')
ax.set_xlabel('grain size $a$ (mm)', fontsize=fontsize)
ax.set_ylabel(r'$\phi_{\rm crit}$   (trapped over 4.5 Gyr)', fontsize=fontsize)
ax.tick_params(labelsize=14)
ax.grid(alpha=.3, which='both')
ax.legend(fontsize=10, loc='lower right')
for a in ['top', 'bottom', 'left', 'right']:
    ax.spines[a].set_linewidth(bwith)
plt.tight_layout()

# ── Fig C: compaction timescale, wider eta range ─────────────────────────────
# tau = sqrt(eta k / mu) / (w0 (1-phi)), eta used directly as (zeta + 4eta/3)
phi = np.logspace(-6, -0.3, 60)
fig, ax = plt.subplots(figsize=(8, 6))
for eta in [1e13, 1e15, 1e17, 1e19]:
    tau = np.sqrt(eta / MU_MELT * k_phi(phi)) / (w0(phi) * (1 - phi)) / SEC_YR
    ax.loglog(phi, tau, lw=2, label=f'$\\eta_{{eff}}$={eta:.0e} Pa s')
    print(f"eta={eta:.0e}  tau(1e-6)={tau[0]:.2e}  tau(0.3)={tau[-1]:.2e} yr")
ax.axhline(AGE, color='k', ls='--', lw=1.8, label='age of Mars')
ax.set_xlabel('melt fraction $\\phi$')
ax.set_ylabel('compaction timescale (yr)')
ax.tick_params(labelsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()
