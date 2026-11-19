# 51.check_Ra.py
# BML thermal-state diagnostic for old chain.json files (accepted steps only):
# Rayleigh number of the whole BML, convective vs conductive, and T_bml vs solidus.
# Usage: python 51.check_Ra.py chain_a21_00/chain.json chain_a21_01/chain.json ...
#        (no arguments: all files matching CHAIN_GLOB)
import json, glob, sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

CHAIN_GLOB = '/Users/chingchen/Desktop/HeFESTo/chain_json/*/chain.json'

TRUE_CMB   = 1743.3   # km
R          = 8.314
ADIAB_DTDP = 15.0     # K/GPa, mantle adiabat dT/dP

# Ra = alpha * rho * g * dT * h^3 / (kappa * eta)
ALPHA = 2e-5; RHO = 3700.0; G = 3.72; KAPPA = 1e-6
RA_C  = 1000.0

# Arrhenius: eta = eta0 * exp(E*/R * (1/T - 1/T0))
ETA0  = 1e21; T0 = 1600.0; ESTAR = 300e3

# adiabatic gradient in solid: dT/dz = alpha * T * g / Cp
CP = 1200.0

# ── read chain files (accepted steps) ────────────────────────────────────────
paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob(CHAIN_GLOB)
all_p = []
for path in paths:
    for s in json.load(open(path)):
        if s.get('accepted'):
            all_p.append(s['params'])
print(f"Total accepted steps: {len(all_p)}")

T_lit = np.array([p['T_lit'] for p in all_p])
P_lit = np.array([p['P_lit'] for p in all_p])
T_bml = np.array([p['T_bml'] for p in all_p])           # MCMC free param = CMB temperature
Mg    = np.array([p['Mg#_bml'] for p in all_p])
thick = np.array([p['BML_thickness'] for p in all_p])   # km

# geometry: P = 0.01135 GPa/km * depth
bml_top = TRUE_CMB - thick
P_top   = 0.01135 * bml_top
P_mid   = 0.01135 * (bml_top + TRUE_CMB) / 2

# T at BML top from the mantle adiabat
T_top = T_lit + ADIAB_DTDP * (P_top - P_lit)

# solidus at P_mid: Duncan 2018 (deg C) + Fe correction -6 K per Mg# percent below 0.75
# P <= 10 : -4.877 P^2 + 120.2 P + 1088
# P <= 23 : -1.323 (P-10)^2 + 38.18 (P-10) + 1802
# P >  23 : 77.75 (P-23) + 2075
P = P_mid
TC = np.where(P <= 10, -4.877*P**2 + 120.2*P + 1088,
     np.where(P <= 23, -1.323*(P-10)**2 + 38.18*(P-10) + 1802,
                       77.75*(P-23) + 2075))
Fe_corr = -6.0 * (100*(1-Mg) - 100*(1-0.75))
T_sol   = TC + 273.15 + Fe_corr

# dT across the whole BML, viscosity at T_top, Ra for the whole BML thickness
dT  = np.maximum(T_bml - T_top, 0)
eta = ETA0 * np.exp(ESTAR/R * (1/np.maximum(T_top, 400) - 1/T0))
h   = thick * 1000   # m
Ra  = ALPHA * RHO * G * dT * h**3 / (KAPPA * eta)
conv = Ra > RA_C

# T at the solid/liquid interface (melting where T > T_sol):
# convective: T_top + adiabatic rise alpha*T_top*g/Cp * thick ; conductive: T_bml (bottom)
T_int = np.where(conv, T_top + ALPHA * T_top * G / CP * thick, T_bml)
melt  = T_int > T_sol

# ── statistics ───────────────────────────────────────────────────────────────
n = len(Ra)
print(f"  E* = {ESTAR/1000:.0f} kJ/mol   Ra_c = {RA_C:.0f}")
print(f"  Convective  : {conv.sum()}/{n} = {100*conv.mean():.1f}%")
print(f"  Conductive  : {(~conv).sum()}/{n} = {100*(~conv).mean():.1f}%")
print(f"  Melting     : {melt.sum()}/{n} = {100*melt.mean():.1f}%")
print(f"    conv+melt : {(conv & melt).sum()}")
print(f"    cond+melt : {(~conv & melt).sum()}")
print()
print(f"  T_bml median       : {np.median(T_bml):.0f} K")
print(f"  T_sol median       : {np.median(T_sol):.0f} K")
print(f"  T_bml - T_sol      : {np.median(T_bml - T_sol):.0f} K  (median)")
print(f"  T_top median       : {np.median(T_top):.0f} K")
print(f"  % T_bml > T_sol    : {100*np.mean(T_bml > T_sol):.1f}%")
print(f"  Ra median          : {np.median(Ra):.2e}")

# ── figure ───────────────────────────────────────────────────────────────────
log_Ra = np.log10(np.maximum(Ra, 1))
sh     = T_bml - T_sol    # CMB superheat (>0 = BML bottom above solidus)
lim    = [1800, 3600]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Ra histogram
ax = axes[0, 0]
ax.hist(log_Ra[conv],  bins=40, color='steelblue', alpha=0.7, label=f'convective ({conv.sum()})')
ax.hist(log_Ra[~conv], bins=40, color='tomato',    alpha=0.7, label=f'conductive ({(~conv).sum()})')
ax.axvline(np.log10(RA_C), color='k', ls='--', lw=2, label=f'Ra_c = {RA_C:.0f}')
ax.set_xlabel('log10(Ra), whole BML', fontsize=11)

# 2. T_bml vs T_sol (above line = CMB hotter than solidus)
ax = axes[0, 1]
ax.scatter(T_sol[conv],  T_bml[conv],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(T_sol[~conv], T_bml[~conv], s=5, c='tomato',    alpha=0.4, label='conductive')
ax.plot(lim, lim, 'k--', lw=1.5, label='T_bml = T_sol')
ax.set_xlabel('T_solidus (K)', fontsize=11)
ax.set_ylabel('T_bml / T_core (K)', fontsize=11)

# 3. superheat
ax = axes[0, 2]
ax.hist(sh[conv],  bins=40, color='steelblue', alpha=0.7, label='convective')
ax.hist(sh[~conv], bins=40, color='tomato',    alpha=0.7, label='conductive')
ax.axvline(0, color='k', ls='--', lw=2, label='T_bml = T_sol')
ax.set_xlabel('T_bml - T_solidus (K)', fontsize=11)

# 4. BML_thickness vs superheat, coloured by log10(Ra)
ax = axes[1, 0]
sc = ax.scatter(thick, sh, c=log_Ra, cmap='RdBu_r', s=8, alpha=0.6,
                vmin=np.log10(RA_C)-2, vmax=np.log10(RA_C)+2)
plt.colorbar(sc, ax=ax, label='log10(Ra)')
ax.axhline(0, color='k', ls='--', lw=1.5)
ax.set_xlabel('BML_thickness (km)', fontsize=11)
ax.set_ylabel('T_bml - T_sol (K)', fontsize=11)

# 5. Mg#_bml vs T_sol
ax = axes[1, 1]
ax.scatter(Mg[conv],  T_sol[conv],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(Mg[~conv], T_sol[~conv], s=5, c='tomato',    alpha=0.4, label='conductive')
ax.set_xlabel('Mg#_bml', fontsize=11)
ax.set_ylabel('T_solidus (K)', fontsize=11)

# 6. T_top vs T_sol (does the mantle adiabat reach the solidus?)
ax = axes[1, 2]
ax.scatter(T_sol[conv],  T_top[conv],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(T_sol[~conv], T_top[~conv], s=5, c='tomato',    alpha=0.4, label='conductive')
ax.plot(lim, lim, 'k--', lw=1.5)
ax.set_xlabel('T_solidus (K)', fontsize=11)
ax.set_ylabel('T_mantle_bottom (K)', fontsize=11)

for ax in axes.flat:
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
