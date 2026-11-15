#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:51:24 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
BML 熱狀態診斷
用法: python diagnose_bml.py chain_a21_00/chain.json chain_a21_01/chain.json ...
"""
import json, glob, sys
import numpy as np
import matplotlib.pyplot as plt

# ── 物理常數 ──────────────────────────────────────────────────
TRUE_CMB   = 1743.3   # km
R          = 8.314
ADIAB_DTDP = 15.0     # K/GPa，mantle adiabat 的 dT/dP

# Ra = α·ρ·g·ΔT·h³ / (κ·η)
ALPHA = 2e-5; RHO = 3700.0; G = 3.72; KAPPA = 1e-6
RA_C  = 1000.0

# Arrhenius: η = η0·exp(E*/R·(1/T - 1/T0))
ETA0  = 1e21; T0 = 1600.0; ESTAR = 300e3

# adiabatic gradient in solid: dT/dz = α·T·g/Cp (K/km)
CP = 1200.0

# ── 讀 chain files ────────────────────────────────────────────
paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob('/Users/chingchen/Desktop/HeFESTo/chain_json/*/chain.json')
all_p, all_m = [], []
for path in paths:
    chain = json.load(open(path))
    for s in chain:
        if s.get('accepted'):
            all_p.append(s['params'])
            all_m.append(s['misfit'])
print(f"Total accepted steps: {len(all_p)}")

# ── 每個 step 的計算 ──────────────────────────────────────────
Ra_arr    = []
eta_arr   = []
dT_arr    = []
T_top_arr = []
T_bml_arr = []
T_sol_arr = []
T_liq_arr = []
conv_arr  = []
melt_arr  = []
thick_arr = []
Mg_arr    = []
superheat_arr = []

for p in all_p:
    T_lit = p['T_lit'];  P_lit = p['P_lit']
    T_bml = p['T_bml'];  Mg    = p['Mg#_bml']
    thick = p['BML_thickness']

    # geometry
    bml_top = TRUE_CMB - thick
    P_top   = 0.01135 * bml_top    # GPa
    P_mid   = 0.01135 * (bml_top + TRUE_CMB) / 2

    # T at BML top from mantle adiabat
    T_top = T_lit + ADIAB_DTDP * (P_top - P_lit)

    # solidus(P, Mg#) — Duncan 2018 + Fe correction
    P = P_mid
    if   P <= 10:  TC = -4.877*P**2 + 120.2*P + 1088
    elif P <= 23:  TC = -1.323*(P-10)**2 + 38.18*(P-10) + 1802
    else:          TC = 77.75*(P-23) + 2075
    Fe_corr = -6.0 * (100*(1-Mg) - 100*(1-0.75))
    T_sol   = TC + 273.15 + Fe_corr

    # liquidus (Ruedas 2017 + Fe correction)
    T_liq = 2160.6 + 64.7109*P - 3.97463*P**2 + 0.0957894*P**3 + Fe_corr

    # ΔT across whole BML (T_bml is the MCMC free param = CMB temperature)
    dT = max(T_bml - T_top, 0)

    # viscosity at T_top
    eta = ETA0 * np.exp(ESTAR/R * (1/max(T_top, 400) - 1/T0))

    # Ra for the whole BML thickness
    h   = thick * 1000   # m
    Ra  = ALPHA * RHO * G * dT * h**3 / (KAPPA * eta)

    is_conv = Ra > RA_C

    # T at solid/liquid interface
    # convective: T rises along adiabat from T_top
    # conductive: T rises linearly → T_bml at bottom
    # In both cases, melting starts where T > T_sol
    # For convective case: T_interface = T_top + adiabatic rise
    # For conductive case: T_interface = T_bml (bottom)
    if is_conv:
        dT_adiab  = ALPHA * T_top * G / CP * thick   # K over h_solid km
        T_int = T_top + dT_adiab
    else:
        T_int = T_bml   # bottom of BML = T_bml

    melting   = T_int > T_sol
    superheat = T_int - T_sol

    Ra_arr.append(Ra);      eta_arr.append(eta)
    dT_arr.append(dT);      T_top_arr.append(T_top)
    T_bml_arr.append(T_bml);T_sol_arr.append(T_sol)
    T_liq_arr.append(T_liq);conv_arr.append(is_conv)
    melt_arr.append(melting);thick_arr.append(thick)
    Mg_arr.append(Mg);      superheat_arr.append(superheat)

Ra_arr    = np.array(Ra_arr)
T_top_arr = np.array(T_top_arr)
T_bml_arr = np.array(T_bml_arr)
T_sol_arr = np.array(T_sol_arr)
T_liq_arr = np.array(T_liq_arr)
dT_arr    = np.array(dT_arr)
thick_arr = np.array(thick_arr)
Mg_arr    = np.array(Mg_arr)
superheat_arr = np.array(superheat_arr)
conv_arr  = np.array(conv_arr)
melt_arr  = np.array(melt_arr)

# ── 統計 ─────────────────────────────────────────────────────
n = len(Ra_arr)
print(f"  Convective  : {conv_arr.sum()}/{n} = {100*conv_arr.mean():.1f}%")
print(f"  Conductive  : {(~conv_arr).sum()}/{n} = {100*(~conv_arr).mean():.1f}%")
print(f"  Melting     : {melt_arr.sum()}/{n} = {100*melt_arr.mean():.1f}%")
print(f"    conv+melt : {(conv_arr & melt_arr).sum()}")
print(f"    cond+melt : {(~conv_arr & melt_arr).sum()}")
print()
print(f"  T_bml median       : {np.median(T_bml_arr):.0f} K")
print(f"  T_sol median       : {np.median(T_sol_arr):.0f} K")
print(f"  T_bml - T_sol      : {np.median(T_bml_arr - T_sol_arr):.0f} K  (median)")
print(f"  T_top median       : {np.median(T_top_arr):.0f} K")
print(f"  % T_bml > T_sol    : {100*np.mean(T_bml_arr > T_sol_arr):.1f}%")
print(f"  Ra median          : {np.median(Ra_arr):.2e}")

# ── 圖 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Ra histogram
ax = axes[0, 0]
log_Ra = np.log10(np.maximum(Ra_arr, 1))
ax.hist(log_Ra[conv_arr],  bins=40, color='steelblue', alpha=0.7, label=f'convective ({conv_arr.sum()})')
ax.hist(log_Ra[~conv_arr], bins=40, color='tomato',    alpha=0.7, label=f'conductive ({(~conv_arr).sum()})')
ax.axvline(np.log10(RA_C), color='k', ls='--', lw=2, label=f'Ra_c = {RA_C:.0f}')
ax.set_xlabel('log₁₀(Ra)', fontsize=11)
ax.set_title('Rayleigh Number\n(whole BML)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 2. T_bml vs T_sol（核心問題圖）
ax = axes[0, 1]
ax.scatter(T_sol_arr[conv_arr],  T_bml_arr[conv_arr],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(T_sol_arr[~conv_arr], T_bml_arr[~conv_arr], s=5, c='tomato',    alpha=0.4, label='conductive')
lim = [1800, 3600]
ax.plot(lim, lim, 'k--', lw=1.5, label='T_bml = T_sol')
ax.set_xlabel('T_solidus (K)', fontsize=11)
ax.set_ylabel('T_bml / T_core (K)', fontsize=11)
ax.set_title('T_bml vs T_solidus\n(above line = CMB hotter than solidus)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
frac = np.mean(T_bml_arr > T_sol_arr)
ax.text(0.05, 0.95, f'{100*frac:.1f}% T_bml > T_sol',
        transform=ax.transAxes, fontsize=10, va='top',
        color='darkgreen' if frac > 0.3 else 'darkred')

# 3. Superheat = T_bml - T_sol
ax = axes[0, 2]
sh = T_bml_arr - T_sol_arr
ax.hist(sh[conv_arr],  bins=40, color='steelblue', alpha=0.7, label='convective')
ax.hist(sh[~conv_arr], bins=40, color='tomato',    alpha=0.7, label='conductive')
ax.axvline(0, color='k', ls='--', lw=2, label='T_bml = T_sol')
ax.set_xlabel('T_bml − T_solidus (K)', fontsize=11)
ax.set_title('CMB superheat\n(>0 = BML bottom above solidus)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 4. BML_thickness vs T_bml - T_sol
ax = axes[1, 0]
sc = ax.scatter(thick_arr, sh, c=np.log10(np.maximum(Ra_arr, 1)),
                cmap='RdBu_r', s=8, alpha=0.6,
                vmin=np.log10(RA_C)-2, vmax=np.log10(RA_C)+2)
plt.colorbar(sc, ax=ax, label='log₁₀(Ra)')
ax.axhline(0, color='k', ls='--', lw=1.5)
ax.set_xlabel('BML_thickness (km)', fontsize=11)
ax.set_ylabel('T_bml − T_sol (K)', fontsize=11)
ax.set_title('Parameter space\nBlue=convective  Red=conductive', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# 5. Mg#_bml vs T_sol
ax = axes[1, 1]
ax.scatter(Mg_arr[conv_arr],  T_sol_arr[conv_arr],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(Mg_arr[~conv_arr], T_sol_arr[~conv_arr], s=5, c='tomato',    alpha=0.4, label='conductive')
ax.set_xlabel('Mg#_bml', fontsize=11)
ax.set_ylabel('T_solidus (K)', fontsize=11)
ax.set_title('Mg#_bml vs T_solidus\n(Fe-rich → lower solidus)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 6. T_top vs T_sol
ax = axes[1, 2]
ax.scatter(T_sol_arr[conv_arr],  T_top_arr[conv_arr],  s=5, c='steelblue', alpha=0.4, label='convective')
ax.scatter(T_sol_arr[~conv_arr], T_top_arr[~conv_arr], s=5, c='tomato',    alpha=0.4, label='conductive')
ax.plot(lim, lim, 'k--', lw=1.5)
ax.set_xlabel('T_solidus (K)', fontsize=11)
ax.set_ylabel('T_mantle_bottom (K)', fontsize=11)
ax.set_title('T_mantle_bottom vs T_solidus\n(check if mantle adiabat reaches solidus)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle(
    f'BML Thermal Diagnostic  |  n={n} accepted steps\n'
    f'Convective: {100*conv_arr.mean():.1f}%  |  '
    f'T_bml > T_sol: {100*np.mean(T_bml_arr>T_sol_arr):.1f}%  |  '
    f'E*={ESTAR/1000:.0f} kJ/mol  Ra_c={RA_C:.0f}',
    fontsize=11, fontweight='bold')

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/bml_diagnostic.png', dpi=150, bbox_inches='tight')
#print("Saved: bml_diagnostic.png")