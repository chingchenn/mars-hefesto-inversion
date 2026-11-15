#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:11:55 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
BML Concept Figures v3
======================
Correct physics:
  solid layer : linear T (conduction), compute phi at each point
                when phi >= 0.30 → interface found
  liquid layer: isothermal at T_interface (convection)
                T_interface = T_core = free parameter
  interface depth is determined by where linear T crosses solidus
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

# ═══════════════════════════════════════════════════════════════
# PLOT FLAGS
# ═══════════════════════════════════════════════════════════════
PLOT_FIG1 = True
PLOT_FIG4 = False

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════
MARS_RADIUS    = 3389.5
TRUE_CMB_DEPTH = 1743.3
T_SURF         = 220.0
T_lit          = 1539.0
P_lit          = 3.69
BML_THICKNESS  = 150.0

# ★ T_core = T_interface = free parameter
# This sets the isothermal temperature of the liquid layer
# Interface depth is determined by where linear T reaches solidus(mg_bml_solid)
T_core         = 2900.0

mg_mantle      = 0.78
mg_bml_solid   = 0.72     # free parameter (MCMC)
mg_bml_bulk    = 0.62     # fixed from literature

MG_DUNCAN_REF  = 0.75
RHO_MANTLE     = 4200.0
RHO_SOLID_BML  = 3600.0
RHO_LIQUID_BML = 3200.0
G_MARS         = 3.72
DTDP_ADIABAT   = 23.0
PHI_THRESHOLD  = 0.30

T_core_values = [2000, 2200, 2400, 2600, 2800, 3000, 3200]

bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000

# ═══════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════

def depth_to_P(d_km):
    return d_km * 1e3 * RHO_MANTLE * G_MARS / 1e9

def solidus_duncan2018(P):
    P = np.atleast_1d(np.asarray(P, float))
    T_C = np.where(P <= 10,
                   -4.877*P**2 + 120.2*P + 1088.0,
          np.where(P <= 23,
                   -1.323*(P-10)**2 + 38.18*(P-10) + 1802.0,
                   77.75*(P-23) + 2075.0))
    return T_C + 273.15

def liquidus_ruedas2017(P):
    P = np.atleast_1d(np.asarray(P, float))
    return 2160.6 + 64.7109*P - 3.97463*P**2 + 0.0957894*P**3

def fe_correction(Mg):
    return -6.0 * (100*(1-Mg) - 100*(1-MG_DUNCAN_REF))

def solidus(P, Mg):
    return solidus_duncan2018(P).flatten() + fe_correction(Mg)

def liquidus(P, Mg):
    return liquidus_ruedas2017(P).flatten() + fe_correction(Mg)

def ref_velocities(d_km):
    Vp0 = 13.7 - 0.5 * (d_km / 3000.0)
    Vs0 =  7.3 - 0.3 * (d_km / 3000.0)
    return Vp0, Vs0

def apply_melt_correction_pierru(Vp, Vs, phi):
    phi_nodes = np.array([0.0, 0.10, 0.20, 0.30, 1.0])
    dVp_nodes = np.array([0.0, 0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0, 0.20, 0.40,  1.00, 1.00])
    dVp = np.interp(phi, phi_nodes, dVp_nodes)
    dVs = np.interp(phi, phi_nodes, dVs_nodes)
    return Vp*(1-dVp), np.maximum(Vs*(1-dVs), 0.0)

def compute_liquid_mgnum(mg_bulk, mg_solid, h_solid, h_liquid,
                          rho_solid=RHO_SOLID_BML, rho_liquid=RHO_LIQUID_BML):
    M_solid  = rho_solid  * h_solid
    M_liquid = rho_liquid * h_liquid
    M_total  = M_solid + M_liquid
    if M_liquid <= 0:
        return mg_bulk
    mg_liq = (mg_bulk * M_total - mg_solid * M_solid) / M_liquid
    return float(np.clip(mg_liq, 0.01, mg_solid))

# ═══════════════════════════════════════════════════════════════
# ★ CORRECT temperature + phi profile builder
# ═══════════════════════════════════════════════════════════════

def build_profile(T_core_val, mg_solid=mg_bml_solid,
                  mg_bulk=mg_bml_bulk, n_bml=120):
    """
    Solid layer:
      - T increases linearly from T_mantle_bottom toward T_core_val
      - phi computed at each point using mg_solid
      - when phi >= PHI_THRESHOLD: interface found, T stops rising

    Liquid layer:
      - T = T_core_val (isothermal, convection)
      - phi computed using mg_liquid (from mass conservation)
    """
    # lithosphere
    d_lith = np.linspace(0, depth_lit, 60)
    T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)

    # mantle adiabat
    d_man = np.linspace(depth_lit, bml_top_km, 150)
    P_man = depth_to_P(d_man)
    T_man = T_lit + DTDP_ADIABAT * (P_man - P_lit)
    T_mantle_bottom = float(T_man[-1])

    # BML grid
    d_bml = np.linspace(bml_top_km, TRUE_CMB_DEPTH, n_bml)
    P_bml = depth_to_P(d_bml)

    # ★ solid layer: scan downward with linear T
    #   T_linear goes from T_mantle_bottom to T_core_val
    T_linear = np.linspace(T_mantle_bottom, T_core_val, n_bml)

    # ★ interface detection uses mg_bulk (bulk composition decides when melting starts)
    idx_interface = n_bml  # default: no interface (all solid)
    for i in range(n_bml):
        T_sol_i = solidus(P_bml[i:i+1], mg_bulk)[0]
        T_liq_i = liquidus(P_bml[i:i+1], mg_bulk)[0]
        if T_liq_i <= T_sol_i:
            T_liq_i = T_sol_i + 200.0
        phi_i = float(np.clip(
            (T_linear[i] - T_sol_i) / (T_liq_i - T_sol_i), 0.0, 1.0))
        if phi_i >= PHI_THRESHOLD:
            idx_interface = i
            break

    # ★ liquid layer thickness → mg_liquid from mass conservation
    h_solid  = float(d_bml[idx_interface] - bml_top_km) if idx_interface < n_bml else BML_THICKNESS
    h_liquid = BML_THICKNESS - h_solid
    mg_liquid = compute_liquid_mgnum(mg_bulk, mg_solid, h_solid, h_liquid)

    # ★ build T array:
    #   solid part : T_linear[:idx_interface]
    #   liquid part: T_core_val (isothermal)
    T_bml = np.empty(n_bml)
    T_bml[:idx_interface]  = T_linear[:idx_interface]
    T_bml[idx_interface:]  = T_core_val   # ★ isothermal

    # ★ compute phi for full BML using correct Mg# per layer
    phi_bml = np.zeros(n_bml)
    for i in range(n_bml):
        mg_use = mg_liquid if i >= idx_interface else mg_solid
        T_sol_i = solidus(P_bml[i:i+1],  mg_use)[0]
        T_liq_i = liquidus(P_bml[i:i+1], mg_use)[0]
        if T_liq_i <= T_sol_i:
            T_liq_i = T_sol_i + 200.0
        phi_bml[i] = float(np.clip(
            (T_bml[i] - T_sol_i) / (T_liq_i - T_sol_i), 0.0, 1.0))

    d_interface = float(d_bml[idx_interface]) if idx_interface < n_bml else TRUE_CMB_DEPTH

    depth_full = np.concatenate([d_lith, d_man, d_bml])
    T_full     = np.concatenate([T_lith, T_man, T_bml])
    P_full     = depth_to_P(depth_full)

    return dict(
        d_lith=d_lith,  T_lith=T_lith,
        d_man=d_man,    T_man=T_man,
        d_bml=d_bml,    T_bml=T_bml,    phi_bml=phi_bml,
        T_mantle_bottom=T_mantle_bottom,
        d_interface=d_interface,
        idx_interface=idx_interface,
        mg_liquid=mg_liquid,
        h_solid=h_solid, h_liquid=h_liquid,
        depth_full=depth_full, T_full=T_full, P_full=P_full,
    )

# ── default profile ──────────────────────────────────────────
prof = build_profile(T_core)
print(f"  mg_bml_solid  = {mg_bml_solid:.3f}")
print(f"  mg_bml_bulk   = {mg_bml_bulk:.3f}  (fixed)")
print(f"  mg_bml_liquid = {prof['mg_liquid']:.3f}  (mass conservation)")
print(f"  h_solid   = {prof['h_solid']:.1f} km")
print(f"  h_liquid  = {prof['h_liquid']:.1f} km")
print(f"  d_interface = {prof['d_interface']:.1f} km")
print(f"  T_core = T_interface = {T_core:.0f} K")

depth_full  = prof['depth_full']
T_full      = prof['T_full']
P_full      = prof['P_full']
mask_mantle = depth_full <= bml_top_km
mask_bml    = depth_full >  bml_top_km
mg_bml_liquid = prof['mg_liquid']

# ═══════════════════════════════════════════════════════════════
# AXIS HELPERS
# ═══════════════════════════════════════════════════════════════

def setup_ax(ax, xlim=(0, 3800)):
    ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
    ax.set_xlim(*xlim)
    ax.set_ylabel('Depth (km)', fontsize=14)
    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.25, lw=0.8)
    ax.tick_params(labelsize=12)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)

def draw_interfaces(ax, pr):
    ax.axhline(depth_lit,          color='gray',      ls=':', lw=1.3, alpha=0.7)
    ax.axhline(bml_top_km,         color='#7D6608',   ls=':', lw=1.8, alpha=0.9)
    ax.axhline(pr['d_interface'],  color='steelblue', ls='--', lw=1.8, alpha=0.85)
    ax.axhline(TRUE_CMB_DEPTH,     color='#922B21',   ls=':', lw=1.8, alpha=0.9)

# ═══════════════════════════════════════════════════════════════
# FIG 1
# ═══════════════════════════════════════════════════════════════
if PLOT_FIG1:
    fig1, ax = plt.subplots(figsize=(9, 11))
    setup_ax(ax)
    ax.set_yticks(np.arange(0, TRUE_CMB_DEPTH + 200, 200))

    cs = '#A52A2A'
    cl = '#E74C3C'

    # mantle solidus/liquidus
    ax.plot(solidus(P_full[mask_mantle],  mg_mantle), depth_full[mask_mantle],
            color=cs, lw=1.8, ls='-',  alpha=0.5)
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle],
            color=cl, lw=1.8, ls='--', alpha=0.5)

    P_bml_plot = P_full[mask_bml]
    d_bml_plot = depth_full[mask_bml]

    # BML solid solidus/liquidus
    ax.plot(solidus(P_bml_plot,  mg_bml_solid),  d_bml_plot,
            color=cs, lw=2.5, ls='-',
            label=f'Solidus solid BML  (Mg#={mg_bml_solid:.2f})')
    ax.plot(liquidus(P_bml_plot, mg_bml_solid),  d_bml_plot,
            color=cl, lw=2.5, ls='--',
            label=f'Liquidus solid BML (Mg#={mg_bml_solid:.2f})')

    # BML bulk solidus (used for interface detection)
    ax.plot(solidus(P_bml_plot, mg_bml_bulk), d_bml_plot,
            color='darkorange', lw=2.0, ls='-.',
            label=f'Solidus bulk BML  (Mg#={mg_bml_bulk:.2f}, interface detection)')

    # T profile
    ax.plot(prof['T_full'], prof['depth_full'],
            color='#2C3E50', lw=3.0, zorder=5, label='T profile')

    draw_interfaces(ax, prof)

    # region shading
    ax.axhspan(bml_top_km,           prof['d_interface'], alpha=0.07, color='#7D6608')
    ax.axhspan(prof['d_interface'],  TRUE_CMB_DEPTH,      alpha=0.10, color='steelblue')

    def rl(text, depth, color, dy=0):
        ax.text(1.01, depth+dy, text, transform=ax.get_yaxis_transform(),
                fontsize=10, color=color, va='center', ha='left')
    def ll(text, depth, color):
        ax.text(0.02, depth, text, transform=ax.get_yaxis_transform(),
                fontsize=10, color=color, va='center', fontweight='bold')

    rl(f'Lith base {depth_lit:.0f} km\nT={T_lit:.0f} K',
       depth_lit, 'gray', dy=30)
    rl(f'BML top {bml_top_km:.0f} km\nT={prof["T_mantle_bottom"]:.0f} K',
       bml_top_km, '#7D6608', dy=15)
    rl(f'φ=0.30 interface\n{prof["d_interface"]:.0f} km\nT=T_core={T_core:.0f} K',
       prof['d_interface'], 'steelblue', dy=20)
    rl(f'CMB {TRUE_CMB_DEPTH:.0f} km', TRUE_CMB_DEPTH, '#922B21', dy=-15)

    ll('Lithosphere\n(conduction)',
       depth_lit/2,                          '#1A5276')
    ll('Mantle\n(adiabat)',
       (depth_lit + bml_top_km)/2,            '#145A32')
    ll(f'BML solid\n(conduction)\nMg#={mg_bml_solid:.2f}',
       (bml_top_km + prof['d_interface'])/2,  '#7D6608')
    ll(f'BML liquid\n(isothermal)\nMg#={mg_bml_liquid:.2f}',
       (prof['d_interface'] + TRUE_CMB_DEPTH)/2, 'steelblue')

    ax.legend(fontsize=9, loc='center left', framealpha=0.9)
    ax.set_title(
        f'Two-layer BML  —  bulk Mg#={mg_bml_bulk:.2f} (fixed)  |  '
        f'solid Mg#={mg_bml_solid:.2f}  |  liquid Mg#={mg_bml_liquid:.2f} (mass consv.)\n'
        f'BML={BML_THICKNESS:.0f} km  |  '
        f'T_core = T_interface = {T_core:.0f} K (free parameter)\n'
        f'solid {prof["h_solid"]:.0f} km / liquid {prof["h_liquid"]:.0f} km',
        fontsize=10)

    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/fig1_v3.png', dpi=150, bbox_inches='tight')
    # print("Saved fig1")

# ═══════════════════════════════════════════════════════════════
# FIG 4
# ═══════════════════════════════════════════════════════════════
if PLOT_FIG4:
    fig4, (ax_T, ax_phi, ax_vel) = plt.subplots(
        1, 3, figsize=(21, 9), gridspec_kw={"wspace": 0.06})

    Y_TOP = 1400.0
    Y_BOT = TRUE_CMB_DEPTH + 30

    norm4 = Normalize(vmin=min(T_core_values), vmax=max(T_core_values))
    cmap4 = plt.get_cmap('Blues')
    cs = '#A52A2A'

    def setup_zoom(ax, xlabel, xlim=None):
        ax.set_ylim(Y_BOT, Y_TOP)
        ax.set_ylabel('Depth (km)', fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.grid(True, ls='--', alpha=0.25, lw=0.8)
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.tick_params(which='minor', length=3, color='gray')
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.axhline(bml_top_km,     color='#7D6608', ls=':', lw=1.8, alpha=0.9)
        ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)
        if xlim:
            ax.set_xlim(*xlim)

    # ── LEFT: temperature ────────────────────────────────────
    setup_zoom(ax_T, 'Temperature (K)', (1400, 3800))

    man_zoom = mask_mantle & (depth_full >= Y_TOP)
    ax_T.plot(solidus(P_full[man_zoom], mg_mantle), depth_full[man_zoom],
              color=cs, lw=1.5, ls='-', alpha=0.4,
              label=f'Sol mantle Mg#={mg_mantle:.2f}')
    ax_T.plot(solidus(P_full[mask_bml], mg_bml_solid),  depth_full[mask_bml],
              color=cs,        lw=2.2, ls='-',
              label=f'Sol BML solid  Mg#={mg_bml_solid:.2f}')
    ax_T.plot(solidus(P_full[mask_bml], mg_bml_bulk),   depth_full[mask_bml],
              color='darkorange', lw=2.0, ls='-.',
              label=f'Sol BML bulk   Mg#={mg_bml_bulk:.2f} (interface detection)')

    for T_c in T_core_values:
        col  = cmap4(norm4(T_c))
        pr_i = build_profile(T_c)
        d_i  = pr_i['depth_full']
        T_i  = pr_i['T_full']
        zoom = d_i >= Y_TOP
        ax_T.plot(T_i[zoom], d_i[zoom], color=col, lw=2.0, zorder=5)
        # mark interface with horizontal dashed line
        ax_T.axhline(pr_i['d_interface'], color=col, ls='--', lw=0.8, alpha=0.6)

    ax_T.plot(np.concatenate([prof['T_lith'], prof['T_man']]),
              np.concatenate([prof['d_lith'], prof['d_man']]),
              color='#2C3E50', lw=2.5, zorder=6, label='Mantle T (adiabat)')
    ax_T.legend(fontsize=9, loc='lower left', framealpha=0.9)

    # ── MIDDLE: melt fraction ────────────────────────────────
    setup_zoom(ax_phi, 'Melt fraction  φ', (-0.02, 1.05))
    ax_phi.yaxis.set_ticklabels([])
    ax_phi.set_ylabel('')
    ax_phi.axvline(PHI_THRESHOLD, color='steelblue', ls='--', lw=1.8,
                   label=f'φ={PHI_THRESHOLD}  solid/liquid interface')

    for T_c in T_core_values:
        col  = cmap4(norm4(T_c))
        pr_i = build_profile(T_c)
        ax_phi.plot(pr_i['phi_bml'], pr_i['d_bml'], color=col, lw=2.0, zorder=5)
        ax_phi.axhline(pr_i['d_interface'], color=col, ls='--', lw=0.8, alpha=0.6)

    ax_phi.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # ── RIGHT: velocity reduction ────────────────────────────
    setup_zoom(ax_vel, 'Velocity reduction (%)', (0, 105))
    ax_vel.yaxis.set_ticklabels([])
    ax_vel.set_ylabel('')
    ax_vel.axvline(10, color='gray', ls='--', lw=1.5, alpha=0.6, label='10% ref')

    for T_c in T_core_values:
        col  = cmap4(norm4(T_c))
        pr_i = build_profile(T_c)
        d_i  = pr_i['d_bml']
        phi_i = pr_i['phi_bml']
        Vp0_arr, Vs0_arr = ref_velocities(d_i)
        Vp_c, Vs_c = apply_melt_correction_pierru(Vp0_arr, Vs0_arr, phi_i)
        dVp = 100.0 * (Vp0_arr - Vp_c) / Vp0_arr
        dVs = 100.0 * (Vs0_arr - Vs_c) / Vs0_arr
        ax_vel.plot(dVp, d_i, color=col, lw=2.0)
        ax_vel.plot(dVs, d_i, color=col, lw=2.0, ls='--')

    ax_vel.plot([], [], color='gray', lw=2,      label='ΔVp (%)')
    ax_vel.plot([], [], color='gray', lw=2, ls='--', label='ΔVs (%)')
    ax_vel.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # colorbar
    sm = ScalarMappable(cmap='plasma', norm=norm4)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=[ax_T, ax_phi, ax_vel],
                         orientation='vertical', fraction=0.022,
                         pad=0.02, shrink=0.55)
    cbar.set_label('T_core = T_interface (K)', fontsize=12)
    cbar.set_ticks(T_core_values)
    cbar.ax.tick_params(labelsize=10)

    fig4.suptitle(
        f'BML Zoom — solid: linear T + phi scan  |  liquid: isothermal at T_core\n'
        f'Mantle Mg#={mg_mantle:.2f}  |  bulk Mg#={mg_bml_bulk:.2f} (fixed)  |  '
        f'solid Mg#={mg_bml_solid:.2f}  |  liquid Mg#={mg_bml_liquid:.2f}\n'
        f'T_core = T_interface = {min(T_core_values)}–{max(T_core_values)} K  '
        f'(free parameter)',
        fontsize=11)

    plt.savefig('/mnt/user-data/outputs/fig4_v3.png', dpi=150, bbox_inches='tight')
    print("Saved fig4")

plt.show()