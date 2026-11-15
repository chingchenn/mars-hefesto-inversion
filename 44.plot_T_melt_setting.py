#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:29:30 2026

@author: chingchen
"""

#!/usr/bin/env python3
"""
BML Concept Figures
===================
Three figures sharing the same physics functions.
Toggle which figures to plot with the flags below.

Fig 1 : Standard concept — fixed mg_mantle & mg_bml, text annotations
Fig 2 : Mg# variation   — rainbow colorbar over Mg# values
Fig 3 : T_bml variation — rainbow colorbar over T_bml values
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ═══════════════════════════════════════════════════════════════
# ★ PLOT FLAGS
# ═══════════════════════════════════════════════════════════════
PLOT_FIG1 = 1    # standard concept figure
PLOT_FIG2 = 0    # Mg# variation
PLOT_FIG3 = 0    # T_bml variation
PLOT_FIG4 = 1    # bml zoom-in

# ═══════════════════════════════════════════════════════════════
# PARAMETERS  (shared)
# ═══════════════════════════════════════════════════════════════
MARS_RADIUS    = 3389.5   # km
TRUE_CMB_DEPTH = 1743.3   # km
T_SURF         = 220.0    # K
T_lit          = 1539.0   # K  lithosphere/adiabat transition
P_lit          = 3.69     # GPa
BML_THICKNESS  = 250    # km
T_bml_default  = 2800.0   # K  BML bottom temperature (Fig 1 & 2)
mg_mantle      = 0.78     # mantle Mg#
mg_bml         = 0.65     # BML Mg#
MG_DUNCAN_REF  = 0.75
RHO_MANTLE     = 4200.0   # kg/m³  (lower mantle approx)
G_MARS         = 3.72     # m/s²
DTDP_ADIABAT   = 23.0     # K/GPa  mantle adiabat gradient

# Fig 2 — Mg# sweep
mg_values      = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Fig 3 — T_bml sweep
T_bml_values   = [2000, 2200, 2400, 2600, 2800, 3000, 3200]   # K

# derived
bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000   # km

# ═══════════════════════════════════════════════════════════════
# SHARED PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def depth_to_P(d_km):
    """Approximate pressure in GPa from depth in km."""
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

def apply_melt_correction_pierru(Vp, Vs, phi):

    phi_nodes = np.array([0.0, 0.10, 0.20, 0.30, 1.0])

    dVp_nodes = np.array([0.0, 0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0, 0.20, 0.40, 1.00, 1.00])

    dVp = np.interp(phi, phi_nodes, dVp_nodes)
    dVs = np.interp(phi, phi_nodes, dVs_nodes)

    Vp_corr = Vp * (1.0 - dVp)
    Vs_corr = Vs * (1.0 - dVs)

    Vs_corr = np.maximum(Vs_corr, 0.0)

    return Vp_corr, Vs_corr
# ═══════════════════════════════════════════════════════════════
# SHARED TEMPERATURE PROFILE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_T_profile(T_bml):
    """
    Build full depth/T arrays for a given T_bml.
    Returns: d_lith, d_man, d_bml, T_lith, T_man, T_bml_arr,
             T_mantle_bottom, depth_full, T_full, P_full
    """
    d_lith = np.linspace(0, depth_lit, 80)
    T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)

    d_man  = np.linspace(depth_lit, bml_top_km, 200)
    P_man  = depth_to_P(d_man)
    T_man  = T_lit + DTDP_ADIABAT * (P_man - P_lit)
    T_mantle_bottom = float(T_man[-1])

    d_bml     = np.linspace(bml_top_km, TRUE_CMB_DEPTH, 60)
    T_bml_arr = np.linspace(T_mantle_bottom, T_bml, len(d_bml))

    depth_full = np.concatenate([d_lith, d_man, d_bml])
    T_full     = np.concatenate([T_lith, T_man, T_bml_arr])
    P_full     = depth_to_P(depth_full)

    return (d_lith, d_man, d_bml,
            T_lith, T_man, T_bml_arr,
            T_mantle_bottom, depth_full, T_full, P_full)

# ── build default profile (used by Fig 1 & 2) ────────────────
(d_lith, d_man, d_bml,
 T_lith, T_man, T_bml_arr,
 T_mantle_bottom,
 depth_full, T_full, P_full) = build_T_profile(T_bml_default)

mask_mantle = depth_full <= bml_top_km
mask_bml    = depth_full >  bml_top_km

# ═══════════════════════════════════════════════════════════════
# SHARED AXIS SETUP
# ═══════════════════════════════════════════════════════════════

def setup_ax(ax):
    ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
    ax.set_xlim(0, 3800)
    ax.set_ylabel('Depth (km)', fontsize=14)
    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.25, lw=0.8)
    ax.tick_params(labelsize=12)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)

def draw_interfaces(ax):
    ax.axhline(depth_lit,      color='gray',    ls=':', lw=1.3, alpha=0.7)
    ax.axhline(bml_top_km,     color='#7D6608', ls=':', lw=1.8, alpha=0.9)
    ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)

def rlabel(ax, text, depth, color, dy=0):
    ax.text(1.01, depth+dy, text, transform=ax.get_yaxis_transform(),
            fontsize=12, color=color, va='center', ha='left')

def llabel(ax, text, depth_c, color):
    ax.text(0.02, depth_c, text, transform=ax.get_yaxis_transform(),
            fontsize=12, color=color, va='center', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# FIG 1 — Standard concept
# ═══════════════════════════════════════════════════════════════
if PLOT_FIG1:
    fig1, ax = plt.subplots(figsize=(8, 10))
    setup_ax(ax)
    ax.set_yticks(np.arange(0, TRUE_CMB_DEPTH + 200, 200))

    color_solidus = '#A52A2A'
    color_liquid = '#E74C3C'

    # solidus / liquidus — mantle part
    ax.plot(solidus(P_full[mask_mantle],  mg_mantle), depth_full[mask_mantle],
            color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle],
            color=color_liquid, lw=2.0, ls='--')

    # solidus / liquidus — BML part
    ax.plot(solidus(P_full[mask_bml],  mg_bml), depth_full[mask_bml],
            color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml],
            color=color_liquid, lw=2.0, ls='--')

    # faint mantle sol/liq extended into BML (reference)
    ax.plot(solidus(P_full[mask_bml],  mg_mantle), depth_full[mask_bml],
            color=color_solidus, lw=1.0, ls='-',  alpha=0.25)
    ax.plot(liquidus(P_full[mask_bml], mg_mantle), depth_full[mask_bml],
            color=color_liquid, lw=1.0, ls='--', alpha=0.25)

    # T profile
    ax.plot(T_full, depth_full, color='#2C3E50', lw=2.8, zorder=5)

    draw_interfaces(ax)

    # annotations
    rlabel(ax, f'Lithosphere base\n{depth_lit:.0f} km | {P_lit:.1f} GPa\nT={T_lit:.0f} K',
           depth_lit, 'gray', dy=40)
    rlabel(ax, f'BML top\n{bml_top_km:.0f} km\nT={T_mantle_bottom:.0f} K',
           bml_top_km, '#7D6608', dy=20)
    rlabel(ax, f'CMB\n{TRUE_CMB_DEPTH:.0f} km\nT_bml={T_bml_default:.0f} K',
           TRUE_CMB_DEPTH, '#922B21', dy=-25)

    llabel(ax, 'Lithosphere\n(conductive)',  depth_lit/2,                   '#1A5276')
    llabel(ax, 'Mantle adiabat\n(S=S_lit)', (depth_lit+bml_top_km)/3,      '#145A32')
    llabel(ax, 'BML\n(linear T)',            (bml_top_km+TRUE_CMB_DEPTH)/2, '#7D6608')

    
    ax.legend(handles=[
        Line2D([0],[0], color='#2C3E50', lw=2.8,
               label='T profile'),
        Line2D([0],[0], color=color_solidus, lw=2.0, ls='-',
               label=f'Solidus (Duncan 2018)'),
        Line2D([0],[0], color=color_liquid, lw=2.0, ls='--',
               label=f'Liquidus (Ruedas 2017) '),
       
    ], fontsize=10, loc='center left', framealpha=0.9)

    ax.set_title(
        f'Standard Concept\n'
        f'Mantle Mg#={mg_mantle:.2f}  |  BML Mg#={mg_bml:.2f}  |  '
        f'BML={BML_THICKNESS:.0f} km  |  T_bml={T_bml_default:.0f} K',
        fontsize=12)

    plt.tight_layout()
    

# ═══════════════════════════════════════════════════════════════
# FIG 2 — Mg# variation (rainbow)
# ═══════════════════════════════════════════════════════════════
if PLOT_FIG2:
    fig2, ax = plt.subplots(figsize=(8, 10))
    setup_ax(ax)

    mg_norm = Normalize(vmin=min(mg_values), vmax=max(mg_values))
    mg_cmap = plt.get_cmap('rainbow')

    for mm in mg_values:
        col = mg_cmap(mg_norm(mm))
        ax.plot(solidus(P_full,  mm), depth_full, color=col, lw=1.8, ls='-',  zorder=3)
        ax.plot(liquidus(P_full, mm), depth_full, color=col, lw=1.8, ls='--', zorder=3)

    # T profile (fixed)
    ax.plot(T_full, depth_full, color='#2C3E50', lw=2.8, zorder=5,
            label='T profile')
    draw_interfaces(ax)
    # colorbar
    sm = ScalarMappable(cmap='rainbow', norm=mg_norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax, orientation='vertical',
                         fraction=0.03, pad=0.12, shrink=0.35, label='Mg#')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(mg_values)
    cbar.set_ticklabels([f'{m:.2f}' for m in mg_values])
    ax.set_title(
        f'Mg# Variation\n'
        f'T profile fixed  |  BML={BML_THICKNESS:.0f} km  |  T_bml={T_bml_default:.0f} K',
        fontsize=10)

    plt.tight_layout()
    

# ═══════════════════════════════════════════════════════════════
# FIG 3 — T_bml variation (rainbow)
# ═══════════════════════════════════════════════════════════════
if PLOT_FIG3:
    fig3, ax = plt.subplots(figsize=(8, 10))
    
    setup_ax(ax)

    color_solidus = '#A52A2A'
    color_liquid = '#E74C3C'

    # solidus / liquidus — mantle part
    ax.plot(solidus(P_full[mask_mantle],  mg_mantle), depth_full[mask_mantle],
            color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle],
            color=color_liquid, lw=2.0, ls='--')

    # solidus / liquidus — BML part
    ax.plot(solidus(P_full[mask_bml],  mg_bml), depth_full[mask_bml],
            color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml],
            color=color_liquid, lw=2.0, ls='--')

    # rainbow T profiles
    tbml_norm = Normalize(vmin=min(T_bml_values), vmax=max(T_bml_values))
    tbml_cmap = plt.get_cmap('rainbow')

    for T_bml in T_bml_values:
        col = tbml_cmap(tbml_norm(T_bml))
        (_, _, _,
         _, _, _,
         _, d_full_i, T_full_i, _) = build_T_profile(T_bml)
        ax.plot(T_full_i, d_full_i, color=col, lw=2.2, zorder=5)

    # fixed mantle adiabat highlighted
    ax.plot(np.concatenate([T_lith, T_man]),
            np.concatenate([d_lith, d_man]),
            color='#2C3E50', lw=2.8, zorder=6,
            label='Mantle T (fixed adiabat)')

    draw_interfaces(ax)

    ax.axvline(T_mantle_bottom, color='#7D6608', ls=':', lw=1.2, alpha=0.6)
    ax.text(T_mantle_bottom-1200, bml_top_km-80,
            f'T_mantle_bottom\n={T_mantle_bottom:.0f} K',
            fontsize=12, color='#7D6608', va='bottom')

    # colorbar
    sm = ScalarMappable(cmap='rainbow', norm=tbml_norm)
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=ax, orientation='vertical',
                         fraction=0.03, pad=0.12, shrink=0.35)
    cbar.set_label('T_bml (K)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(T_bml_values)

    ax.legend(handles=[
        Line2D([0],[0], color='#2C3E50', lw=2.8,
               label='Mantle T (fixed)'),
        
        Line2D([0],[0], color=color_solidus, lw=1.8, ls='-',
               label=f'Solidus mantle '),
        Line2D([0],[0], color=color_liquid, lw=1.8, ls='--',
               label=f'Liquidus mantle (Mg#={mg_mantle:.2f})')
       
    ], fontsize=9, loc='center left', framealpha=0.9)

    ax.set_title(
        f'Fig 3 — T_bml Variation\n'
        f'Mantle Mg#={mg_mantle:.2f}  |  BML Mg#={mg_bml:.2f}  |  '
        f'BML={BML_THICKNESS:.0f} km\n'
        f'T_bml = {min(T_bml_values)}–{max(T_bml_values)} K',
        fontsize=10)

    plt.tight_layout()

#===================================
#===================================
if PLOT_FIG4:
   
    fig4, (ax_T, ax_phi, ax_vel) = plt.subplots(1, 3, figsize=(20, 9), gridspec_kw={"wspace": 0.08})
    Y_TOP = 1400.0
    Y_BOT = TRUE_CMB_DEPTH + 30
 
    tbml_norm = Normalize(vmin=min(T_bml_values), vmax=max(T_bml_values))
    tbml_cmap = plt.get_cmap("Greens")
 
    color_solidus = "#A52A2A"
    color_liquid  = "#E74C3C"
 
    from matplotlib.ticker import MultipleLocator
 
    def setup_zoom(ax, xlabel):
        ax.set_ylim(Y_BOT, Y_TOP)
        ax.set_ylabel("Depth (km)", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.grid(True, ls="--", alpha=0.25, lw=0.8)
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.tick_params(which="minor", length=3, color="gray")
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.axhline(bml_top_km,     color="#7D6608", ls=":", lw=1.8, alpha=0.9)
        ax.axhline(TRUE_CMB_DEPTH, color="#922B21", ls=":", lw=1.8, alpha=0.9)
 
    # ── LEFT: temperature ─────────────────────────────────────
    setup_zoom(ax_T, "Temperature (K)")
    ax_T.set_xlim(1500, 3800)
 
    man_zoom = mask_mantle & (depth_full >= Y_TOP)
    ax_T.plot(solidus(P_full[man_zoom],  mg_mantle), depth_full[man_zoom],
              color=color_solidus, lw=2.0, ls="-",  zorder=3,
              label=f"Solidus  (mantle Mg#={mg_mantle:.2f})")
    ax_T.plot(liquidus(P_full[man_zoom], mg_mantle), depth_full[man_zoom],
              color=color_liquid,  lw=2.0, ls="--", zorder=3,
              label=f"Liquidus (mantle Mg#={mg_mantle:.2f})")
    ax_T.plot(solidus(P_full[mask_bml],  mg_bml), depth_full[mask_bml],
              color=color_solidus, lw=2.0, ls="-",  zorder=3,
              label=f"Solidus  (BML Mg#={mg_bml:.2f})")
    ax_T.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml],
              color=color_liquid,  lw=2.0, ls="--", zorder=3,
              label=f"Liquidus (BML Mg#={mg_bml:.2f})")
 
    for T_bml in T_bml_values:
        col = tbml_cmap(tbml_norm(T_bml))
        (_, _, _, _, _, _, _, d_i, T_i, _) = build_T_profile(T_bml)
        ax_T.plot(T_i, d_i, color=col, lw=2.2, zorder=5)
 
    ax_T.plot(np.concatenate([T_lith, T_man]),
              np.concatenate([d_lith, d_man]),
              color="#2C3E50", lw=2.5, zorder=6,
              label="Mantle T (fixed adiabat)")
 
    ax_T.axvline(T_mantle_bottom, color="#7D6608", ls=":", lw=1.3, alpha=0.7)
    ax_T.text(T_mantle_bottom + 30, bml_top_km - 25,
              f"T_mb={T_mantle_bottom:.0f} K",
              fontsize=11, color="#7D6608", va="bottom")

 
    # ── RIGHT: melt fraction ──────────────────────────────────
    setup_zoom(ax_phi, "Melt fraction  φ")
    ax_phi.set_xlim(-0.02, 1.05)
    ax_phi.yaxis.set_ticklabels([])
    ax_phi.set_ylabel("")
 
    ax_phi.axvline(0.30, color="#922B21", ls=":", lw=1.5,
                   label="φ = 0.30 (outer core threshold)")
 
    for T_bml in T_bml_values:
        col = tbml_cmap(tbml_norm(T_bml))
        (_, _, _, _, _, _, _, d_i, T_i, P_i) = build_T_profile(T_bml)
 
        bml_mask_i = d_i > bml_top_km
        P_bml_i = P_i[bml_mask_i]
        T_bml_i = T_i[bml_mask_i]
        d_bml_i = d_i[bml_mask_i]
 
        T_sol_i = solidus_duncan2018(P_bml_i).flatten() + fe_correction(mg_bml)
        T_liq_i = liquidus_ruedas2017(P_bml_i).flatten() + fe_correction(mg_bml)
        T_liq_i = np.where(T_liq_i <= T_sol_i, T_sol_i + 200, T_liq_i)
        phi_i   = np.clip((T_bml_i - T_sol_i) / (T_liq_i - T_sol_i), 0, 1)
 
        ax_phi.plot(phi_i, d_bml_i, color=col, lw=2.2, zorder=5)
 
    man_zoom2 = (depth_full >= Y_TOP) & mask_mantle
    ax_phi.plot(np.zeros(man_zoom2.sum()), depth_full[man_zoom2],
                color="#2C3E50", lw=2.5, zorder=6, label="φ = 0 (solid mantle)")
 
    ax_phi.legend(fontsize=12, loc="upper right", framealpha=0.9)
 
    # ── shared colorbar ───────────────────────────────────────
    sm = ScalarMappable(cmap="Greens", norm=tbml_norm)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=[ax_T, ax_phi,ax_vel],
                         orientation="vertical",
                         fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("T_bml (K)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(T_bml_values)
 
    fig4.suptitle(
        f"BML Zoom: Temperature & Melt Fraction\n"
        f"Mantle Mg#={mg_mantle:.2f}  |  BML Mg#={mg_bml:.2f}  |  "
        f"BML={BML_THICKNESS:.0f} km  |  "
        f"T_bml = {min(T_bml_values)}-{max(T_bml_values)} K",
        fontsize=11)
    
    setup_zoom(ax_vel, r"velocity reduction (%)")

    ax_vel.set_xlim(0, 100)
    
    ax_vel.yaxis.set_ticklabels([])
    ax_vel.set_ylabel("")
    
    for T_bml in T_bml_values:
    
        col = tbml_cmap(tbml_norm(T_bml))
    
        (_, _, _, _, _, _, _, d_i, T_i, P_i) = build_T_profile(T_bml)
    
        bml_mask_i = d_i > bml_top_km
    
        P_bml_i = P_i[bml_mask_i]
        T_bml_i = T_i[bml_mask_i]
        d_bml_i = d_i[bml_mask_i]
    
        # melt fraction
        T_sol_i = solidus_duncan2018(P_bml_i).flatten() + fe_correction(mg_bml)
    
        T_liq_i = liquidus_ruedas2017(P_bml_i).flatten() + fe_correction(mg_bml)
    
        T_liq_i = np.where(
            T_liq_i <= T_sol_i,
            T_sol_i + 200,
            T_liq_i
        )
    
        phi_i = np.clip(
            (T_bml_i - T_sol_i) /
            (T_liq_i - T_sol_i),
            0,
            1
        )
    
        # ------------------------------------
        # reference seismic velocity
        # ------------------------------------
    
        Vp0 = np.full_like(phi_i, 13.0)
        Vs0 = np.full_like(phi_i, 7.0)
    
        Vp_melt, Vs_melt = apply_melt_correction_pierru(
            Vp0,
            Vs0,
            phi_i
        )
    
        # percent reduction
        dVp_percent = 100 * (Vp0 - Vp_melt) / Vp0
        dVs_percent = 100 * (Vs0 - Vs_melt) / Vs0
    
        ax_vel.plot(dVp_percent,d_bml_i,color=col,lw=2.2)
        ax_vel.plot(dVs_percent,d_bml_i,color=col,lw=2.2,linestyle='dashed')
    
        ax_vel.axvline(10,color="gray",ls="--",lw=1.5,alpha=0.7 )
        

 
