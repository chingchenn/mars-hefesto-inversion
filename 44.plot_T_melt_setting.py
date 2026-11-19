# 44.plot_T_melt_setting.py
# BML concept figures (BML with linear T from T_mantle_bottom to T_bml at the CMB):
#   Fig 1: standard concept, fixed mantle and BML Mg#
#   Fig 2: Mg# variation (rainbow)
#   Fig 3: T_bml variation (rainbow)
#   Fig 4: BML zoom: T, melt fraction and velocity reduction for the T_bml sweep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'DejaVu Sans'

PLOT_FIG1 = 1    # standard concept figure
PLOT_FIG2 = 0    # Mg# variation
PLOT_FIG3 = 0    # T_bml variation
PLOT_FIG4 = 1    # BML zoom-in

TRUE_CMB_DEPTH = 1743.3   # km
T_SURF         = 220.0    # K
T_lit          = 1539.0   # K  lithosphere/adiabat transition
P_lit          = 3.69     # GPa
BML_THICKNESS  = 250      # km
T_bml_default  = 2800.0   # K  BML bottom temperature (Fig 1 & 2)
mg_mantle      = 0.78     # mantle Mg#
mg_bml         = 0.65     # BML Mg#
MG_DUNCAN_REF  = 0.75
RHO_MANTLE     = 4200.0   # kg/m3 (lower mantle approx)
G_MARS         = 3.72     # m/s2
DTDP_ADIABAT   = 23.0     # K/GPa  mantle adiabat gradient

mg_values    = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]          # Fig 2
T_bml_values = [2000, 2200, 2400, 2600, 2800, 3000, 3200]    # Fig 3, 4 (K)

color_solidus = '#A52A2A'
color_liquid  = '#E74C3C'

bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000   # km


# Duncan 2018 solidus (deg C -> K) and Ruedas 2017 liquidus (K), both shifted by the Fe correction
#   dT_Fe = -6 * [100 (1 - Mg) - 100 (1 - 0.75)]
def solidus(P, Mg):
    T_C = np.where(P <= 10, -4.877 * P**2 + 120.2 * P + 1088.0,
          np.where(P <= 23, -1.323 * (P - 10)**2 + 38.18 * (P - 10) + 1802.0,
                            77.75 * (P - 23) + 2075.0))
    return T_C + 273.15 - 6.0 * (100 * (1 - Mg) - 100 * (1 - MG_DUNCAN_REF))


def liquidus(P, Mg):
    T_liq = 2160.6 + 64.7109 * P - 3.97463 * P**2 + 0.0957894 * P**3
    return T_liq - 6.0 * (100 * (1 - Mg) - 100 * (1 - MG_DUNCAN_REF))


# ── temperature profile ──────────────────────────────────────────────────────
# lithosphere: linear T_SURF -> T_lit ; mantle: T = T_lit + dT/dP (P - P_lit) ; P = d rho g
d_lith = np.linspace(0, depth_lit, 80)
T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)
d_man  = np.linspace(depth_lit, bml_top_km, 200)
T_man  = T_lit + DTDP_ADIABAT * (d_man * 1e3 * RHO_MANTLE * G_MARS / 1e9 - P_lit)
T_mantle_bottom = float(T_man[-1])
d_bml  = np.linspace(bml_top_km, TRUE_CMB_DEPTH, 60)

depth_full  = np.concatenate([d_lith, d_man, d_bml])
P_full      = depth_full * 1e3 * RHO_MANTLE * G_MARS / 1e9
mask_mantle = depth_full <= bml_top_km
mask_bml    = depth_full >  bml_top_km
# BML: linear from T_mantle_bottom to T_bml at the CMB (only this part depends on T_bml)
T_full      = np.concatenate([T_lith, T_man, np.linspace(T_mantle_bottom, T_bml_default, len(d_bml))])
T_full_list = [np.concatenate([T_lith, T_man, np.linspace(T_mantle_bottom, Tb, len(d_bml))])
               for Tb in T_bml_values]

print(f'Lithosphere base {depth_lit:.0f} km | {P_lit:.1f} GPa | T = {T_lit:.0f} K')
print(f'BML top {bml_top_km:.0f} km | T_mantle_bottom = {T_mantle_bottom:.0f} K')
print(f'CMB {TRUE_CMB_DEPTH:.0f} km | T_bml = {T_bml_default:.0f} K')
print(f'Mantle Mg# = {mg_mantle:.2f}, BML Mg# = {mg_bml:.2f}, BML = {BML_THICKNESS:.0f} km')


def setup_ax(ax):
    ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
    ax.set_xlim(0, 3800)
    ax.set_ylabel('Depth (km)', fontsize=14)
    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.25, lw=0.8)
    ax.tick_params(labelsize=14)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)
    ax.axhline(depth_lit,      color='gray',    ls=':', lw=1.3, alpha=0.7)
    ax.axhline(bml_top_km,     color='#7D6608', ls=':', lw=1.8, alpha=0.9)
    ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)


# ── FIG 1: standard concept ──────────────────────────────────────────────────
if PLOT_FIG1:
    fig1, ax = plt.subplots(figsize=(8, 10))
    # solidus / liquidus: mantle Mg# in the mantle, BML Mg# in the BML
    ax.plot(solidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=color_solidus, lw=2.0, ls='-',
            label='Solidus (Duncan 2018)')
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=color_liquid, lw=2.0, ls='--',
            label='Liquidus (Ruedas 2017)')
    ax.plot(solidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_liquid, lw=2.0, ls='--')
    # faint mantle solidus / liquidus extended into the BML (reference)
    ax.plot(solidus(P_full[mask_bml], mg_mantle), depth_full[mask_bml], color=color_solidus, lw=1.0, ls='-', alpha=0.25)
    ax.plot(liquidus(P_full[mask_bml], mg_mantle), depth_full[mask_bml], color=color_liquid, lw=1.0, ls='--', alpha=0.25)
    ax.plot(T_full, depth_full, color='#2C3E50', lw=2.8, zorder=5, label='T profile')
    setup_ax(ax)
    ax.set_yticks(np.arange(0, TRUE_CMB_DEPTH + 200, 200))
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    plt.tight_layout()

# ── FIG 2: Mg# variation ─────────────────────────────────────────────────────
if PLOT_FIG2:
    fig2, ax = plt.subplots(figsize=(8, 10))
    mg_norm = Normalize(vmin=min(mg_values), vmax=max(mg_values))
    for mm in mg_values:
        col = plt.get_cmap('rainbow')(mg_norm(mm))
        ax.plot(solidus(P_full, mm), depth_full, color=col, lw=1.8, ls='-', zorder=3)
        ax.plot(liquidus(P_full, mm), depth_full, color=col, lw=1.8, ls='--', zorder=3)
    ax.plot(T_full, depth_full, color='#2C3E50', lw=2.8, zorder=5, label='T profile')
    setup_ax(ax)
    sm = ScalarMappable(cmap='rainbow', norm=mg_norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.12, shrink=0.35, label='Mg#')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(mg_values)
    cbar.set_ticklabels([f'{m:.2f}' for m in mg_values])
    plt.tight_layout()

# ── FIG 3: T_bml variation ───────────────────────────────────────────────────
if PLOT_FIG3:
    fig3, ax = plt.subplots(figsize=(8, 10))
    setup_ax(ax)
    ax.plot(solidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=color_solidus, lw=2.0, ls='-',
            label='Solidus mantle ')
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=color_liquid, lw=2.0, ls='--',
            label=f'Liquidus mantle (Mg#={mg_mantle:.2f})')
    ax.plot(solidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_solidus, lw=2.0, ls='-')
    ax.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_liquid, lw=2.0, ls='--')
    tbml_norm = Normalize(vmin=min(T_bml_values), vmax=max(T_bml_values))
    for Tb, T_i in zip(T_bml_values, T_full_list):
        ax.plot(T_i, depth_full, color=plt.get_cmap('rainbow')(tbml_norm(Tb)), lw=2.2, zorder=5)
    ax.plot(np.concatenate([T_lith, T_man]), np.concatenate([d_lith, d_man]),
            color='#2C3E50', lw=2.8, zorder=6, label='Mantle T (fixed)')
    ax.axvline(T_mantle_bottom, color='#7D6608', ls=':', lw=1.2, alpha=0.6)
    sm = ScalarMappable(cmap='rainbow', norm=tbml_norm)
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.12, shrink=0.35)
    cbar.set_label('T_bml (K)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(T_bml_values)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    plt.tight_layout()

# ── FIG 4: BML zoom ──────────────────────────────────────────────────────────
if PLOT_FIG4:
    Y_TOP = 1400.0
    Y_BOT = TRUE_CMB_DEPTH + 30
    tbml_norm = Normalize(vmin=min(T_bml_values), vmax=max(T_bml_values))
    tbml_cmap = plt.get_cmap('Greens')

    fig4, (ax_T, ax_phi, ax_vel) = plt.subplots(1, 3, figsize=(20, 9), gridspec_kw={"wspace": 0.08})
    for ax, xlabel in [(ax_T, 'Temperature (K)'), (ax_phi, 'Melt fraction  φ'), (ax_vel, 'velocity reduction (%)')]:
        ax.set_ylim(Y_BOT, Y_TOP)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.grid(True, ls='--', alpha=0.25, lw=0.8)
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.tick_params(which='minor', length=3, color='gray')
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.axhline(bml_top_km,     color='#7D6608', ls=':', lw=1.8, alpha=0.9)
        ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)
    ax_T.set_ylabel('Depth (km)', fontsize=14)
    ax_T.set_xlim(1500, 3800)
    ax_phi.set_xlim(-0.02, 1.05)
    ax_vel.set_xlim(0, 100)
    ax_phi.yaxis.set_ticklabels([])
    ax_vel.yaxis.set_ticklabels([])

    # LEFT: temperature
    man_zoom = mask_mantle & (depth_full >= Y_TOP)
    ax_T.plot(solidus(P_full[man_zoom], mg_mantle), depth_full[man_zoom], color=color_solidus, lw=2.0, ls='-', zorder=3,
              label=f'Solidus  (mantle Mg#={mg_mantle:.2f})')
    ax_T.plot(liquidus(P_full[man_zoom], mg_mantle), depth_full[man_zoom], color=color_liquid, lw=2.0, ls='--', zorder=3,
              label=f'Liquidus (mantle Mg#={mg_mantle:.2f})')
    ax_T.plot(solidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_solidus, lw=2.0, ls='-', zorder=3,
              label=f'Solidus  (BML Mg#={mg_bml:.2f})')
    ax_T.plot(liquidus(P_full[mask_bml], mg_bml), depth_full[mask_bml], color=color_liquid, lw=2.0, ls='--', zorder=3,
              label=f'Liquidus (BML Mg#={mg_bml:.2f})')
    for Tb, T_i in zip(T_bml_values, T_full_list):
        ax_T.plot(T_i, depth_full, color=tbml_cmap(tbml_norm(Tb)), lw=2.2, zorder=5)
    ax_T.plot(np.concatenate([T_lith, T_man]), np.concatenate([d_lith, d_man]),
              color='#2C3E50', lw=2.5, zorder=6, label='Mantle T (fixed adiabat)')
    ax_T.axvline(T_mantle_bottom, color='#7D6608', ls=':', lw=1.3, alpha=0.7)

    # MIDDLE: melt fraction phi = (T - T_sol) / (T_liq - T_sol), clipped to [0, 1]
    #         (if T_liq <= T_sol use T_sol + 200), BML Mg#
    # RIGHT : Pierru velocity reduction, piecewise linear in phi:
    #         Vp = Vp0 (1 - dVp), Vs = max(Vs0 (1 - dVs), 0), Vp0 = 13, Vs0 = 7 km/s
    ax_phi.axvline(0.30, color='#922B21', ls=':', lw=1.5, label='φ = 0.30 (outer core threshold)')
    ax_vel.axvline(10, color='gray', ls='--', lw=1.5, alpha=0.7)
    phi_nodes = np.array([0.0, 0.10, 0.20, 0.30, 1.0])
    dVp_nodes = np.array([0.0, 0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0, 0.20, 0.40, 1.00, 1.00])
    P_b = P_full[mask_bml]
    d_b = depth_full[mask_bml]
    T_sol = solidus(P_b, mg_bml)
    T_liq = liquidus(P_b, mg_bml)
    T_liq = np.where(T_liq <= T_sol, T_sol + 200, T_liq)
    Vp0 = np.full_like(P_b, 13.0)
    Vs0 = np.full_like(P_b, 7.0)
    for Tb, T_i in zip(T_bml_values, T_full_list):
        col = tbml_cmap(tbml_norm(Tb))
        phi = np.clip((T_i[mask_bml] - T_sol) / (T_liq - T_sol), 0, 1)
        Vp_melt = Vp0 * (1.0 - np.interp(phi, phi_nodes, dVp_nodes))
        Vs_melt = np.maximum(Vs0 * (1.0 - np.interp(phi, phi_nodes, dVs_nodes)), 0.0)
        ax_phi.plot(phi, d_b, color=col, lw=2.2, zorder=5)
        ax_vel.plot(100 * (Vp0 - Vp_melt) / Vp0, d_b, color=col, lw=2.2)
        ax_vel.plot(100 * (Vs0 - Vs_melt) / Vs0, d_b, color=col, lw=2.2, linestyle='dashed')
    ax_phi.plot(np.zeros(man_zoom.sum()), depth_full[man_zoom], color='#2C3E50', lw=2.5, zorder=6,
                label='φ = 0 (solid mantle)')
    ax_phi.legend(fontsize=12, loc='upper right', framealpha=0.9)

    sm = ScalarMappable(cmap='Greens', norm=tbml_norm)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=[ax_T, ax_phi, ax_vel], orientation='vertical', fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label('T_bml (K)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(T_bml_values)

plt.show()
