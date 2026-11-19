# 44.plot_T_melt_setting2.py
# BML concept figures v3, two-layer BML:
#   solid layer : linear T (conduction) from T_mantle_bottom toward T_core;
#                 interface = first depth where phi(bulk Mg#) >= 0.30
#   liquid layer: isothermal at T_core (convection), Mg# from mass conservation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'DejaVu Sans'

PLOT_FIG1 = True
PLOT_FIG4 = False

TRUE_CMB_DEPTH = 1743.3   # km
T_SURF         = 220.0    # K
T_lit          = 1539.0   # K
P_lit          = 3.69     # GPa
BML_THICKNESS  = 150.0    # km
T_core         = 2900.0   # K, = T_interface, free parameter (liquid-layer temperature)
mg_mantle      = 0.78
mg_bml_solid   = 0.72     # free parameter (MCMC)
mg_bml_bulk    = 0.62     # fixed from literature
MG_DUNCAN_REF  = 0.75
RHO_MANTLE     = 4200.0   # kg/m3
RHO_SOLID_BML  = 3600.0
RHO_LIQUID_BML = 3200.0
G_MARS         = 3.72     # m/s2
DTDP_ADIABAT   = 23.0     # K/GPa
PHI_THRESHOLD  = 0.30
n_bml          = 120

T_core_values = [2000, 2200, 2400, 2600, 2800, 3000, 3200]   # Fig 4 sweep

bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000


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


# melt fraction phi = (T - T_sol) / (T_liq - T_sol), clipped to [0, 1]; if T_liq <= T_sol use T_sol + 200
def melt_fraction(T, P, Mg):
    T_sol = solidus(P, Mg)
    T_liq = liquidus(P, Mg)
    T_liq = np.where(T_liq <= T_sol, T_sol + 200.0, T_liq)
    return np.clip((T - T_sol) / (T_liq - T_sol), 0.0, 1.0)


# ── fixed part of the profile: lithosphere (linear) + mantle adiabat ─────────
d_lith = np.linspace(0, depth_lit, 60)
T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)
d_man  = np.linspace(depth_lit, bml_top_km, 150)
T_man  = T_lit + DTDP_ADIABAT * (d_man * 1e3 * RHO_MANTLE * G_MARS / 1e9 - P_lit)   # T_lit + dT/dP (P - P_lit)
T_mantle_bottom = float(T_man[-1])

d_bml = np.linspace(bml_top_km, TRUE_CMB_DEPTH, n_bml)
P_bml = d_bml * 1e3 * RHO_MANTLE * G_MARS / 1e9
depth_full = np.concatenate([d_lith, d_man, d_bml])
P_full     = depth_full * 1e3 * RHO_MANTLE * G_MARS / 1e9
mask_mantle = depth_full <= bml_top_km
mask_bml    = depth_full >  bml_top_km


# ── BML part for one T_core ──────────────────────────────────────────────────
def build_profile(T_core_val):
    # solid layer: linear T from T_mantle_bottom to T_core_val;
    # interface = first point with phi(T_linear, bulk Mg#) >= PHI_THRESHOLD (default n_bml: all solid)
    T_linear = np.linspace(T_mantle_bottom, T_core_val, n_bml)
    hit = melt_fraction(T_linear, P_bml, mg_bml_bulk) >= PHI_THRESHOLD
    idx = int(np.argmax(hit)) if hit.any() else n_bml

    # liquid Mg# by mass conservation: mg_bulk M_tot = mg_solid M_s + mg_liq M_l, M = rho h
    h_solid  = float(d_bml[idx] - bml_top_km) if idx < n_bml else BML_THICKNESS
    h_liquid = BML_THICKNESS - h_solid
    M_solid  = RHO_SOLID_BML * h_solid
    M_liquid = RHO_LIQUID_BML * h_liquid
    if M_liquid <= 0:
        mg_liquid = mg_bml_bulk
    else:
        mg_liquid = float(np.clip((mg_bml_bulk * (M_solid + M_liquid) - mg_bml_solid * M_solid) / M_liquid,
                                  0.01, mg_bml_solid))

    # T: linear in the solid part, isothermal T_core_val in the liquid part
    T_bml = np.where(np.arange(n_bml) < idx, T_linear, T_core_val)
    # phi with solid Mg# above the interface, liquid Mg# below
    phi_bml = melt_fraction(T_bml, P_bml, np.where(np.arange(n_bml) >= idx, mg_liquid, mg_bml_solid))

    d_interface = float(d_bml[idx]) if idx < n_bml else TRUE_CMB_DEPTH
    return dict(T_bml=T_bml, phi_bml=phi_bml, d_interface=d_interface, mg_liquid=mg_liquid,
                h_solid=h_solid, h_liquid=h_liquid, T_full=np.concatenate([T_lith, T_man, T_bml]))


prof = build_profile(T_core)
mg_bml_liquid = prof['mg_liquid']
print(f"  mg_bml_solid  = {mg_bml_solid:.3f}")
print(f"  mg_bml_bulk   = {mg_bml_bulk:.3f}  (fixed)")
print(f"  mg_bml_liquid = {mg_bml_liquid:.3f}  (mass conservation)")
print(f"  h_solid   = {prof['h_solid']:.1f} km")
print(f"  h_liquid  = {prof['h_liquid']:.1f} km")
print(f"  d_interface = {prof['d_interface']:.1f} km")
print(f"  T_core = T_interface = {T_core:.0f} K")
print(f"  lithosphere base {depth_lit:.0f} km, BML top {bml_top_km:.0f} km at T = {T_mantle_bottom:.0f} K")

# ── FIG 1 ────────────────────────────────────────────────────────────────────
if PLOT_FIG1:
    cs = '#A52A2A'
    cl = '#E74C3C'
    fig1, ax = plt.subplots(figsize=(9, 11))

    # mantle solidus / liquidus
    ax.plot(solidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=cs, lw=1.8, ls='-', alpha=0.5)
    ax.plot(liquidus(P_full[mask_mantle], mg_mantle), depth_full[mask_mantle], color=cl, lw=1.8, ls='--', alpha=0.5)
    # solid BML solidus / liquidus
    ax.plot(solidus(P_full[mask_bml], mg_bml_solid), depth_full[mask_bml], color=cs, lw=2.5, ls='-',
            label=f'Solidus solid BML  (Mg#={mg_bml_solid:.2f})')
    ax.plot(liquidus(P_full[mask_bml], mg_bml_solid), depth_full[mask_bml], color=cl, lw=2.5, ls='--',
            label=f'Liquidus solid BML (Mg#={mg_bml_solid:.2f})')
    # bulk BML solidus (interface detection)
    ax.plot(solidus(P_full[mask_bml], mg_bml_bulk), depth_full[mask_bml], color='darkorange', lw=2.0, ls='-.',
            label=f'Solidus bulk BML  (Mg#={mg_bml_bulk:.2f}, interface detection)')

    ax.plot(prof['T_full'], depth_full, color='#2C3E50', lw=3.0, zorder=5, label='T profile')

    ax.axhline(depth_lit, color='gray', ls=':', lw=1.3, alpha=0.7)
    ax.axhline(bml_top_km, color='#7D6608', ls=':', lw=1.8, alpha=0.9)
    ax.axhline(prof['d_interface'], color='steelblue', ls='--', lw=1.8, alpha=0.85)
    ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)
    ax.axhspan(bml_top_km, prof['d_interface'], alpha=0.07, color='#7D6608')
    ax.axhspan(prof['d_interface'], TRUE_CMB_DEPTH, alpha=0.10, color='steelblue')

    ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
    ax.set_xlim(0, 3800)
    ax.set_yticks(np.arange(0, TRUE_CMB_DEPTH + 200, 200))
    ax.set_ylabel('Depth (km)', fontsize=14)
    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.25, lw=0.8)
    ax.tick_params(labelsize=14)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    plt.tight_layout()

# ── FIG 4: zoom on the BML for a T_core sweep ────────────────────────────────
if PLOT_FIG4:
    Y_TOP = 1400.0
    Y_BOT = TRUE_CMB_DEPTH + 30
    cs = '#A52A2A'
    norm4 = Normalize(vmin=min(T_core_values), vmax=max(T_core_values))
    cmap4 = plt.get_cmap('Blues')
    profs = [build_profile(T_c) for T_c in T_core_values]

    fig4, (ax_T, ax_phi, ax_vel) = plt.subplots(1, 3, figsize=(21, 9), gridspec_kw={"wspace": 0.06})
    for ax, xlabel, xlim in [(ax_T, 'Temperature (K)', (1400, 3800)),
                             (ax_phi, 'Melt fraction  φ', (-0.02, 1.05)),
                             (ax_vel, 'Velocity reduction (%)', (0, 105))]:
        ax.set_ylim(Y_BOT, Y_TOP)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.grid(True, ls='--', alpha=0.25, lw=0.8)
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.tick_params(which='minor', length=3, color='gray')
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.axhline(bml_top_km, color='#7D6608', ls=':', lw=1.8, alpha=0.9)
        ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, alpha=0.9)
    ax_T.set_ylabel('Depth (km)', fontsize=14)
    ax_phi.yaxis.set_ticklabels([])
    ax_vel.yaxis.set_ticklabels([])

    # LEFT: temperature
    man_zoom = mask_mantle & (depth_full >= Y_TOP)
    ax_T.plot(solidus(P_full[man_zoom], mg_mantle), depth_full[man_zoom], color=cs, lw=1.5, ls='-', alpha=0.4,
              label=f'Sol mantle Mg#={mg_mantle:.2f}')
    ax_T.plot(solidus(P_full[mask_bml], mg_bml_solid), depth_full[mask_bml], color=cs, lw=2.2, ls='-',
              label=f'Sol BML solid  Mg#={mg_bml_solid:.2f}')
    ax_T.plot(solidus(P_full[mask_bml], mg_bml_bulk), depth_full[mask_bml], color='darkorange', lw=2.0, ls='-.',
              label=f'Sol BML bulk   Mg#={mg_bml_bulk:.2f} (interface detection)')
    zoom = depth_full >= Y_TOP
    for T_c, pr_i in zip(T_core_values, profs):
        col = cmap4(norm4(T_c))
        ax_T.plot(pr_i['T_full'][zoom], depth_full[zoom], color=col, lw=2.0, zorder=5)
        ax_T.axhline(pr_i['d_interface'], color=col, ls='--', lw=0.8, alpha=0.6)
    ax_T.plot(np.concatenate([T_lith, T_man]), np.concatenate([d_lith, d_man]),
              color='#2C3E50', lw=2.5, zorder=6, label='Mantle T (adiabat)')
    ax_T.legend(fontsize=9, loc='lower right', framealpha=0.9)

    # MIDDLE: melt fraction
    ax_phi.axvline(PHI_THRESHOLD, color='steelblue', ls='--', lw=1.8, label=f'φ={PHI_THRESHOLD}  solid/liquid interface')
    for T_c, pr_i in zip(T_core_values, profs):
        col = cmap4(norm4(T_c))
        ax_phi.plot(pr_i['phi_bml'], d_bml, color=col, lw=2.0, zorder=5)
        ax_phi.axhline(pr_i['d_interface'], color=col, ls='--', lw=0.8, alpha=0.6)
    ax_phi.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # RIGHT: velocity reduction, Pierru melt correction
    #   Vp = Vp0 (1 - dVp(phi)), Vs = max(Vs0 (1 - dVs(phi)), 0), piecewise linear in phi
    #   reference Vp0 = 13.7 - 0.5 d/3000, Vs0 = 7.3 - 0.3 d/3000 (km/s, d in km)
    phi_nodes = np.array([0.0, 0.10, 0.20, 0.30, 1.0])
    dVp_nodes = np.array([0.0, 0.10, 0.175, 0.20, 0.20])
    dVs_nodes = np.array([0.0, 0.20, 0.40, 1.00, 1.00])
    Vp0 = 13.7 - 0.5 * (d_bml / 3000.0)
    Vs0 = 7.3 - 0.3 * (d_bml / 3000.0)
    ax_vel.axvline(10, color='gray', ls='--', lw=1.5, alpha=0.6, label='10% ref')
    for T_c, pr_i in zip(T_core_values, profs):
        col = cmap4(norm4(T_c))
        Vp_c = Vp0 * (1 - np.interp(pr_i['phi_bml'], phi_nodes, dVp_nodes))
        Vs_c = np.maximum(Vs0 * (1 - np.interp(pr_i['phi_bml'], phi_nodes, dVs_nodes)), 0.0)
        ax_vel.plot(100.0 * (Vp0 - Vp_c) / Vp0, d_bml, color=col, lw=2.0)
        ax_vel.plot(100.0 * (Vs0 - Vs_c) / Vs0, d_bml, color=col, lw=2.0, ls='--')
    ax_vel.plot([], [], color='gray', lw=2, label='ΔVp (%)')
    ax_vel.plot([], [], color='gray', lw=2, ls='--', label='ΔVs (%)')
    ax_vel.legend(fontsize=10, loc='upper right', framealpha=0.9)

    sm = ScalarMappable(cmap='plasma', norm=norm4)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=[ax_T, ax_phi, ax_vel], orientation='vertical', fraction=0.022, pad=0.02, shrink=0.55)
    cbar.set_label('T_core = T_interface (K)', fontsize=12)
    cbar.set_ticks(T_core_values)
    cbar.ax.tick_params(labelsize=10)

plt.show()
