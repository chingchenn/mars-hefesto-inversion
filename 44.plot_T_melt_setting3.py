# 44.plot_T_melt_setting3.py
# BML temperature profile v2:
#   solid BML : parabola (conduction with HPE heating)
#   liquid BML: adiabat
#   interface : first depth where a trial linear T reaches solidus(mg_bml_bulk)
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── fixed constants ──────────────────────────────────────────────────────────
TRUE_CMB_DEPTH = 1743.3   # km
T_SURF         = 220.0    # K
RHO_MANTLE     = 4200.0   # kg/m3
G_MARS         = 3.72     # m/s2
DTDP_ADIABAT   = 23.0     # K/GPa, mantle adiabat
H_rad          = 2.5e-8   # W/m3, BML heat production
k_bml          = 4.0      # W/m/K, BML conductivity
DTDP_liquid    = 10.0     # K/GPa, liquid-layer adiabat
MG_DUNCAN_REF  = 0.75

# ── MCMC free parameters ─────────────────────────────────────────────────────
T_lit         = 1539.0    # K
P_lit         = 3.69      # GPa
BML_THICKNESS = 150.0     # km
mg_bml_bulk   = 0.62
mg_mantle     = 0.78
n_bml         = 120


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


# P (GPa) = depth (m) * rho * g / 1e9
bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000

# 1. lithosphere: linear from T_SURF to T_lit
d_lith = np.linspace(0, depth_lit, 60)
T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)

# 2. mantle adiabat: T = T_lit + dT/dP (P - P_lit)
d_man = np.linspace(depth_lit, bml_top_km, 150)
P_man = d_man * 1e3 * RHO_MANTLE * G_MARS / 1e9
T_man = T_lit + DTDP_ADIABAT * (P_man - P_lit)
T_mantle_bottom = float(T_man[-1])

# 3. BML grid
d_bml = np.linspace(bml_top_km, TRUE_CMB_DEPTH, n_bml)
P_bml = d_bml * 1e3 * RHO_MANTLE * G_MARS / 1e9

# 4. interface: first point where trial linear T (T_mb -> T_mb + 1000) >= solidus; default all solid
T_linear = np.linspace(T_mantle_bottom, T_mantle_bottom + 1000, n_bml)
hit = T_linear >= solidus(P_bml, mg_bml_bulk)
idx = int(np.argmax(hit)) if hit.any() else n_bml - 1

# 5. solid layer: T(z) = T_mb + A z - B z^2, B = H_rad / (2k),
#    A from T(D_solid) = T_solidus(P_interface)
D_solid     = (d_bml[idx] - bml_top_km) * 1e3           # m
z_solid     = (d_bml[:idx + 1] - bml_top_km) * 1e3      # m
T_interface = float(solidus(P_bml[idx], mg_bml_bulk))
B_coeff = H_rad / (2 * k_bml)
A_coeff = (T_interface - T_mantle_bottom + B_coeff * D_solid**2) / D_solid if D_solid > 0 else 0.0
T_solid = T_mantle_bottom + A_coeff * z_solid - B_coeff * z_solid**2

# 6. liquid layer: T = T_interface + dT/dP_liquid (P - P_interface); T_core is derived
T_liquid = T_interface + DTDP_liquid * (P_bml[idx:] - P_bml[idx])
T_core   = float(T_liquid[-1])

# 7. combine
T_bml = np.empty(n_bml)
T_bml[:idx + 1] = T_solid
T_bml[idx:]     = T_liquid
d_int      = float(d_bml[idx])
depth_full = np.concatenate([d_lith, d_man, d_bml])
T_full     = np.concatenate([T_lith, T_man, T_bml])

print(f"T_mantle_bottom = {T_mantle_bottom:.0f} K")
print(f"T_interface     = {T_interface:.0f} K  (solidus crossing)")
print(f"T_core          = {T_core:.0f} K  (derived, adiabat)")
print(f"d_interface     = {d_int:.1f} km")
print(f"h_solid         = {d_int - bml_top_km:.1f} km")
print(f"h_liquid        = {TRUE_CMB_DEPTH - d_int:.1f} km")
print(f"mg_bml_bulk = {mg_bml_bulk:.2f}, BML_thickness = {BML_THICKNESS:.0f} km, H_rad = {H_rad:.1e} W/m3")

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 11))

# mantle solidus / liquidus
ax.plot(solidus(P_man, mg_mantle), d_man, color='tomato', lw=1.5, ls='-', alpha=0.4,
        label=f'Solidus mantle Mg#={mg_mantle:.2f}')
ax.plot(liquidus(P_man, mg_mantle), d_man, color='tomato', lw=1.5, ls='--', alpha=0.4,
        label=f'Liquidus mantle Mg#={mg_mantle:.2f}')
# BML bulk solidus / liquidus (interface detection)
ax.plot(solidus(P_bml, mg_bml_bulk), d_bml, color='darkorange', lw=2.5, ls='-',
        label=f'Solidus BML bulk Mg#={mg_bml_bulk:.2f}')
ax.plot(liquidus(P_bml, mg_bml_bulk), d_bml, color='darkorange', lw=2.5, ls='--',
        label=f'Liquidus BML bulk Mg#={mg_bml_bulk:.2f}')

# T profile
ax.plot(T_full, depth_full, color='#2C3E50', lw=2.0, alpha=0.4, zorder=4)
ax.plot(T_bml[:idx + 1], d_bml[:idx + 1], 'b-', lw=3.5, zorder=5, label='Solid BML (parabola+HPE)')
ax.plot(T_bml[idx:], d_bml[idx:], 'r-', lw=3.5, zorder=5,
        label=f'Liquid BML (adiabat dT/dP={DTDP_liquid} K/GPa)')

# interfaces
ax.axhline(bml_top_km, color='#7D6608', ls=':', lw=1.8, alpha=0.8, label=f'BML top {bml_top_km:.0f} km')
ax.axhline(d_int, color='steelblue', ls='--', lw=2.0, label=f'Solid/Liquid interface {d_int:.0f} km')
ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8, label=f'CMB {TRUE_CMB_DEPTH:.0f} km')

ax.scatter([T_interface], [d_int], s=50, color='steelblue', zorder=10)
ax.scatter([T_core], [TRUE_CMB_DEPTH], s=50, color='#922B21', zorder=10)
ax.scatter([T_mantle_bottom], [bml_top_km], s=50, color='#7D6608', zorder=10)

ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
ax.set_xlim(500, 4500)
ax.set_ylabel('Depth (km)', fontsize=14)
ax.set_xlabel('Temperature (K)', fontsize=14)
ax.grid(True, ls='--', alpha=0.25)
ax.tick_params(labelsize=14)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.show()
