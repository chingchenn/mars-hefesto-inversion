#!/usr/bin/env python3
"""
New BML Temperature Profile v2
================================
固態 BML：拋物線（HPE 加熱）
液態 BML：絕熱梯度
界面：T_profile 超過 solidus(mg_bml_bulk) 的地方

★ 改動這裡 ★
"""
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 固定常數
# ═══════════════════════════════════════════════════════════════
MARS_RADIUS    = 3389.5
TRUE_CMB_DEPTH = 1743.3
T_SURF         = 220.0
RHO_MANTLE     = 4200.0
G_MARS         = 3.72
DTDP_ADIABAT   = 23.0     # K/GPa，地幔絕熱梯度
H_rad          = 2.5e-8   # W/m³，BML HPE 生熱率
k_bml          = 4.0      # W/m/K，BML 熱導率
DTDP_liquid    = 10.0     # K/GPa，液態層絕熱梯度

# ═══════════════════════════════════════════════════════════════
# ★ MCMC 自由參數
# ═══════════════════════════════════════════════════════════════
T_lit         = 1539.0
P_lit         = 3.69
BML_THICKNESS = 150.0
mg_bml_bulk   = 0.62    # ★ 自由參數（原本固定）
mg_mantle     = 0.78
MG_DUNCAN_REF = 0.75

# ═══════════════════════════════════════════════════════════════
# 輔助函數
# ═══════════════════════════════════════════════════════════════
def depth_to_P(d_km):
    return d_km * 1e3 * RHO_MANTLE * G_MARS / 1e9

def fe_correction(Mg):
    return -6.0 * (100*(1-Mg) - 100*(1-MG_DUNCAN_REF))

def solidus(P, Mg):
    P = np.atleast_1d(np.asarray(P, float))
    T_C = np.where(P <= 10,
                   -4.877*P**2 + 120.2*P + 1088.0,
          np.where(P <= 23,
                   -1.323*(P-10)**2 + 38.18*(P-10) + 1802.0,
                   77.75*(P-23) + 2075.0))
    return T_C + 273.15 + fe_correction(Mg)

def liquidus(P, Mg):
    P = np.atleast_1d(np.asarray(P, float))
    T_liq = 2160.6 + 64.7109*P - 3.97463*P**2 + 0.0957894*P**3
    return T_liq + fe_correction(Mg)

# ═══════════════════════════════════════════════════════════════
# build_profile
# ═══════════════════════════════════════════════════════════════
def build_profile(T_lit, P_lit, BML_THICKNESS, mg_bml_bulk, mg_mantle,
                  n_bml=120):

    bml_top_km = TRUE_CMB_DEPTH - BML_THICKNESS
    depth_lit  = P_lit * 1e9 / (RHO_MANTLE * G_MARS) / 1000

    # 1. Lithosphere
    d_lith = np.linspace(0, depth_lit, 60)
    T_lith = T_SURF + (T_lit - T_SURF) * (d_lith / depth_lit)

    # 2. Mantle adiabat
    d_man = np.linspace(depth_lit, bml_top_km, 150)
    P_man = depth_to_P(d_man)
    T_man = T_lit + DTDP_ADIABAT * (P_man - P_lit)
    T_mantle_bottom = float(T_man[-1])

    # 3. BML grid
    d_bml = np.linspace(bml_top_km, TRUE_CMB_DEPTH, n_bml)
    P_bml = depth_to_P(d_bml)

    # 4. 先用線性 T 掃描找界面位置（第一次估計）
    T_linear = np.linspace(T_mantle_bottom, T_mantle_bottom + 1000, n_bml)
    idx_interface = n_bml - 1   # default: 全固態
    for i in range(n_bml):
        T_sol_i = solidus(P_bml[i:i+1], mg_bml_bulk)[0]
        if T_linear[i] >= T_sol_i:
            idx_interface = i
            break

    # 5. 固態層：拋物線（HPE）
    # T(z) = T_mantle_bottom + A*z - B*z²
    # B = H_rad/(2k)
    # A 由底端條件 T(D_solid) = T_solidus(P_interface) 決定
    D_solid    = (d_bml[idx_interface] - bml_top_km) * 1e3   # m
    z_solid    = (d_bml[:idx_interface+1] - bml_top_km) * 1e3  # m
    T_interface = float(solidus(P_bml[idx_interface:idx_interface+1],
                                mg_bml_bulk)[0])
    B_coeff = H_rad / (2 * k_bml)
    if D_solid > 0:
        A_coeff = (T_interface - T_mantle_bottom + B_coeff*D_solid**2) / D_solid
    else:
        A_coeff = 0.0
    T_solid = T_mantle_bottom + A_coeff*z_solid - B_coeff*z_solid**2

    # 6. 液態層：絕熱梯度
    P_interface = float(P_bml[idx_interface])
    T_liquid    = T_interface + DTDP_liquid * (P_bml[idx_interface:] - P_interface)
    T_core      = float(T_liquid[-1])   # 派生量

    # 7. 組合
    T_bml = np.empty(n_bml)
    T_bml[:idx_interface+1] = T_solid
    T_bml[idx_interface:]   = T_liquid

    depth_full = np.concatenate([d_lith, d_man, d_bml])
    T_full     = np.concatenate([T_lith, T_man, T_bml])

    return dict(
        d_lith=d_lith, T_lith=T_lith,
        d_man=d_man,   T_man=T_man,
        d_bml=d_bml,   T_bml=T_bml,
        P_bml=P_bml,
        depth_full=depth_full, T_full=T_full,
        T_mantle_bottom=T_mantle_bottom,
        T_interface=T_interface,
        T_core=T_core,
        d_interface=float(d_bml[idx_interface]),
        idx_interface=idx_interface,
        bml_top_km=bml_top_km,
    )

# ═══════════════════════════════════════════════════════════════
# 跑並印出
# ═══════════════════════════════════════════════════════════════
prof = build_profile(T_lit, P_lit, BML_THICKNESS, mg_bml_bulk, mg_mantle)

print(f"T_mantle_bottom = {prof['T_mantle_bottom']:.0f} K")
print(f"T_interface     = {prof['T_interface']:.0f} K  (solidus crossing)")
print(f"T_core          = {prof['T_core']:.0f} K  (derived, adiabat)")
print(f"d_interface     = {prof['d_interface']:.1f} km")
print(f"h_solid         = {prof['d_interface']-prof['bml_top_km']:.1f} km")
print(f"h_liquid        = {TRUE_CMB_DEPTH-prof['d_interface']:.1f} km")

# ═══════════════════════════════════════════════════════════════
# 畫圖
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_ylim(TRUE_CMB_DEPTH + 30, 0)
ax.set_xlim(500, 4500)
ax.set_ylabel('Depth (km)', fontsize=14)
ax.set_xlabel('Temperature (K)', fontsize=14)
ax.grid(True, ls='--', alpha=0.25)

P_bml = prof['P_bml']
d_bml = prof['d_bml']
bml_top = prof['bml_top_km']
d_int   = prof['d_interface']
idx     = prof['idx_interface']

# ── Solidus and Liquidus curves ──
# 地幔
P_man_plot = depth_to_P(prof['d_man'])
ax.plot(solidus(P_man_plot,  mg_mantle), prof['d_man'],
        color='tomato', lw=1.5, ls='-', alpha=0.4,
        label=f'Solidus mantle Mg#={mg_mantle:.2f}')
ax.plot(liquidus(P_man_plot, mg_mantle), prof['d_man'],
        color='tomato', lw=1.5, ls='--', alpha=0.4,
        label=f'Liquidus mantle Mg#={mg_mantle:.2f}')

# BML bulk（界面判斷用）
ax.plot(solidus(P_bml,  mg_bml_bulk), d_bml,
        color='darkorange', lw=2.5, ls='-',
        label=f'Solidus BML bulk Mg#={mg_bml_bulk:.2f}')
ax.plot(liquidus(P_bml, mg_bml_bulk), d_bml,
        color='darkorange', lw=2.5, ls='--',
        label=f'Liquidus BML bulk Mg#={mg_bml_bulk:.2f}')

# ── T profile ──
ax.plot(prof['T_full'], prof['depth_full'],
        color='#2C3E50', lw=2.0, alpha=0.4, zorder=4)

# 固態（拋物線，藍）
ax.plot(prof['T_bml'][:idx+1], d_bml[:idx+1],
        'b-', lw=3.5, zorder=5,
        label=f'Solid BML (parabola+HPE)')

# 液態（絕熱，紅）
ax.plot(prof['T_bml'][idx:], d_bml[idx:],
        'r-', lw=3.5, zorder=5,
        label=f'Liquid BML (adiabat dT/dP={DTDP_liquid} K/GPa)')

# ── 界面線 ──
ax.axhline(bml_top, color='#7D6608', ls=':', lw=1.8, alpha=0.8,
           label=f'BML top {bml_top:.0f} km')
ax.axhline(d_int,   color='steelblue', ls='--', lw=2.0,
           label=f'Solid/Liquid interface {d_int:.0f} km')
ax.axhline(TRUE_CMB_DEPTH, color='#922B21', ls=':', lw=1.8,
           label=f'CMB {TRUE_CMB_DEPTH:.0f} km')

# ── 溫度標注 ──
ax.scatter([prof['T_interface']], [d_int], s=50,
           color='steelblue', zorder=10)
ax.text(prof['T_interface']+80, d_int-15,
        f'T_interface={prof["T_interface"]:.0f}K\n= solidus(mg_bulk)',
        fontsize=9, color='steelblue')

ax.scatter([prof['T_core']], [TRUE_CMB_DEPTH], s=50,
           color='#922B21', zorder=10)
ax.text(prof['T_core']+80, TRUE_CMB_DEPTH-15,
        f'T_core={prof["T_core"]:.0f}K\n(derived)',
        fontsize=9, color='#922B21')

ax.scatter([prof['T_mantle_bottom']], [bml_top], s=50,
           color='#7D6608', zorder=10)

# ── 區域標籤 ──
ax.text(800, (bml_top+d_int)/2,
        f'Solid BML\nparabola\nH_rad={H_rad:.1e}',
        fontsize=10, color='blue', fontweight='bold')
ax.text(800, (d_int+TRUE_CMB_DEPTH)/2,
        f'Liquid BML\nadiabat',
        fontsize=10, color='red', fontweight='bold')

ax.legend(fontsize=8, loc='upper right')
ax.set_title(
    f'BML Temperature Profile (new)\n'
    f'mg_bml_bulk={mg_bml_bulk:.2f} (free)  |  '
    f'BML_thickness={BML_THICKNESS:.0f} km\n'
    f'Solid: parabola+HPE  |  Liquid: adiabat\n'
    f'T_core={prof["T_core"]:.0f}K (derived from adiabat)',
    fontsize=10, fontweight='bold')

plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/new_T_profile_v2.png',
            # dpi=150, bbox_inches='tight')
# plt.close()
print("\nSaved: new_T_profile_v2.png")