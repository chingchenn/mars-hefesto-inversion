#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:40:16 2026

@author: chingchen
"""
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import argparse, os


T0          = 205.0          # K, surface temperature (InSight paper)
KAPPA       = 1e-6           # m²/s, thermal diffusivity
T_AGE_MYR   = 4250.0         # Myr, age of Mars
T_SEC       = T_AGE_MYR * 365.2425e6 * 86400   # seconds
SQRT_KAPPAT = np.sqrt(KAPPA * T_SEC)            # m ≈ 366,200 m
G_MARS      = 3.72     # m/s²




"""
make_adin_v2.py
===============
Generate HeFESTo ad.in files using:
  - Areotherm: InSight paper Eq.11
  - Crust density: Table S2-1 + Birch's law (Eq.12)
  - Mantle density: approximate from Fig S2-2(a)

Usage
-----
    python make_adin_v2.py                      # interactive
    python make_adin_v2.py --batch              # 3 scenarios × 2 P-ranges
    python make_adin_v2.py --Tm 1300 --dTdz 0.15 --P1 17 --P2 22 --nP 20

NOTE on depth→pressure accuracy
---------------------------------
The mantle density profile here is read from Fig S2-2(a) by eye and is an
approximation.  For best accuracy, replace MANTLE_ANCHORS with values from
your HeFESTo fort.56 output (column 4 = density) once you have run the
no-BML scenarios.  That gives you a self-consistent P(z) for the Mars model
you are actually using.
"""



# ── Crust: Table S2-1 + Birch's law Eq.12 ────────────────────────────────────
# rho (kg/m3) = (Vp_km_s + 1.87) / 0.00305
# Layer 1 (0–7.98 km):    Vp=2.76, 30% porosity
# Layer 2 (7.98–20.45):   Vp=4.42, 0%  porosity
# Layer 3 (20.45–39.71):  Vp=6.10, 0%  porosity
# Layer 4 (39.71–80.00):  Vp=6.87, 0%  porosity

def _birch(Vp, porosity=0.0):
    return (Vp + 1.87) / 0.00305 * (1.0 - porosity)

CRUST_LAYERS = [
    (0.00,   7.98,  _birch(2.76, 0.30)),   # 1062 kg/m3
    (7.98,  20.45,  _birch(4.42)),           # 2062 kg/m3
    (20.45, 39.71,  _birch(6.10)),           # 2613 kg/m3
    (39.71, 80.00,  _birch(6.87)),           # 2866 kg/m3
]

# ── Mantle: approximate reads from Fig S2-2(a) ────────────────────────────────
# z in km, rho in kg/m3  (roughly averaged across 6 composition models)
# Replace these with fort.56 col-4 after your first HeFESTo run.
MANTLE_ANCHORS_Z   = np.array([  80,  200,  500,  800, 1000, 1100,
                                1200, 1350, 1450, 1560, 1700, 1810])
MANTLE_ANCHORS_RHO = np.array([3300, 3350, 3450, 3550, 3630, 3700,
                                3800, 3930, 4020, 4100, 4150, 4200])

_rho_mantle_interp = interp1d(MANTLE_ANCHORS_Z, MANTLE_ANCHORS_RHO,
                               kind='linear', fill_value='extrapolate')

def rho_of_z(z_km):
    """Density (kg/m3) at depth z_km using crust table + mantle interpolation."""
    for z_top, z_bot, rho in CRUST_LAYERS:
        if z_km <= z_bot:
            return rho
    return float(_rho_mantle_interp(z_km))

# ── Build depth→pressure lookup table ────────────────────────────────────────

def build_P_table(z_max_km=1850.0, dz_km=0.05):
    """
    Numerically integrate dP/dz = rho(z)*g.

    Returns
    -------
    z_arr : 1-D array, km
    P_arr : 1-D array, GPa
    """
    z_arr = np.arange(0.0, z_max_km + dz_km, dz_km)
    P_arr = np.zeros_like(z_arr)
    for i in range(1, len(z_arr)):
        rho = rho_of_z(z_arr[i])
        P_arr[i] = P_arr[i-1] + rho * G_MARS * (dz_km * 1e3) / 1e9
    return z_arr, P_arr

# Build once at import time
_Z_TABLE, _P_TABLE = build_P_table()


def depth_to_pressure(z_km):
    """Interpolate P(GPa) from depth z(km)."""
    return np.interp(z_km, _Z_TABLE, _P_TABLE)

def pressure_to_depth(P_GPa):
    """Interpolate z(km) from pressure P(GPa)."""
    return np.interp(P_GPa, _P_TABLE, _Z_TABLE)


# ── Areotherm ─────────────────────────────────────────────────────────────────

def areotherm_T(z_m, Tsurf, Tm, age_in_myrs, dTdz_K_per_km):
    """
    InSight Eq.11 areotherm.
    z_m : depth in metres (same unit as your original)
    Tm  : mantle T parameter (K), Tp = Tsurf + Tm
    """
    diffusivity = 1e-6
    myrs2sec    = 86400 * 365.2425e6
    t           = age_in_myrs * myrs2sec

    T_erf = Tsurf + Tm * erf(z_m / np.sqrt(diffusivity * t))
    T_adi = (z_m / 1e3) * dTdz_K_per_km   # z 轉 km

    return T_erf + T_adi

def T_of_P(P_GPa, Tm, dTdz, Tsurf=T0, age=T_AGE_MYR):
    """Temperature at pressure P_GPa."""
    z_km = pressure_to_depth(np.asarray(P_GPa))
    return areotherm_T(z_km * 1e3, Tsurf, Tm, age, dTdz)


# ── Write ad.in ───────────────────────────────────────────────────────────────

def write_adin(P_array, Tm, dTdz, outpath, Tsurf=T0, age=T_AGE_MYR):
    """
    Write a HeFESTo ad.in file (3 columns: P  0.0  T).

    Parameters
    ----------
    P_array : 1-D array of pressures in GPa
    Tm      : K
    dTdz    : K/km
    outpath : output file path
    """
    T_array = T_of_P(P_array, Tm, dTdz, Tsurf, age)
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w') as f:
        for P, T in zip(P_array, T_array):
            f.write(f"{P:.4f}  0.0  {T:.2f}\n")
    Tp = Tsurf + Tm
    print(f"  {outpath}")
    print(f"    Tp={Tp:.0f}K  dTdz={dTdz}K/km  "
          f"T({P_array[0]:.0f}GPa)={T_array[0]:.0f}K  "
          f"T({P_array[-1]:.0f}GPa)={T_array[-1]:.0f}K")


# ── Preset scenarios ───────────────────────────────────────────────────────────

PRESETS = {
    'low':    (1100, 0.125),   # Tp = 1305 K
    'medium': (1300, 0.150),   # Tp = 1505 K
    'high':   (1500, 0.175),   # Tp = 1705 K
}

ALL_15 = [(Tm, dTdz)
          for Tm   in [1100, 1200, 1300, 1400, 1500]
          for dTdz in [0.125, 0.150, 0.175]]


def make_batch(outdir='adin_files', all15=False):
    pairs = ALL_15 if all15 else list(PRESETS.values())
    names = ([f"Tm{t}_dT{d}" for t, d in ALL_15] if all15
             else list(PRESETS.keys()))
    os.makedirs(outdir, exist_ok=True)
    print(f"\nWriting {len(pairs)*2} ad.in files → '{outdir}/'")
    print(f"sqrt(kappa*t) = {SQRT_KAPPAT/1e3:.1f} km\n")
    for name, (Tm, dTdz) in zip(names, pairs):
        write_adin(np.linspace(0, 22, 50), Tm, dTdz,
                   f"{outdir}/ad_{name}_noBML.in")
        write_adin(np.linspace(17, 22, 20), Tm, dTdz,
                   f"{outdir}/ad_{name}_BML.in")


def make_single(Tm, dTdz, P1, P2, nP, outpath):
    P_arr = np.linspace(P1, P2, nP)
    print(f"\nGenerating: Tm={Tm}K  dTdz={dTdz}K/km  P={P1}–{P2}GPa  nP={nP}")
    write_adin(P_arr, Tm, dTdz, outpath)


def interactive():
    print("=" * 60)
    print("  HeFESTo ad.in generator v2 — InSight Eq.11 + Table S2-1")
    print("=" * 60)
    print(f"  Crust model: Table S2-1 (Birch's law, 4 layers)")
    print(f"  sqrt(kappa*t) = {SQRT_KAPPAT/1e3:.1f} km\n")

    print("Preset scenarios:")
    for name, (Tm, dTdz) in PRESETS.items():
        Tp = T0 + Tm
        t17 = T_of_P(17, Tm, dTdz)
        t22 = T_of_P(22, Tm, dTdz)
        print(f"  [{name:6s}] Tm={Tm}  dTdz={dTdz}  Tp={Tp:.0f}K"
              f"  T@17GPa={t17:.0f}K  T@22GPa={t22:.0f}K")

    print("\n  1  Preset  2  Custom  3  Batch (3 presets)  4  All 15\n")
    choice = input("Choose [1/2/3/4]: ").strip()

    if choice == '1':
        name = input("Preset (low/medium/high): ").strip().lower()
        Tm, dTdz = PRESETS[name]
    elif choice == '2':
        Tm   = float(input("Tm (K): "))
        dTdz = float(input("dTdz (K/km): "))
        name = f"Tm{int(Tm)}_dT{dTdz}"
    elif choice == '3':
        make_batch('adin_files'); return
    elif choice == '4':
        make_batch('adin_files', all15=True); return
    else:
        print("Invalid"); return

    ptype = input("noBML (0-22 GPa) or BML (17-22 GPa)? [noBML/BML]: ").strip()
    P1, P2, nP = (17, 22, 20) if ptype.lower() == 'bml' else (0, 22, 50)
    out = input(f"Output filename [ad_{name}_{ptype}.in]: ").strip()
    if not out:
        out = f"ad_{name}_{ptype}.in"
    make_single(Tm, dTdz, P1, P2, nP, out)

    print("\nDepth–pressure–temperature check:")
    print(f"  {'z(km)':>8s}  {'P(GPa)':>8s}  {'rho(kg/m3)':>12s}  {'T(K)':>8s}")
    for z in [0, 7.98, 39.71, 80, 200, 500, 1000, 1200, 1350, 1450, 1560]:
        P   = depth_to_pressure(z)
        rho = rho_of_z(z)
        T   = areotherm_T(z*1e3, T0, Tm, T_AGE_MYR, dTdz)
        print(f"  {z:8.2f}  {P:8.3f}  {rho:12.1f}  {T:8.1f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',  action='store_true')
    parser.add_argument('--all15',  action='store_true')
    parser.add_argument('--Tm',     type=float)
    parser.add_argument('--dTdz',   type=float)
    parser.add_argument('--P1',     type=float, default=0)
    parser.add_argument('--P2',     type=float, default=22)
    parser.add_argument('--nP',     type=int,   default=50)
    parser.add_argument('--out',    type=str,   default=None)
    parser.add_argument('--outdir', type=str,   default='adin_files')
    args = parser.parse_args()

    if args.all15:
        make_batch(args.outdir, all15=True)
    elif args.batch:
        make_batch(args.outdir)
    elif args.Tm is not None and args.dTdz is not None:
        out = args.out or f"ad_Tm{int(args.Tm)}_dT{args.dTdz}_P{args.P1}to{args.P2}.in"
        make_single(args.Tm, args.dTdz, args.P1, args.P2, args.nP, out)
    else:
        interactive()