#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:42:30 2026

@author: chingchen
"""

import numpy as np 
import glob
import os
import re
from mars_config import SAMUEL_DIR

SAMUEL_DATA_DIR = (f"{SAMUEL_DIR}/METADATA_BML/DATA_FIG2/PANEL_B")

SAMUEL_RHO_PROFILE_PATH = (f"{SAMUEL_DIR}/METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat")


SAMUEL_FIG1K_DIR = (f"{SAMUEL_DIR}/"
                    "METADATA_BML/DATA_FIG1/PANEL_K")

import numpy as np
data = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
rho, r = data[:, 0], data[:, 1]
print('點數      :', len(r))
print('r 範圍    :', r.min(), '→', r.max(), 'km')
print('rho 範圍  :', rho.min(), '→', rho.max(), '(單位?)')
print('地表密度  :', rho[np.argmax(r)], ' 中心密度:', rho[np.argmin(r)])

TRUE_CMB_DEPTH = None   # km  true CMB
MARS_RADIUS      = 3389.5

def _detect_metal_cmb_from_density(rho, r_km):
    depth = MARS_RADIUS - r_km
    o = np.argsort(depth)
    depth, rho = depth[o], rho[o]
    return float(depth[np.argmax(np.abs(np.diff(rho))) + 1])

def _build_gravity_pressure():
    G = 6.674e-11
    data = np.loadtxt(SAMUEL_RHO_PROFILE_PATH)
    rho, r_km = data[:, 0], data[:, 1]
    cmb_depth = _detect_metal_cmb_from_density(rho, r_km)

    idx = np.argsort(r_km)
    rho, r = rho[idx], r_km[idx] * 1000.0   # m, center→surface

    M = np.zeros(len(r))
    for i in range(1, len(r)):
        M[i] = M[i-1] + 4*np.pi * ((rho[i]+rho[i-1])/2) * ((r[i]+r[i-1])/2)**2 * (r[i]-r[i-1])
    g = np.zeros(len(r));  g[1:] = G * M[1:] / r[1:]**2

    depth = (r[-1] - r) / 1000.0   # km, surface=0
    idx2  = np.argsort(depth)
    depth, g, rho = depth[idx2], g[idx2], rho[idx2]

    P = np.zeros(len(depth))
    for i in range(1, len(depth)):
        P[i] = P[i-1] + ((rho[i]+rho[i-1])/2) * ((g[i]+g[i-1])/2) * (depth[i]-depth[i-1]) * 1e3 / 1e9

    # also return rho profile in g/cm³ for samuel median
    rho_gcm3 = rho / 1000.0
    return depth, g, depth.copy(), P, depth.copy(), rho_gcm3, cmb_depth  

_grav_depth, _grav_g, _pres_depth, _pres_gpa, _rho_depth, _rho_all, TRUE_CMB_DEPTH= _build_gravity_pressure()

def gravity_mars(depth_km):  return np.interp(depth_km, _grav_depth, _grav_g)
def pressure_mars(depth_km): return np.interp(depth_km, _pres_depth, _pres_gpa)

# ── Samuel 2023 median model ──────────────────────────────────────────────────
def _load_samuel_reference():
  
    vp_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vp_profile.dat'))
    vs_data = np.loadtxt(os.path.join(SAMUEL_FIG1K_DIR, 'vs_profile.dat'))
    # 兩檔共用同一組半徑;欄位 = [速度 m/s, 半徑 km]
    depth = MARS_RADIUS - vp_data[:, 1]      # km
    vp    = vp_data[:, 0] / 1000.0           # km/s
    vs    = vs_data[:, 0] / 1000.0           # km/s
    o = np.argsort(depth)
    depth, vp, vs = depth[o], vp[o], vs[o]

    crust_z = np.linspace(0.0, 100.0, 200)
    core_z  = np.linspace(TRUE_CMB_DEPTH, MARS_RADIUS, 200)   # 深度: 金屬CMB -> 地心

    # 地殼: 取固體部分內插到 0-100 km
    solid = vs >= 0.1
    crust_vp = np.interp(crust_z, depth[solid], vp[solid])
    crust_vs = np.interp(crust_z, depth[solid], vs[solid])

    # 地核: 金屬 CMB 以下 (密度決定),Vp 內插;Vs=0 (液核)
    core_sel = depth >= TRUE_CMB_DEPTH
    core_vp  = np.interp(core_z, depth[core_sel], vp[core_sel])

    # 診斷: 表觀核 / BML 頂 = 熔融區頂界 (Vs->0 最淺處)
    liquid = vs < 0.1
    bml_top_depth = float(depth[liquid].min()) if liquid.any() else np.nan

    # 密度來自同一個 Fig 1 密度剖面
    crust_rho = np.interp(crust_z, _rho_depth, _rho_all)
    core_rho  = np.interp(core_z,  _rho_depth, _rho_all)

    print(f"Samuel Fig1 reference: metal CMB={TRUE_CMB_DEPTH:.1f} km  "
          f"apparent CMB / BML top (Vs->0)={bml_top_depth:.1f} km")
    return {
        'crust_z':        crust_z,
        'crust_vp':       crust_vp,
        'crust_vs':       crust_vs,
        'crust_rho':      crust_rho,
        'core_z':         core_z,
        'core_vp':        core_vp,
        'core_vs':        np.zeros(len(core_z)),
        'core_rho':       core_rho,
        'true_cmb_depth': TRUE_CMB_DEPTH,
        'bml_top_depth':  bml_top_depth,   # 新增診斷
    }

_samuel_cache = _load_samuel_reference()

