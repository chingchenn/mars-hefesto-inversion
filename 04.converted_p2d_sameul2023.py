#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:50:00 2026

@author: chingchen
"""

import numpy as np
import matplotlib.pyplot as plt



MARS_RADIUS = 3389.5  
g_mars = 3.72         

SAMUEL_T_FILE_BML = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1/PANEL_I/Tprofile.dat'
SAMUEL_RHO_FILE_BML = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1/PANEL_J/rho_profile.dat'

SAMUEL_T_FILE_noBML = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1/PANEL_C/Tprofile.dat'
SAMUEL_RHO_FILE_noBML = '/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1/PANEL_D/rho_profile.dat'

colors = ['#282130', '#849DAB', '#CD5C5C', '#35838D',
          '#97795D', '#414F67', '#4198B9', '#2F4F4F']

G = 6.67430e-11
labelsize=15

def load_and_process(rho_file, T_file):

    T_raw, r_raw = np.loadtxt(T_file).T
    rho_raw, _   = np.loadtxt(rho_file).T

    idx     = np.argsort(r_raw)
    r_raw   = r_raw[idx]
    rho_raw = rho_raw[idx]
    T_raw   = T_raw[idx]

    mask    = np.diff(r_raw) > 0
    r_sam   = r_raw[:-1][mask]
    rho_sam = rho_raw[:-1][mask]
    T_sam   = T_raw[:-1][mask]

    # km 2 m
    r_m  = r_sam * 1000
    dr   = np.diff(r_m)
    r_mid   = 0.5 * (r_m[:-1] + r_m[1:])
    rho_mid = 0.5 * (rho_sam[:-1] + rho_sam[1:])

    # mass
    dM = 4 * np.pi * r_mid**2 * rho_mid * dr
    M  = np.zeros(len(r_m))
    M[1:] = np.cumsum(dM)

    # g
    g = np.zeros_like(r_m)
    g[1:] = G * M[1:] / r_m[1:]**2

    # pressure
    g_mid   = 0.5 * (g[:-1] + g[1:])
    dP      = rho_mid * g_mid * dr / 1e9   # GPa
    P       = np.zeros_like(r_m)
    P[:-1]  = np.cumsum(dP[::-1])[::-1]

    depth_km = (MARS_RADIUS * 1000 - r_m) / 1000

    return dict(depth_km=depth_km, r_m=r_m, g=g, M=M, P=P, T=T_sam)
bml    = load_and_process(SAMUEL_RHO_FILE_BML, SAMUEL_T_FILE_BML)
no_bml = load_and_process(SAMUEL_RHO_FILE_noBML, SAMUEL_T_FILE_noBML)

mask = (no_bml['P'] >= 0.2) & (no_bml['P'] <= 23.0)
P_out = no_bml['P'][mask]
T_out = no_bml['T'][mask]
idx   = np.argsort(P_out)  
no_bml_out = np.column_stack([P_out[idx], np.zeros(len(idx)), T_out[idx]])
# np.savetxt('/Users/chingchen/Desktop/HeFESTo/no_bml_on_hefestoP.in', no_bml_out, fmt='%.6f %.6f %.6f')


mask = (bml['P'] >= 1.0) & (bml['P'] <= 23.0)
P_out = bml['P'][mask]
T_out = bml['T'][mask]
idx   = np.argsort(P_out)
bml_out = np.column_stack([P_out[idx], np.zeros(len(idx)), T_out[idx]])
# np.savetxt('/Users/chingchen/Desktop/HeFESTo/bml_on_hefestoP.in', bml_out, fmt='%.6f %.6f %.6f')
# PPP,_,TTT = np.loadtxt('/Users/chingchen/Desktop/HeFESTo/ad_samuel_noBML_on_hefestoP.in').T
# plt.plot(PPP,TTT)

# np.savetxt('/Users/chingchen/Desktop/HeFESTo/ad_samuel_BML_on_hefestoP.in', out)

fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15, 8))

ax1.plot(bml['g'], bml['depth_km'],label='BML',color=colors[2],lw=2)
ax1.plot(no_bml['g'], no_bml['depth_km'], label='no BML',color=colors[1],lw=2)
ax2.plot(bml['M'],bml['depth_km'], label='BML',color=colors[2],lw=2)
ax2.plot(no_bml['M'], no_bml['depth_km'], label='no BML',color=colors[1],lw=2)
ax3.plot(bml['P'],bml['depth_km'],label='BML',color=colors[2],lw=2)
ax3.plot(no_bml['P'], no_bml['depth_km'], label='no BML',color=colors[1],lw=2)

ax1.axvline(x=3.73, ymin=0, ymax=MARS_RADIUS,linestyle='dashed')
ax1.legend(fontsize=labelsize)
ax2.set_xlabel("Mass (10^23 kg)",fontsize=labelsize)
ax1.set_xlabel("gravity (m/s$^2$)",fontsize=labelsize)
ax3.set_xlabel("pressure (GPa)",fontsize=labelsize)
ax1.set_ylabel("Depth (km)",fontsize=labelsize)
# ax3.plot(P_sam, depth)

for aa in [ax1,ax2,ax3]:
    aa.grid(True, ls='--', alpha=0.35)
    aa.tick_params(labelsize=labelsize)
    for sp in aa.spines.values(): sp.set_linewidth(2)
    aa.set_ylim(MARS_RADIUS,0)