#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:02:27 2026

@author: chingchen
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labelsize=15
bwith=3
colors = ['#282130','#849DAB','#CD5C5C','#35838D',
          '#97795D','#414F67','#4198B9','#2F4F4F']

g   = 3.72  # m/s^2
file = '/Users/chingchen/Desktop/HeFESTo/YM_noBML_medium/fort.56'
file = '/Users/chingchen/Desktop/HeFESTo/YM_noBML_high/fort.56'
with open(file) as f:
    f.readline()                   
    cols = f.readline().split()

df56 = pd.read_csv(file, sep=r'\s+', skiprows=2, names=cols)

pressure = df56['P(GPa)'].values
rho   = df56['rho(g/cm^3)'].values
vs    = df56['VS(km/s)'].values
vp    = df56['VP(km/s)'].values
temp = df56['T(K)'].values

dP = np.diff(pressure) * 1e9  # Pa
rho_mid = (rho[:-1] + rho[1:]) / 2  # g/cm³ → kg/m³
dz = dP / (rho_mid * 1000 * g) / 1000  # km
depth = np.zeros(len(pressure))
depth[1:] = np.cumsum(dz)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8, 8))
ax1.plot(vp, pressure,label='Vp (km/s)', linewidth=2,color=colors[2])
ax1.plot(vs, pressure,label='Vs (km/s)', linewidth=2,color=colors[4])
ax1.set_ylabel('pressure (GPa)', fontsize=labelsize)
ax1.set_xlabel('velocity (km/s)', fontsize=labelsize)


# ax1b = ax1.twiny()

ax2.plot( rho,pressure, color=colors[3], linewidth=2, label='ρ (g/cm³)')
ax2.set_xlabel('ρ (g/cm³)', fontsize=labelsize)

l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc='best', fontsize=labelsize)
fig.tight_layout()



from scipy.interpolate import interp1d

p2d = interp1d(pressure, depth)

ax2_twin = ax2.twinx()
ax2_twin.set_ylim(25, 0) 
ax2_twin2 = ax2.twiny()
ax2_twin2.plot(temp,pressure,label='temperature', linewidth=2,color=colors[0])

p_ticks = [0, 5, 10, 15, 20, pressure.max()]
d_labels = [f'{p2d(p):.0f}' if p <= pressure.max() else '0' for p in p_ticks]

ax2_twin.set_yticks(p_ticks)
ax2_twin.set_yticklabels(d_labels)
ax2_twin.set_ylabel('depth (km)', fontsize=labelsize)
ax2_twin.tick_params(labelsize=labelsize)
ax2_twin2.set_xlabel('temperature(K)', fontsize=labelsize)

ax1.tick_params(labelsize=labelsize)
ax2.tick_params(labelsize=labelsize, labelcolor=colors[3])
for aa in [ax1,ax2]:
    aa.set_ylim(25,0)
    aa.grid(True, which="both", ls="--", alpha=0.4)
    for axis in ['top','bottom','left','right']:
        aa.spines[axis].set_linewidth(bwith)