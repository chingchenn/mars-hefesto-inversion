#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:51:08 2026

@author: chingchen
"""

#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import os



ND_FILE    = 'mcmc_c00_s00042.nd'      
NPZ_FILE   = 'mcmc_c00_s00042.npz'     
OUTPUT_PNG = 'mars_raypaths.png'

MARS_RADIUS = 3389.5   # km
CMB_DEPTH   = 1532.65  # km

# Samuel 2023 


SAMUEL_DATA = {
    'S0154a': {'delta': 29.9,  'depth': 22.0,  'phases': ['SS', 'SSS']},
    'S0173a': {'delta': 30.9,  'depth': 20.9,  'phases': ['P','S','sP','PP','PPP','sS','SS','SSS','ScS']},
    'S0235b': {'delta': 30.1,  'depth': 21.2,  'phases': ['P','S','PP','PPP','sS','SS','SSS','ScS']},
    'S0325a': {'delta': 41.9,  'depth': 30.9,  'phases': ['P','S','pP','PP','PPP','sS','SS','SSS','ScS']},
    'S0407a': {'delta': 28.7,  'depth': 21.1,  'phases': ['P','S','pP','PP','sS','SS','SSS','ScS']},
    'S0409d': {'delta': 29.7,  'depth': 26.2,  'phases': ['P','S','pP','PP','PPP','sS','SS','SSS','ScS']},
    'S0474a': {'delta': 20.5,  'depth': 24.1,  'phases': ['P','S','PP','PPP','SS','SSS']},
    'S0484b': {'delta': 28.7,  'depth': 21.1,  'phases': ['P','S','pP','PP','sS','SS','ScS']},
    'S0784a': {'delta': 30.0,  'depth': 12.9,  'phases': ['P','S','pP','PP','PPP','sS','SS','SSS']},
    'S0802a': {'delta': 28.9,  'depth': 15.8,  'phases': ['P','S','pP','PP','PPP','sS','SS','SSS','ScS']},
    'S0809a': {'delta': 30.6,  'depth': 12.1,  'phases': ['P','S','pP','PP','PPP','sS','SS','SSS','ScS']},
    'S0820a': {'delta': 30.1,  'depth': 15.4,  'phases': ['P','S','PP','PPP','sS']},
    'S0861a': {'delta': 55.3,  'depth': 34.3,  'phases': ['P','S','PP','PPP','SS']},
    'S0864a': {'delta': 29.9,  'depth': 23.9,  'phases': ['P','S','PP','PPP','sS','SS']},
    'S0916d': {'delta': 30.2,  'depth': 11.1,  'phases': ['P','S','pP','PP','PPP','SS','SSS','ScS']},
    'S0918a': {'delta': 16.9,  'depth': 16.9,  'phases': ['P','S','PP','PPP','SS','SSS']},
    'S0976a': {'delta': 143.6, 'depth': 13.5,  'phases': ['SS','SKS']},
    'S1000a': {'delta': 125.9, 'depth': 0.0,   'phases': ['SS','SKS']},
    'S1094b': {'delta': 58.5,  'depth': 0.0,   'phases': ['P','S']},
    'S1222a': {'delta': 37.3,  'depth': 14.9,  'phases': ['P','S','ScS']},
}

# color
PHASE_COLORS = {
    'P':   '#1565C0',
    'pP':  '#1976D2',
    'sP':  '#42A5F5',
    'PP':  '#0D47A1',
    'PPP': '#082880',
    'S':   '#C62828',
    'sS':  '#E53935',
    'SS':  '#B71C1C',
    'SSS': '#7B0000',
    'ScS': '#E65100',
    'SKS': '#6A1B9A',
}



model = TauPyModel(model=NPZ_FILE)

# plot

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

theta = np.linspace(0, 2*np.pi, 500)
r_cmb = (MARS_RADIUS - CMB_DEPTH) / MARS_RADIUS

ax.plot(theta, np.ones(500), color='black', lw=1.8, zorder=5)
ax.fill(theta, np.ones(500) * r_cmb, color='#fff3e0', alpha=0.7, zorder=1)
ax.plot(theta, np.ones(500) * r_cmb,
        color='#e65100', lw=1.2, ls='--', zorder=4, alpha=0.8)
ax.text(np.radians(270), r_cmb - 0.04,
        f'CMB  ({CMB_DEPTH:.0f} km)',
        fontsize=8, color='#e65100', ha='center', va='top', zorder=10)

# ── Ray paths ──
plotted_phases = set()

for name, data in SAMUEL_DATA.items():
    delta  = data['delta']
    depth  = data['depth']
    phases = data['phases']

    try:
        arrivals = model.get_ray_paths(
            source_depth_in_km=depth,
            distance_in_degree=delta,
            phase_list=phases
        )
    except Exception as e:
        print(f"  {name} 計算失敗: {e}")
        continue

    for arrival in arrivals:
        pname = arrival.name
        if pname not in PHASE_COLORS or arrival.path is None:
            continue

        path     = arrival.path
        dist_rad = path['dist']
        r        = (MARS_RADIUS - path['depth']) / MARS_RADIUS
        color    = PHASE_COLORS[pname]
        label    = pname if pname not in plotted_phases else None

        ax.plot(dist_rad, r,
                color=color, lw=0.7, alpha=0.55,
                label=label, solid_capstyle='round', zorder=6)
        plotted_phases.add(pname)

    ax.plot(np.radians(delta), (MARS_RADIUS - depth) / MARS_RADIUS,
            marker='*', color='#333333', markersize=5,
            zorder=10, lw=0, alpha=0.8)

# InSight 
ax.plot(0, 1.0, marker='^', color='#007744', markersize=14,
        zorder=15, lw=0, markeredgecolor='black', markeredgewidth=0.8)
ax.text(np.radians(4), 1.035, 'InSight',
        fontsize=9, color='#007744', ha='left', fontweight='bold')


ax.set_rmax(1.06)
ax.set_rmin(0)
ax.set_theta_zero_location('N')   
ax.set_theta_direction(-1)        
ax.set_rticks([])
ax.set_xticks([])
ax.grid(False)
ax.spines['polar'].set_visible(False)

ax.set_title(
    f"Samuel 2023 — Ray Paths (Mars model)\n"
    f"CMB = {CMB_DEPTH:.0f} km  |  Core radius = {MARS_RADIUS - CMB_DEPTH:.0f} km"
    f"  |  N = {len(SAMUEL_DATA)} events",
    fontsize=12, fontweight='bold', color='black', pad=18)
legend_elements = []
for pname in ['P', 'pP', 'sP', 'PP', 'PPP', 'S', 'sS', 'SS', 'SSS', 'ScS', 'SKS']:
    if pname in plotted_phases:
        legend_elements.append(
            Line2D([0], [0], color=PHASE_COLORS[pname], lw=2.5, label=pname))
legend_elements += [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#007744',
           markeredgecolor='black', markersize=10, label='InSight'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#333333',
           markersize=9, label='Source'),
]
ax.legend(handles=legend_elements,
          loc='lower left', bbox_to_anchor=(-0.12, -0.08),
          fontsize=9, ncol=2, framealpha=0.9,
          edgecolor='#cccccc', title='Phases', title_fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches='tight', facecolor='white')

plt.show()
