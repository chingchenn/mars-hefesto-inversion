# 16.plot_ray_path.py
# Ray paths of the Samuel 2023 event/phase set through one TauP model (polar view, InSight at top).
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy.taup import TauPyModel

plt.rcParams['font.family'] = 'DejaVu Sans'

NPZ_FILE    = 'mcmc_c00_s00042.npz'
MARS_RADIUS = 3389.5   # km
CMB_DEPTH   = 1532.65  # km

# Samuel 2023: epicentral distance (deg), source depth (km), phases used
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
print(f"CMB = {CMB_DEPTH:.0f} km, core radius = {MARS_RADIUS - CMB_DEPTH:.0f} km, N = {len(SAMUEL_DATA)} events")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# surface and CMB (radius normalised by MARS_RADIUS)
theta = np.linspace(0, 2*np.pi, 500)
r_cmb = (MARS_RADIUS - CMB_DEPTH) / MARS_RADIUS
ax.plot(theta, np.ones(500), color='black', lw=1.8, zorder=5)
ax.fill(theta, np.ones(500) * r_cmb, color='#fff3e0', alpha=0.7, zorder=1)
ax.plot(theta, np.ones(500) * r_cmb, color='#e65100', lw=1.2, ls='--', zorder=4, alpha=0.8)

# ray paths and sources
plotted_phases = set()
for name, data in SAMUEL_DATA.items():
    arrivals = model.get_ray_paths(source_depth_in_km=data['depth'],
                                   distance_in_degree=data['delta'],
                                   phase_list=data['phases'])
    for arrival in arrivals:
        pname = arrival.name
        if pname not in PHASE_COLORS or arrival.path is None:
            continue
        r = (MARS_RADIUS - arrival.path['depth']) / MARS_RADIUS
        ax.plot(arrival.path['dist'], r, color=PHASE_COLORS[pname], lw=0.7, alpha=0.55,
                label=pname if pname not in plotted_phases else None,
                solid_capstyle='round', zorder=6)
        plotted_phases.add(pname)
    ax.plot(np.radians(data['delta']), (MARS_RADIUS - data['depth']) / MARS_RADIUS,
            marker='*', color='#333333', markersize=5, zorder=10, lw=0, alpha=0.8)

# InSight
ax.plot(0, 1.0, marker='^', color='#007744', markersize=14, zorder=15, lw=0,
        markeredgecolor='black', markeredgewidth=0.8)

ax.set_rmax(1.06)
ax.set_rmin(0)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rticks([])
ax.set_xticks([])
ax.grid(False)
ax.spines['polar'].set_visible(False)

legend_elements = [Line2D([0], [0], color=PHASE_COLORS[p], lw=2.5, label=p)
                   for p in PHASE_COLORS if p in plotted_phases]
legend_elements += [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#007744',
           markeredgecolor='black', markersize=10, label='InSight'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#333333',
           markersize=9, label='Source'),
]
ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.12, -0.08),
          fontsize=9, ncol=2, framealpha=0.9, edgecolor='#cccccc',
          title='Phases', title_fontsize=9)

plt.tight_layout()
plt.show()
