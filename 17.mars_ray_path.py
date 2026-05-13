import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
import importlib.util

# ============================================================
# load MCMC module
# ============================================================

spec = importlib.util.spec_from_file_location(
    "mcmc",
    "/home/jcchen2/mars-hefesto-inversion/15.mcmc_withBML_MOI.py"
)

mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

mcmc.load_gravity_profile()

SAMUEL_DATA = mcmc.SAMUEL_DATA

# ============================================================
# TauP model
# ============================================================

TAUP_MODEL_PATH = (
    '/net/beno3/data1/jcchen2/'
    'mars-hefesto-runs/taup_models/'
    'best_fit_BML_plot.npz'
)

taup = TauPyModel(model=TAUP_MODEL_PATH)

# ============================================================
# Mars model
# ============================================================

R_MARS = 3389.5
CMB_RADIUS = 1857.0

# ============================================================
# phases
# ============================================================

PHASES = {
    'P':   'P',
    'PP':  'PP',
    'PPP': 'PPP',

    'S':   'S',
    'SS':  'SS',
    'SSS': 'SSS',

    'ScS': 'ScS',
}

# ============================================================
# colors
# ============================================================

PHASE_COLORS = {
    'P': 'royalblue',
    'PP': 'dodgerblue',
    'PPP': 'navy',

    'S': 'red',
    'SS': 'firebrick',
    'SSS': 'darkred',

    'ScS': 'darkorange',
}

# ============================================================
# figure
# ============================================================

fig, ax = plt.subplots(figsize=(10,10))

# ============================================================
# Mars surface
# ============================================================

surface = plt.Circle(
    (0,0),
    R_MARS,
    edgecolor='black',
    facecolor='none',
    lw=2
)

ax.add_patch(surface)

# ============================================================
# CMB
# ============================================================

cmb = plt.Circle(
    (0,0),
    CMB_RADIUS,
    edgecolor='orangered',
    facecolor='wheat',
    alpha=0.25,
    linestyle='--',
    lw=1.5
)

ax.add_patch(cmb)

# ============================================================
# plot rays
# ============================================================

for phase_name, phase_str in PHASES.items():

    color = PHASE_COLORS.get(phase_name, 'gray')

    for event, obs in SAMUEL_DATA.items():

        try:

            paths = taup.get_ray_paths(
                source_depth_in_km=obs['depth'],
                distance_in_degree=obs['delta'],
                phase_list=[phase_str]
            )

        except Exception:
            continue

        for path in paths:

            if path.name != phase_str:
                continue

            depth = path.path['depth']

            # TauP dist is radians
            theta = path.path['dist']

            r = R_MARS - depth

            x = r * np.sin(theta)
            y = r * np.cos(theta)

            ax.plot(
                x,
                y,
                color=color,
                lw=0.8,
                alpha=0.6
            )

            break

# ============================================================
# InSight location
# ============================================================

ax.scatter(
    0,
    R_MARS,
    marker='^',
    s=300,
    color='seagreen',
    edgecolor='black',
    zorder=10
)

ax.text(
    0,
    R_MARS + 80,
    'InSight',
    color='seagreen',
    ha='center',
    fontsize=12,
    weight='bold'
)

# ============================================================
# labels
# ============================================================

ax.text(
    -300,
    0,
    f'CMB ({R_MARS - CMB_RADIUS:.0f} km depth)',
    color='orangered',
    fontsize=11
)

# ============================================================
# formatting
# ============================================================

ax.set_aspect('equal')

ax.set_xlim(-R_MARS*1.05, R_MARS*1.05)
ax.set_ylim(-R_MARS*1.05, R_MARS*1.05)

ax.axis('off')

ax.set_title(
    'Mars ray paths\n(best-fit BML model)',
    fontsize=16,
    weight='bold'
)

plt.tight_layout()

plt.savefig(
    '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/mars_raypaths.png',
    dpi=300,
    bbox_inches='tight'
)

print('Saved: mars_raypaths.png')