"""
計算各震相的射線轉折深度
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mcmc",
    "/home/jcchen2/mars-hefesto-inversion/15.mcmc_withBML_MOI.py")
mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

mcmc.load_gravity_profile()
# 用你的 best fit model
TAUP_MODEL_PATH = '/net/beno3/data1/jcchen2/mars-hefesto-runs/taup_models/best_fit_BML_plot.npz'
taup = TauPyModel(model=TAUP_MODEL_PATH)

SAMUEL_DATA = mcmc.SAMUEL_DATA

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
# 算每個事件、每個震相的轉折深度
# ============================================================
results = {}

for phase_name, phase_str in PHASES.items():

    turning_depths = []
    all_depths = []

    for event, obs in SAMUEL_DATA.items():

        try:
            paths = taup.get_ray_paths(
                source_depth_in_km=obs['depth'],
                distance_in_degree=obs['delta'],
                phase_list=[phase_str])

        except Exception:
            continue

        for path in paths:

            if path.name != phase_str:
                continue

            dd = path.path['depth']

            turning_depth = float(np.max(dd))

            # 找 unusually deep rays
            if turning_depth > 800:

                print('======================')
                print(f'Phase : {phase_name}')
                print(f'Event : {event}')
                print(f'Delta : {obs["delta"]:.2f}')
                print(f'Source depth : {obs["depth"]:.2f}')
                print(f'Turning depth : {turning_depth:.2f}')

            turning_depths.append(turning_depth)

            # 收集整條 ray 的 depth sampling
            all_depths.extend(dd)

            break

    results[phase_name] = {
        'turning_depths': np.array(turning_depths),
        'all_depths': np.array(all_depths),
    }
    

# ============================================================
# 畫圖：每個震相的轉折深度分布
# ============================================================
fig, ax = plt.subplots(figsize=(6, 8))

depth_bins = np.linspace(0, 1600, 200)
depth_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])

for phase in PHASES:

    if phase not in results:
        continue

    depths = results[phase]['all_depths']

    if len(depths) == 0:
        continue

    hist, _ = np.histogram(depths, bins=depth_bins)

    # normalize
    hist = hist / np.max(hist)

    ax.plot(hist, depth_centers,
            lw=2,
            label=phase)

# BML top
ax.axhline(1558 - 150,
           color='red',
           ls='--',
           lw=1,
           label='BML top')

# CMB
ax.axhline(1558,
           color='orange',
           ls='--',
           lw=1,
           label='CMB')

ax.set_ylim(1600, 0)

ax.set_xlabel('Relative ray sampling')
ax.set_ylabel('Depth (km)')

ax.set_title('Phase depth sensitivity\n(best-fit BML model)')

ax.legend()

plt.tight_layout()

plt.savefig(
    '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/phase_depth_sensitivity.png',
    dpi=200,
    bbox_inches='tight')

print("Saved: phase_depth_sensitivity.png")

fig, ax = plt.subplots(figsize=(10, 6))

phase_names = [p for p in PHASES if p in results]

for i, phase in enumerate(phase_names):

    depths = results[phase]['turning_depths']

    if len(depths) == 0:
        continue

    depths = np.array(depths)

    # deep rays
    deep_mask = depths > 800

    # normal rays
    ax.scatter(
        np.full(np.sum(~deep_mask), i),
        depths[~deep_mask],
        s=45,
        alpha=0.7,
        label=None
    )

    # highlight deep rays
    ax.scatter(
        np.full(np.sum(deep_mask), i),
        depths[deep_mask],
        s=90,
        color='red',
        edgecolor='black',
        zorder=5
    )

    # mean label
    ax.text(
        i,
        np.mean(depths) - 40,
        f'{np.mean(depths):.0f} km',
        ha='center',
        fontsize=10
    )

# BML top
ax.axhline(
    1408,
    color='red',
    ls='--',
    lw=1.5,
    label='BML top (~1408 km)'
)

# CMB
ax.axhline(
    1558,
    color='orange',
    ls='--',
    lw=1.5,
    label='CMB (~1558 km)'
)

ax.set_xticks(range(len(phase_names)))
ax.set_xticklabels(phase_names, fontsize=12)

ax.set_ylabel('Turning depth (km)', fontsize=13)

ax.set_title(
    'Ray turning depth for each phase\n(best-fit BML model)',
    fontsize=14
)

# depth increasing downward
ax.set_ylim(1600, 0)

ax.grid(alpha=0.2)

ax.legend()

plt.tight_layout()

plt.savefig(
    '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/ray_turning_depth2.png',
    dpi=200,
    bbox_inches='tight'
)

print("Saved: ray_turning_depth2.png")