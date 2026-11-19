"""
plot_residual_from_cache.py
直接從 cache 讀取 fort56 data，重新算殘差並畫圖
"""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib.util
from mars_config import MCMC_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script
spec = importlib.util.spec_from_file_location(
    "mcmc",
    f"{HERE}/15.mcmc_withBML_MOI.py")
mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

os.chdir(MCMC_DIR)
mcmc.load_gravity_profile()
khan_cache = mcmc.compute_khan_median()

CACHE_DIR      = 'cache'
phases_to_plot = [
    'S-P',
    'PP-P',
    'PPP-P',

    'SS-S',
    'SSS-S',

    'ScS-S',
    'ScS-SS',
]
SIGMA_dict     = mcmc.SIGMA
RESID_CACHE_DIR = 'residual_cache'
os.makedirs(RESID_CACHE_DIR, exist_ok=True)

# ============================================================
# 找所有 cache 裡的模型名稱
# ============================================================
mantle_files = sorted(glob.glob(f'{CACHE_DIR}/*_mantle.npz'))
model_names  = [os.path.basename(f).replace('_mantle.npz', '')
                for f in mantle_files]
print(f"Found {len(model_names)} cached models: {model_names}")

# ============================================================
# 從 cache 讀取 fort56_data，建 TauPy，算殘差
# ============================================================
def load_from_cache(model_name):
    mantle_path = f'{CACHE_DIR}/{model_name}_mantle.npz'
    bml_path    = f'{CACHE_DIR}/{model_name}_bml.npz'

    if not os.path.exists(mantle_path):
        return None

    d  = np.load(mantle_path, allow_pickle=True)
    fd = {k: d[k] for k in d}

    bd = None
    if os.path.exists(bml_path):
        d2 = np.load(bml_path, allow_pickle=True)
        bd = {k: d2[k] for k in d2}

    return fd, bd


def compute_residuals(model_name):

    cache_file = f'{RESID_CACHE_DIR}/{model_name}_resid.npz'

    # ========================================================
    # load cached residuals
    # ========================================================

    if os.path.exists(cache_file):

        d = np.load(cache_file, allow_pickle=True)

        return d['resid_dict'].item()

    # ========================================================
    # otherwise compute
    # ========================================================

    result = load_from_cache(model_name)

    if result is None:
        return None

    fd, bd = result

    try:

        taup = mcmc.build_taup(
            fd,
            model_name,
            khan_cache,
            bml_data=bd
        )

    except Exception as e:

        print(f"TauP failed for {model_name}: {e}")

        return None

    resid_dict = {}

    for event, obs in mcmc.SAMUEL_DATA.items():

        try:

            arr = taup.get_travel_times(
                source_depth_in_km=obs.get('depth', 10.0),
                distance_in_degree=obs['delta'],
                phase_list=[
                    'P','S',
                    'pP','sP',
                    'PP','PPP',
                    'SS','SSS',
                    'sS','ScS'
                ]
            )

        except Exception:
            continue

        times = {}

        for a in arr:

            if a.name not in times:
                times[a.name] = a.time

        def diff(a, b):

            if a in times and b in times:
                return times[a] - times[b]

            return None

        pred = {

            'S-P':    diff('S',   'P'),
            'PP-P':   diff('PP',  'P'),
            'PPP-P':  diff('PPP', 'P'),

            'SS-S':   diff('SS',  'S'),
            'SSS-S':  diff('SSS', 'S'),

            'ScS-S':  diff('ScS', 'S'),
            'ScS-SS': diff('ScS', 'SS'),
        }

        resid = {}

        for ph, pval in pred.items():

            if pval is None:
                continue

            if ph not in obs:
                continue

            if obs[ph] is None:
                continue

            sigma = SIGMA_dict.get(ph, 1.0)

            raw_resid = pval - obs[ph]

            resid[ph] = {
                'raw': raw_resid,
                'norm': raw_resid / sigma
            }

        resid_dict[event] = {
            'delta': obs['delta'],
            'resid': resid,
            'is_bml': bd is not None
        }

    # ========================================================
    # save cache
    # ========================================================

    np.savez_compressed(
        cache_file,
        resid_dict=resid_dict
    )

    return resid_dict

# ============================================================
# 畫圖
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

best_resid = compute_residuals('best_fit')

# ============================================================
# each phase = one subplot
# ============================================================

for ax, phase in zip(axes.flatten(), phases_to_plot):

    sigma = SIGMA_dict.get(phase, 1.0)

    # ========================================================
    # ensemble models
    # ========================================================

    for name in model_names:

        if name == 'best_fit':
            continue

        resid = compute_residuals(name)

        if resid is None:
            continue

        xs = [
            v['delta']
            for v in resid.values()
            if phase in v['resid']
        ]

        ys = [
            v['resid'][phase]['norm']
            for v in resid.values()
            if phase in v['resid']
        ]

        if len(xs) == 0:
            continue

        is_bml = any(v['is_bml'] for v in resid.values())

        color = 'royalblue' if is_bml else 'darkorange'

        ax.scatter(
            xs,
            ys,
            color=color,
            alpha=0.15,
            s=15,
            zorder=2
        )

    # ========================================================
    # best fit
    # ========================================================

    if best_resid:

        xs_best = [
            v['delta']
            for v in best_resid.values()
            if phase in v['resid']
        ]

        ys_best = [
            v['resid'][phase]['norm']
            for v in best_resid.values()
            if phase in v['resid']
        ]

        ax.scatter(
            xs_best,
            ys_best,
            color='red',
            s=70,
            zorder=5,
            marker='^',
            label='Best fit'
        )

    # ========================================================
    # reference lines
    # ========================================================

    ax.axhline(0,
               color='k',
               lw=1.2,
               ls='--')

    ax.axhline(1,
               color='gray',
               lw=0.8,
               ls=':')

    ax.axhline(-1,
               color='gray',
               lw=0.8,
               ls=':',
               label='±1σ')

    # ========================================================
    # labels
    # ========================================================

    ax.set_xlabel('Epicentral distance (°)')

    ax.set_ylabel('Normalized residual\n(pred−obs)/σ')

    ax.set_title(
        f'{phase}  (n={len(xs_best)})'
    )

    ax.legend(fontsize=8)

# ============================================================
# turn off empty panels
# ============================================================

for ax in axes.flatten()[len(phases_to_plot):]:
    ax.axis('off')

# ============================================================
# title
# ============================================================

plt.suptitle(
    'Normalized travel-time residual vs distance\n'
    'BML ensemble (blue) vs no-BML ensemble (orange)',
    fontsize=13
)

plt.tight_layout()

if SAVE_FIG: plt.savefig(
    os.path.join(FIG_DIR, 'residual_from_cache.png'),
    dpi=180,
    bbox_inches='tight'
)

print("Saved: residual_from_cache.png")
