# ============================================================
# plot_differential_residual.py
#
# 畫：Δt_BML − Δt_noBML  vs  epicentral distance
#
# positive  → BML model predicts MORE delayed arrivals
# negative  → BML model predicts FASTER arrivals
# ============================================================

import os
import glob
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

# ------------------------------------------------------------
# load mcmc module
# ------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "mcmc",
    "/home/jcchen2/mars-hefesto-inversion/15.mcmc_withBML_MOI.py")
mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

os.chdir('/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc')
mcmc.load_gravity_profile()
khan_cache = mcmc.compute_khan_median()

# ------------------------------------------------------------
# config
# ------------------------------------------------------------
CACHE_DIR       = 'cache'
RESID_CACHE_DIR = 'residual_cache'
os.makedirs(RESID_CACHE_DIR, exist_ok=True)

BML_MODEL   = 'best_fit_bml'
NOBML_MODEL = 'best_fit_noBML'

phases_to_plot = [
    'S-P',
    'PP-P',
    'PPP-P',
    'SS-S',
    'SSS-S',
    'ScS-S',
    'ScS-SS',
]

SIGMA_dict = mcmc.SIGMA

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def load_best_model(model_dir):

    fort56_path = f'{model_dir}/fort.56'

    fort56_data = mcmc.read_fort56(
        fort56_path
    )

    return fort56_data


def compute_residuals(model_name):
    cache_file = f'{RESID_CACHE_DIR}/{model_name}_resid.npz'

    if os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=True)
        return d['resid_dict'].item()

    result = load_from_cache(model_name)
    if result is None:
        return None
    fd, bd = result

    try:
        taup = mcmc.build_taup(fd, model_name, khan_cache, bml_data=bd)
    except Exception as e:
        print(f"TauP failed for {model_name}: {e}")
        return None

    resid_dict = {}

    for event, obs in mcmc.SAMUEL_DATA.items():
        try:
            arr = taup.get_travel_times(
                source_depth_in_km=obs.get('depth', 10.0),
                distance_in_degree=obs['delta'],
                phase_list=['P','S','pP','sP','PP','PPP','SS','SSS','sS','ScS']
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
            if pval is None or ph not in obs or obs[ph] is None:
                continue
            sigma    = SIGMA_dict.get(ph, 1.0)
            raw_resid = pval - obs[ph]
            resid[ph] = {'raw': raw_resid, 'norm': raw_resid / sigma}

        resid_dict[event] = {
            'delta':  obs['delta'],
            'resid':  resid,
            'is_bml': bd is not None,
        }

    np.savez_compressed(cache_file, resid_dict=resid_dict)
    return resid_dict


# ------------------------------------------------------------
# compute residuals for both models
# ------------------------------------------------------------
print(f"Computing residuals for {BML_MODEL} ...")
bml_resid = compute_residuals(BML_MODEL)

print(f"Computing residuals for {NOBML_MODEL} ...")
nobml_resid = compute_residuals(NOBML_MODEL)

if bml_resid is None or nobml_resid is None:
    sys.exit("ERROR: could not load residuals for one or both models.")

# ------------------------------------------------------------
# figure  (3 × 3 grid; last panel left blank if 7 phases)
# ------------------------------------------------------------
ncols, nrows = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 13))
fig.suptitle(
    r'$\Delta t_{\rm BML} - \Delta t_{\rm noBML}$  vs  epicentral distance',
    fontsize=16, y=0.98
)

common_events = set(bml_resid.keys()) & set(nobml_resid.keys())
print(f"Common events: {len(common_events)}")

BIN_EDGES    = np.arange(15, 65, 5)   # [15,20,25,...,60]
BIN_CENTERS  = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

for ax_idx, phase in enumerate(phases_to_plot):

    ax = axes.flatten()[ax_idx]

    # ---- collect Δresid = resid_BML − resid_noBML --------
    deltas = []
    diffs  = []

    for ev in common_events:
        rb = bml_resid[ev]['resid']
        rn = nobml_resid[ev]['resid']

        if phase not in rb or phase not in rn:
            continue

        delta = bml_resid[ev]['delta']
        d_val = (
            np.abs(rb[phase]['raw'])
            -
            np.abs(rn[phase]['raw'])
        )

        deltas.append(delta)
        diffs.append(d_val)

    deltas = np.array(deltas)
    diffs  = np.array(diffs)

    # ---- scatter of individual events --------------------
    ax.scatter(
        deltas, diffs,
        color='lightcoral', s=40, alpha=0.6,
        zorder=2, label='events'
    )

    # ---- zero reference line -----------------------------
    ax.axhline(0, color='k', lw=0.8, ls='--', zorder=1)

    # ---- distance-bin statistics -------------------------
    if len(deltas) > 0:
        bin_centers = []
        bin_means   = []
        bin_stds    = []

        for i in range(len(BIN_EDGES) - 1):
            mask = (deltas >= BIN_EDGES[i]) & (deltas < BIN_EDGES[i + 1])
            n    = np.sum(mask)
            if n < 2:
                continue
            bin_centers.append(0.5 * (BIN_EDGES[i] + BIN_EDGES[i + 1]))
            bin_means.append(np.mean(diffs[mask]))
            bin_stds.append(np.std(diffs[mask], ddof=1))

        bin_centers = np.array(bin_centers)
        bin_means   = np.array(bin_means)
        bin_stds    = np.array(bin_stds)

        # ± 1σ shading
        ax.fill_between(
            bin_centers,
            bin_means - bin_stds,
            bin_means + bin_stds,
            color='darkred', alpha=0.20, zorder=3
        )

        # mean trend line
        ax.plot(
            bin_centers, bin_means,
            color='darkred', lw=2.5,
            marker='o', ms=5, zorder=5,
            label='bin mean ± 1σ'
        )

    # ---- annotations -------------------------------------
    ax.set_title(phase, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epicentral distance (°)', fontsize=9)
    ax.set_ylabel(r'$\Delta t_{\rm BML} - \Delta t_{\rm noBML}$  (s)', fontsize=9)
    ax.set_xlim(10, 65)
    ax.tick_params(labelsize=8)

    if ax_idx == 0:
        ax.legend(fontsize=7, loc='upper left')

    # sample-count annotation
    ax.text(
        0.97, 0.97, f'N={len(deltas)}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=8, color='gray'
    )

# ---- hide unused subplots --------------------------------
for ax_idx in range(len(phases_to_plot), nrows * ncols):
    axes.flatten()[ax_idx].set_visible(False)

# ------------------------------------------------------------
# save
# ------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_png = 'differential_residual_BML_vs_noBML.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved → {out_png}")
plt.close(fig)