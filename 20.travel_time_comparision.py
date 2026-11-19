#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare travel-time predictions:
    observed InSight data
    best-fit BML model from chain_a04
    best-fit no-BML model from chain_a03

Run inside:
    /net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/

Output:
    travel_time_distance_BML_vs_noBML.png
"""

import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, RUN_BASE, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script


# ============================================================
# 0. Import two MCMC modules
# ============================================================

BML_MODULE_PATH = f"{HERE}/15.mcmc_withBML_MOI.py"
NOBML_MODULE_PATH = f"{HERE}/13.mcmc_entropy_MoI.py"

WORKDIR = MCMC_DIR
os.chdir(WORKDIR)


from importlib.machinery import SourceFileLoader

def import_module_from_path(name, path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mod = SourceFileLoader(name, path).load_module()

    return mod


mcmc_bml = import_module_from_path("mcmc_bml", BML_MODULE_PATH)
mcmc_no = import_module_from_path("mcmc_no", NOBML_MODULE_PATH)


# ============================================================
# 1. Utility: find best model
# ============================================================

def find_best(chain_pattern, label):
    all_steps = []

    for f in sorted(glob.glob(chain_pattern)):
        try:
            with open(f) as fp:
                chain = json.load(fp)
        except Exception as e:
            print(f"Cannot read {f}: {e}")
            continue

        all_steps.extend([
            s for s in chain
            if s.get("misfit", 999.0) < 10
        ])

    if not all_steps:
        raise RuntimeError(f"No valid steps found for {label}: {chain_pattern}")

    best = min(all_steps, key=lambda x: x["misfit"])

    print("=" * 70)
    print(f"{label} best fit")
    print(f"  Number of valid steps: {len(all_steps)}")
    print(f"  Best misfit: {best['misfit']:.4f}")
    print(f"  Best params: {best['params']}")

    return best


# ============================================================
# 2. Utility: calculate differential travel times
# ============================================================

PHASES_LIST = [
    "P", "S", "pP", "sP",
    "PP", "PPP",
    "SS", "SSS", "sS",
    "ScS", "SKS",
]

PHASES_TO_PLOT = ["S-P", "PP-P", "SS-S", "ScS-S","PPP-P", "SSS-S", "sS-S", "pP-P"]


def compute_predictions(taup_model, samuel_data):
    predictions = {}

    for event, obs in samuel_data.items():
        delta = obs["delta"]
        depth = obs.get("depth", 10.0)

        try:
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=depth,
                distance_in_degree=delta,
                phase_list=PHASES_LIST,
            )
        except Exception as e:
            print(f"TauP failed for {event}: {e}")
            predictions[event] = {}
            continue

        times = {}
        for a in arrivals:
            if a.name not in times:
                times[a.name] = a.time

        def diff(a, b):
            if a in times and b in times:
                return times[a] - times[b]
            return None

        predictions[event] = {
            "S-P":    diff("S", "P"),
            "PP-P":   diff("PP", "P"),
            "PPP-P":  diff("PPP", "P"),
            "SS-S":   diff("SS", "S"),
            "SSS-S":  diff("SSS", "S"),
            "ScS-S":  diff("ScS", "S"),
            "SS-PP":  diff("SS", "PP"),
            "SKS-PP": diff("SKS", "PP"),
            "pP-P":   diff("pP", "P"),
            "sS-S":   diff("sS", "S"),
        }

    return predictions


# ============================================================
# 3. Build no-BML TauP prediction
# ============================================================
def build_nobml_predictions(best_no, khan_cache_no):

    params = best_no["params"]

    run_dir = (
        f"{RUN_BASE}/saved_models/best_noBML"
    )

    os.makedirs(run_dir, exist_ok=True)

    fort56 = os.path.join(
        run_dir,
        "s2_final",
        "fort.56",
    )

    # ======================================================
    # Reuse existing HeFESTo result
    # ======================================================

    if os.path.exists(fort56):

        print("Using existing no-BML HeFESTo result")

    else:

        print("Running HeFESTo for no-BML best fit...")

        fort56 = mcmc_no.run_hefesto(
            params,
            run_dir,
        )

    # ======================================================
    # Safety check
    # ======================================================

    if fort56 is None:
        raise RuntimeError("No-BML HeFESTo failed")

    # ======================================================
    # Read fort56
    # ======================================================

    fort56_data = mcmc_no.read_fort56(fort56)

    if fort56_data is None:
        raise RuntimeError("No-BML fort.56 read failed")

    # ======================================================
    # Build TauP
    # ======================================================

    taup_no = mcmc_no.build_taup(
        fort56_data,
        "best_fit_noBML_plot",
        khan_cache_no,
    )

    return compute_predictions(
        taup_no,
        mcmc_no.SAMUEL_DATA,
    )

# ============================================================
# 4. Build BML TauP prediction
# ============================================================
def build_bml_predictions(best_bml, khan_cache_bml):

    params = best_bml["params"]

    # ======================================================
    # Permanent folders
    # ======================================================

    mantle_run_dir = (
        f"{RUN_BASE}/saved_models/best_BML"
    )

    bml_run_dir = (
        f"{RUN_BASE}/saved_models/best_BML_layer"
    )

    os.makedirs(mantle_run_dir, exist_ok=True)
    os.makedirs(bml_run_dir, exist_ok=True)

    # ======================================================
    # Mantle fort.56 path
    # ======================================================

    fort56 = os.path.join(
        mantle_run_dir,
        "s2_final",
        "fort.56",
    )

    # ======================================================
    # Reuse existing mantle HeFESTo result
    # ======================================================

    if os.path.exists(fort56):

        print("Using existing BML mantle HeFESTo result")

    else:

        print("Running BML mantle HeFESTo...")

        fort56 = mcmc_bml.run_hefesto(
            params,
            mantle_run_dir,
        )

    # ======================================================
    # Safety check
    # ======================================================

    if fort56 is None:
        raise RuntimeError("BML mantle HeFESTo failed")

    # ======================================================
    # Read mantle fort56
    # ======================================================

    fort56_data = mcmc_bml.read_fort56(fort56)

    if fort56_data is None:
        raise RuntimeError("BML mantle fort.56 read failed")

    # ======================================================
    # BML layer
    # ======================================================

    bml_data = None

    fort56_bml = os.path.join(
        bml_run_dir,
        "s2_final",
        "fort.56",
    )

    P_bml_bottom = mcmc_bml.P_BML_FIXED
    P_bml_top = P_bml_bottom - 1.95

    cmb_depth = khan_cache_bml["cmb_depth"]

    # ======================================================
    # Reuse existing BML layer HeFESTo result
    # ======================================================

    if os.path.exists(fort56_bml):

        print("Using existing BML layer HeFESTo result")

        fort56_bml_path = fort56_bml

    else:

        print("Running BML layer HeFESTo...")

        fort56_bml_path = mcmc_bml.run_hefesto_bml(
            params={
                "T_lit": params["T_bml"],
                "P_lit": P_bml_top,
                "Mg#": params["Mg#_bml"],
            },
            run_dir=bml_run_dir,
            P_top=P_bml_top,
            P_bottom=P_bml_bottom,
            T_bml=params["T_bml"],
        )

    # ======================================================
    # Read BML fort56
    # ======================================================

    if fort56_bml_path:

        bml_raw = mcmc_bml.read_fort56(
            fort56_bml_path
        )

        if bml_raw:

            bml_top_km = (
                cmb_depth
                - mcmc_bml.BML_THICKNESS
            )

            bml_raw["depth_km"] = (
                bml_raw["depth_km"]
                + bml_top_km
            )

            rho_mantle_bottom = float(
                fort56_data["rho"][-1]
            )

            rho_bml_top = float(
                bml_raw["rho"][0]
            )

            # ==============================================
            # Density stability check
            # ==============================================

            if rho_bml_top > rho_mantle_bottom:

                bml_data = bml_raw

                print(
                    "Stable BML density structure accepted"
                )

            else:

                print(
                    "BML density unstable; "
                    "plotting without BML layer"
                )

        else:

            print("BML fort.56 read failed")

    else:

        print("BML HeFESTo failed")

    # ======================================================
    # Build TauP
    # ======================================================

    taup_bml = mcmc_bml.build_taup(
        fort56_data,
        "best_fit_BML_plot",
        khan_cache_bml,
        bml_data=bml_data,
    )

    return compute_predictions(
        taup_bml,
        mcmc_bml.SAMUEL_DATA,
    )


# ============================================================
# save predictions
# ============================================================
def save_predictions(predictions, filepath):
    # predictions 裡有 None 值，json 可以處理
    with open(filepath, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {filepath}")

# ============================================================
# read predictions
# ============================================================
def load_predictions(filepath):
    with open(filepath) as f:
        return json.load(f)
    

# ============================================================
# 5. Main
# ============================================================

def main():
    os.makedirs("cache", exist_ok=True)

    # Find best models
    best_bml = find_best("chain_a04_*/chain.json", "BML / chain_a04")
    best_no = find_best("chain_a03_*/chain.json", "no-BML / chain_a03")

    # Load shared model info
    print("=" * 70)
    print("Loading gravity and Khan cache")

    mcmc_bml.load_gravity_profile()
    khan_cache_bml = mcmc_bml.compute_khan_median()

    mcmc_no.load_gravity_profile()
    khan_cache_no = mcmc_no.compute_khan_median()

    # Forward predictions
    print("=" * 70)
    print("Building BML predictions")
    if os.path.exists('pred_bml.json'):
        pred_bml = load_predictions('pred_bml.json')
        print("Loaded pred_bml from cache")
    else:
        pred_bml = build_bml_predictions(best_bml, khan_cache_bml)
        save_predictions(pred_bml, 'pred_bml.json')


    print("=" * 70)
    print("Building no-BML predictions")
    if os.path.exists('pred_no.json'):
        pred_no = load_predictions('pred_no.json')
        print("Loaded pred_no from cache")
    else:
        pred_no = build_nobml_predictions(best_no, khan_cache_no)
        save_predictions(pred_no, 'pred_no.json')


    # Use BML module Samuel data as plotting data
    data = mcmc_bml.SAMUEL_DATA

    # ========================================================
    # Plot
    # ========================================================
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    for ax, phase in zip(axes.flatten(), PHASES_TO_PLOT):
        deltas_obs, vals_obs = [], []
        deltas_bml, vals_bml = [], []
        deltas_no, vals_no = [], []

        for event, obs in data.items():
            if phase not in obs or obs[phase] is None:
                continue

            delta = obs["delta"]
            obs_val = obs[phase]

            deltas_obs.append(delta)
            vals_obs.append(obs_val)

            bml_val = pred_bml.get(event, {}).get(phase)
            no_val = pred_no.get(event, {}).get(phase)

            if bml_val is not None:
                deltas_bml.append(delta)
                vals_bml.append(bml_val)

            if no_val is not None:
                deltas_no.append(delta)
                vals_no.append(no_val)

        ax.scatter(
            deltas_obs,
            vals_obs,
            color="black",
            s=60,
            marker="o",
            zorder=5,
            label="InSight observed",
        )

        ax.scatter(
            deltas_bml,
            vals_bml,
            color="blue",
            s=60,
            marker="^",
            zorder=4,
            label="Best fit BML",
        )

        ax.scatter(
            deltas_no,
            vals_no,
            color="red",
            s=55,
            marker="s",
            zorder=3,
            label="Best fit no BML",
        )

        ax.set_xlabel("Epicentral distance (°)")
        ax.set_ylabel(f"{phase} time (s)")
        ax.set_title(f"{phase}  (n={len(deltas_obs)})")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Travel time vs distance: BML vs no-BML best fits\n"
        f"BML misfit={best_bml['misfit']:.4f}, "
        f"Mg#={best_bml['params']['Mg#']:.3f}, "
        f"Mg#_bml={best_bml['params'].get('Mg#_bml', np.nan):.3f}, "
        f"T_bml={best_bml['params'].get('T_bml', np.nan):.0f} K\n"
        f"no-BML misfit={best_no['misfit']:.4f}, "
        f"Mg#={best_no['params']['Mg#']:.3f}",
        fontsize=11,
    )

    plt.tight_layout()
    out = f"{FIG_DIR}/travel_time_distance_BML_vs_noBML_v2.png"
    if SAVE_FIG: plt.savefig(out, dpi=150, bbox_inches="tight")
    print("=" * 70)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()