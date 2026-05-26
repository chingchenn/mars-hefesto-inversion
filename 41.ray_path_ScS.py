#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:25:41 2026

@author: chingchen
"""

"""
build_all_mars_models.py
Build individual TauPy .npz models from 50 Monte Carlo samples
from Samuel et al. (2023)
"""

import numpy as np
import glob, os, re
from scipy.interpolate import interp1d
from obspy.taup.taup_create import build_taup_model

# ─────────────────────────────────────────────────
# Paths and parameters
# ─────────────────────────────────────────────────
DATA_DIR   = "/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel/METADATA_BML/DATA_FIG2/PANEL_B"
OUTPUT_DIR = "/Users/chingchen/Desktop/Lunar/mars_seismic_data/all_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MARS_RADIUS      = 3389.5   # km
CMB_VS_THRESHOLD = 0.1      # km/s — below this is treated as fully molten
QP, QS           = 1000.0, 500.0

# Depth axis: 5 km spacing, last point exactly at MARS_RADIUS
depth_common = np.arange(0, MARS_RADIUS, 5.0)
depth_common = np.append(depth_common, MARS_RADIUS)

# ─────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────
def interp_to_common(data, depth_common):
    """
    Interpolate a single model sample onto the common depth axis.
    Input: (N, 2) array — col0 = velocity (m/s), col1 = radius (km)
    Output: velocity array in km/s on depth_common
    """
    vel    = data[:, 0] / 1000.0       # m/s → km/s
    radius = data[:, 1]
    depth  = MARS_RADIUS - radius      # convert radius → depth

    # Sort by depth and remove duplicates (step-function models)
    sort_idx   = np.argsort(depth)
    depth, vel = depth[sort_idx], vel[sort_idx]
    _, uniq    = np.unique(depth, return_index=True)
    depth, vel = depth[uniq], vel[uniq]

    f = interp1d(depth, vel, kind="linear",
                 bounds_error=False,
                 fill_value=(vel[0], vel[-1]))
    return f(depth_common)


def write_nd(filepath, depth, vp_kms, vs_kms):
    """
    Write a TauPy .nd file.
    The fully molten BML top (Vs → 0) is treated as the CMB,
    marked with the 'outer-core' keyword so TauPy can compute ScS.
    """
    rho       = 0.32 * vp_kms + 0.77  # Birch's law density estimate (g/cm³)
    in_liquid = False

    with open(filepath, "w") as f:
        f.write("# Mars Samuel 2023 — individual model sample\n")

        for i, d in enumerate(depth):
            vs = max(vs_kms[i], 0.0)

            # Entering fully molten layer → write discontinuity
            if not in_liquid and vs < CMB_VS_THRESHOLD:
                # Line above discontinuity (mantle bottom, Vs > 0)
                f.write(f"{d:.2f}  {vp_kms[i-1]:.4f}  {vs_kms[i-1]:.4f}  "
                        f"{rho[i-1]:.4f}  {QP:.1f}  {QS:.1f}\n")
                f.write("outer-core\n")   # TauPy CMB marker
                # Line below discontinuity (fully molten, Vs = 0)
                f.write(f"{d:.2f}  {vp_kms[i]:.4f}  0.0000  "
                        f"{rho[i]:.4f}  {QP:.1f}  9999.0\n")
                in_liquid = True
                continue

            # Inside liquid layer — force Vs = 0
            if in_liquid:
                f.write(f"{d:.2f}  {vp_kms[i]:.4f}  0.0000  "
                        f"{rho[i]:.4f}  {QP:.1f}  9999.0\n")
            else:
                f.write(f"{d:.2f}  {vp_kms[i]:.4f}  {vs:.4f}  "
                        f"{rho[i]:.4f}  {QP:.1f}  {QS:.1f}\n")


# ─────────────────────────────────────────────────
# Collect all sample IDs from vp*.dat filenames
# ─────────────────────────────────────────────────
vp_files   = sorted(glob.glob(os.path.join(DATA_DIR, "vp*.dat")))
sample_ids = []
for f in vp_files:
    m = re.search(r"vp(\d+)\.dat", os.path.basename(f))
    if m:
        sample_ids.append(m.group(1))

print(f"Found {len(sample_ids)} samples: {sample_ids[:3]} ... {sample_ids[-3:]}")

# ─────────────────────────────────────────────────
# Build models
# ─────────────────────────────────────────────────
success, failed, skipped = [], [], []

for sid in sample_ids:
    vp_path  = os.path.join(DATA_DIR,   f"vp{sid}.dat")
    vs_path  = os.path.join(DATA_DIR,   f"vs{sid}.dat")
    nd_path  = os.path.join(OUTPUT_DIR, f"mars_{sid}.nd")
    npz_path = os.path.join(OUTPUT_DIR, f"mars_{sid}.npz")
    idx      = sample_ids.index(sid) + 1

    # Skip if already built
    if os.path.exists(npz_path):
        print(f"[{idx:2d}/{len(sample_ids)}] mars_{sid} already exists, skipping")
        skipped.append(sid)
        continue

    if not os.path.exists(vs_path):
        print(f"[{idx:2d}/{len(sample_ids)}] WARNING: vs{sid}.dat not found, skipping")
        failed.append(sid)
        continue

    print(f"[{idx:2d}/{len(sample_ids)}] Building mars_{sid} ...", end=" ", flush=True)

    try:
        vp_kms = interp_to_common(np.loadtxt(vp_path), depth_common)
        vs_kms = interp_to_common(np.loadtxt(vs_path), depth_common)

        write_nd(nd_path, depth_common, vp_kms, vs_kms)
        build_taup_model(nd_path, output_folder=OUTPUT_DIR)
        print("✓")
        success.append(sid)

    except Exception as e:
        print(f"✗  {e}")
        failed.append(sid)

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Done — success: {len(success)}  skipped: {len(skipped)}  failed: {len(failed)}")
if failed:
    print(f"Failed samples: {failed}")
print(f"Models saved to: {OUTPUT_DIR}/")