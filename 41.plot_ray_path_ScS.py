#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:44:46 2026

@author: chingchen
"""

"""
plot_ScS_simple.py
比較 S0185a 和 S1222a 的 S 和 ScS 射線路徑
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from obspy.taup.taup_create import build_taup_model
from obspy.taup import TauPyModel


# ─────────────────────────────────────────────────
# build velocity model
# ─────────────────────────────────────────────────
DATA_DIR   = "/Users/chingchen/Desktop/Lunar/Mars_Samuel_2023/Nature_Samuel/METADATA_BML/DATA_FIG2/PANEL_B"
OUTPUT_ND  = "/Users/chingchen/Desktop/Lunar/mars_seismic_data/mars_samuel_median.nd"       
OUTPUT_NPZ = "/Users/chingchen/Desktop/Lunar/mars_seismic_data/mars_samuel_median"          

MARS_RADIUS = 3389.5   # km

if not os.path.exists(OUTPUT_NPZ + ".npz"):
    print("Build TauPy model...")
    build_taup_model(OUTPUT_ND, output_folder="/Users/chingchen/Desktop/Lunar/mars_seismic_data/")
    print(f"mars_samuel_median.npz")
else:
    print("✓model exist")


# ─────────────────────────────────────────────────
# 1. 輸入震源參數
# ─────────────────────────────────────────────────
MODEL_PATH = f"{OUTPUT_NPZ}.npz"   

S0185a_delta = 55.4    
S0185a_depth = 17.4    

S1222a_delta = 37.3
S1222a_depth = 32.8

PHASES = ["S", "ScS"]
# ─────────────────────────────────────────────────
# 2. 讀模型、計算射線路徑
# ─────────────────────────────────────────────────
model = TauPyModel(model=MODEL_PATH)


# 在畫圖之前加這段檢查
# 在 get_ray_paths 之前先加這段
for name, delta, depth in [("S0185a", S0185a_delta, S0185a_depth), 
                             ("S1222a", S1222a_delta, S1222a_depth)]:
    for phase in ["S", "ScS"]:
        arr = model.get_ray_paths(
            source_depth_in_km=depth,
            distance_in_degree=delta,
            phase_list=[phase]
        )
        print(f"{name} {phase}: 找到 {len(arr)} 個到時")
        print(arr)
        if len(arr) > 0:
            print(f"  → travel time = {arr[0].time:.1f} s")
            
            
arrivals_185_S   = model.get_ray_paths(source_depth_in_km=S0185a_depth, distance_in_degree=S0185a_delta, phase_list=["S"])
arrivals_185_ScS = model.get_ray_paths(source_depth_in_km=S0185a_depth, distance_in_degree=S0185a_delta, phase_list=["ScS"])
arrivals_122_S   = model.get_ray_paths(source_depth_in_km=S1222a_depth, distance_in_degree=S1222a_delta, phase_list=["S"])
arrivals_122_ScS = model.get_ray_paths(source_depth_in_km=S1222a_depth, distance_in_degree=S1222a_delta, phase_list=["ScS"])

# 取第一個到時
path_185_S   = arrivals_185_S[0].path
path_185_ScS = arrivals_185_ScS[0].path
path_122_S   = arrivals_122_S[0].path
path_122_ScS = arrivals_122_ScS[0].path

# ─────────────────────────────────────────────────
# 3. 畫圖
# ─────────────────────────────────────────────────


            
            
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.rad2deg(path_185_S["dist"]),   path_185_S["depth"],   color="blue",   ls="-",  lw=1.5, label="S0185a  S")
ax.plot(np.rad2deg(path_185_ScS["dist"]), path_185_ScS["depth"], color="blue",   ls="--", lw=1.5, label="S0185a  ScS")
ax.plot(np.rad2deg(path_122_S["dist"]),   path_122_S["depth"],   color="red",    ls="-",  lw=1.5, label="S1222a  S")
ax.plot(np.rad2deg(path_122_ScS["dist"]), path_122_ScS["depth"], color="red",    ls="--", lw=1.5, label="S1222a  ScS")

ax.invert_yaxis()
ax.set_xlabel("Angular distance along ray (°)", fontsize=12)
ax.set_ylabel("Depth (km)", fontsize=12)
ax.set_title("S and ScS ray paths: S0185a (Δ=55°) vs S1222a (Δ=37°)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
# plt.savefig("ScS_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ScS_comparison.png")