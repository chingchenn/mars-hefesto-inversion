#!/usr/bin/env python3
# 用法: python bml_landscape.py <參考npz 例如 .../chain_a29_XX/profile_s00916.npz>
import importlib.util, sys, io, contextlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

M59 = f"{HERE}/59.mcmc_nGibbs_Mars_a25.py"
spec = importlib.util.spec_from_file_location("m59", M59)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

MOI_BIAS = 0.00136
cache, CMB = m._samuel_cache, m.TRUE_CMB_DEPTH

# 固定參考地函(取自代表 npz,底端線性外插到任意 bml_top)
d = np.load(sys.argv[1])
man_z, man_rho = d['depth_km'], d['rho']
T_MB = float(d['T_K'][-1])
sl = (man_rho[-1]-man_rho[-8]) / (man_z[-1]-man_z[-8])
def mantle_for(top):
    z = np.linspace(0.0, top, 600)
    rho = np.where(z <= man_z[-1], np.interp(z, man_z, man_rho),
                   man_rho[-1] + sl*(z - man_z[-1]))
    return {'depth_km': z, 'rho': rho}

TC = np.linspace(1900., 2900., 41)
XB = np.linspace(0.30,  0.80, 33)
DS = [150.0, 220.0, 300.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
for ax, D in zip(axes, DS):
    f56 = mantle_for(CMB - D)
    moiS = np.full((len(XB), len(TC)), np.nan)
    masS = np.full_like(moiS, np.nan); fliq = np.full_like(moiS, np.nan)
    for i, X in enumerate(XB):
        for j, Tc in enumerate(TC):
            with contextlib.redirect_stdout(io.StringIO()):
                bml = m.run_ngibbs_bml({'BML_thickness': D, 'Mg#_bulk_bml': X},
                                       Tc, T_MB, CMB)
            if bml is None: continue
            M, moi = m.compute_mass_and_moi(f56, cache, bml)
            moiS[i,j] = (moi - MOI_BIAS - m.MOI_OBS) / m.MOI_SIGMA
            masS[i,j] = (M - m.MARS_MASS_OBS) / m.MARS_MASS_SIGMA
            fliq[i,j] = bml['h_liquid_km'] / D
        print(f"D={D:.0f} X={X:.3f}")
    pc = ax.pcolormesh(TC, XB, moiS, cmap='RdBu_r', vmin=-6, vmax=6, shading='auto')
    ax.contour(TC, XB, moiS, levels=[-1,0,1], colors='k', linewidths=[1,2,1])
    ax.contour(TC, XB, masS, levels=[-1,1], colors='limegreen',
               linestyles='--', linewidths=1.5)
    ax.contour(TC, XB, fliq, levels=[0.001,0.999], colors='orange', linewidths=1.2)
    ax.set_title(f'D = {D:.0f} km', fontsize=14)
    ax.set_xlabel('T_core (K)', fontsize=14)
    ax.tick_params(labelsize=14)
axes[0].set_ylabel('Mg#_bulk_bml', fontsize=14)
cb = fig.colorbar(pc, ax=axes, label='(MoI − bias − obs)/σ')
cb.ax.tick_params(labelsize=14)
if SAVE_FIG: plt.savefig(f'{FIG_DIR}/bml_landscape.png', dpi=150, bbox_inches='tight')
print("saved bml_landscape.png")