#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-
# 137: AxiSEM3D install check, Example 00 (global 1D PREM, 50 s, 2011 Virginia Mw 5.8)
#
#   python 137.axisem3d_ex00_check.py
#
# Run the example first (conda env axisem3d):
#   cd $AXISEM3D_DIR/examples/00_global_1D
#   mpirun -np 4 ../../build/axisem3d --input input --output output
#
# Output format (ASCII_STATION): one file per station <NET>.<STA>.ascii,
# columns = U1 U2 U3 in the RTZ frame (radial, transverse, vertical displacement, m),
# time axis in data_time.ascii (s after origin; starts negative because of the erf STF).
# Compare the figure with $AXISEM3D_DIR/examples/00_global_1D/figures/ex00_seismograms_ANMO.png
import numpy as np
import matplotlib.pyplot as plt
from mars_config import AXISEM3D_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

STA   = 'IU.ANMO'
D     = f'{AXISEM3D_DIR}/examples/00_global_1D/output/stations/global_seismic_network_GSN'
COMPS = ['R', 'T', 'Z']

t = np.loadtxt(f'{D}/data_time.ascii')
u = np.loadtxt(f'{D}/{STA}.ascii')
print(f'{STA}: {len(t)} samples, dt = {t[1] - t[0]:.4f} s, t = [{t[0]:.1f}, {t[-1]:.1f}] s')
for i, c in enumerate(COMPS):
    print(f'  max |U{c}| = {np.abs(u[:, i]).max():.3e} m at t = {t[np.argmax(np.abs(u[:, i]))]:.1f} s')

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for i, c in enumerate(COMPS):
    ax[i].plot(t, u[:, i], 'k-', lw=0.8)
    ax[i].set_ylabel(f'U{c} (m)', fontsize=14)
    ax[i].tick_params(labelsize=14)
ax[2].set_xlabel('Time after origin (s)', fontsize=14)
ax[2].set_xlim(t[0], t[-1])
ax[0].set_title(f'AxiSEM3D Example 00, {STA}, PREM 50 s', fontsize=14)
fig.tight_layout()
if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, '137_axisem3d_ex00_ANMO.png'), dpi=200)
plt.show()
