import os
import sys
import json
import glob
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# config
# ============================================================
MCMC_BASE       = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
HEFESTO_PREFIX  = 'chain_a23'
N_BEST          = 20
NGIBBS_MODULE   = '/home/jcchen2/mars-hefesto-inversion/57.mcmc_nGibbs_Mars.py'
OUT_DIR         = '/net/beno3/data1/jcchen2/mars-hefesto-runs/compare_a23_ngibbs'
RUN_DIR         = os.path.join(OUT_DIR, 'runs')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

spec = importlib.util.spec_from_file_location('ng57', NGIBBS_MODULE)
ng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng)

# ============================================================
# top-N unique params from HeFESTo chains
# ============================================================
records = []
for fpath in sorted(glob.glob(f'{MCMC_BASE}/{HEFESTO_PREFIX}_*/chain.json')):
    chain = json.load(open(fpath))
    for s in chain:
        if s.get('accepted'):
            records.append({'misfit': s['misfit'], 'step': s['step'],
                            'chain_dir': os.path.dirname(fpath),
                            'params': s['params']})

records.sort(key=lambda x: x['misfit'])
seen, top = set(), []
for r in records:
    key = tuple(round(v, 6) for v in r['params'].values())
    if key in seen:
        continue
    seen.add(key)
    top.append(r)
    if len(top) >= N_BEST:
        break

print(f"HeFESTo {HEFESTO_PREFIX}: {len(records)} accepted, "
      f"top {len(top)} unique, misfit {top[0]['misfit']:.4f} - {top[-1]['misfit']:.4f}")

# ============================================================
# run nGibbs on same params, plot mantle+BML overlay per model
# ============================================================
C_HF = 'k'
C_NG = '#E74C3C'

for k, r in enumerate(top):
    params = r['params']
    name   = f"cmp_{k:02d}"
    print(f"\n[{k+1}/{len(top)}] hef_misfit={r['misfit']:.4f}  "
          f"T_lit={params['T_lit']:.0f} P_lit={params['P_lit']:.2f} "
          f"Mg#={params['Mg#']:.3f} BML={params['BML_thickness']:.0f}km")

    misfit, n_data, comp, fort, bml = ng.forward(
        params, os.path.join(RUN_DIR, name), name, ng._samuel_cache)
    if misfit is None:
        print("  nGibbs forward FAILED, skip")
        continue

    hef_path = os.path.join(r['chain_dir'], f"profile_s{r['step']:05d}.npz")
    if not os.path.exists(hef_path):
        print(f"  missing {hef_path}, skip plot")
        continue
    hf = np.load(hef_path)

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    panels = [('rho (g/cm$^3$)', 'rho', 'bml_rho'),
              ('Vp (km/s)',      'Vp',  'bml_Vp'),
              ('Vs (km/s)',      'Vs',  'bml_Vs'),
              ('T (K)',          'T_K', 'bml_T_K')]

    for ax, (xlabel, key, bkey) in zip(axes, panels):
        P_hf = np.concatenate([hf['P_GPa'], hf['bml_P_GPa']])
        v_hf = np.concatenate([hf[key],     hf[bkey]])
        P_ng = np.concatenate([fort['P_GPa'], bml['P_GPa']])
        v_ng = np.concatenate([fort[key],     bml[bkey.replace('bml_', '')]])
        ax.plot(v_hf, P_hf, color=C_HF, lw=2.2, ls='--', label='HeFESTo', zorder=3)
        ax.plot(v_ng, P_ng, color=C_NG, lw=1.5, label='nGibbs', zorder=4)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.tick_params(labelsize=14)
        ax.invert_yaxis()

    axes[0].set_ylabel('P (GPa)', fontsize=14)
    axes[0].legend(fontsize=12)
    fig.suptitle(f"rank {k}   HeFESTo={r['misfit']:.3f}   nGibbs={misfit:.3f}   "
                 f"T_lit={params['T_lit']:.0f}K  Mg#={params['Mg#']:.3f}  "
                 f"BML={params['BML_thickness']:.0f}km", fontsize=14)
    out = os.path.join(OUT_DIR, f'profile_rank{k:02d}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")