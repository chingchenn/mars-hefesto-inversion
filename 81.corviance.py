#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
81: proposal covariance + chain end states, for the covariance proposal
    (--prop_cov / --prop_scale in 26fall_main_mcmc_a61.py)

    python 81.corviance.py --prefix chain_a60_YM20

Paths come from mars_config.py (each machine keeps its own copy), so the same
file runs on the Mac and on beno3.

Outputs (in the current directory)
    prop_cov_<run>.npz          cov (blended, used by --prop_cov), cov_post, param_order
    start_<run>/<chain>.json    last state of each chain, for --start
    figures/prop_corr_<run>.png

Previous version (a36/a37/a60 run, hard-coded paths and PREFIX) was replaced on 2026-11-13;
its output prop_cov_a60_YM20.npz is unchanged (same Sigma, same blend).
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--prefix',    required=True)              # e.g. chain_a60_YM20
ap.add_argument('--mcmc_dir',  default=MCMC_DIR)
ap.add_argument('--burn_frac', type=float, default=0.4)    # same as 63v2
ap.add_argument('--min_steps', type=int,   default=300)
ap.add_argument('--blend',     type=float, default=0.2)    # Sigma = (1-b) Sigma_post + b diag(STEP^2)
ap.add_argument('--exclude',   nargs='*',  default=[])
ap.add_argument('--wx_step',   type=float, default=None)   # STEP of w_C / w_H (main script: 0.1*wc_max or 0.1*wh_max)
a = ap.parse_args()

RUN     = a.prefix.replace('chain_', '')
OUT_NPZ = f'prop_cov_{RUN}.npz'
START   = f'start_{RUN}'
FIG     = f'{FIG_DIR}/prop_corr_{RUN}.png'

# order must match PRIOR in the main script (propose() reads the Cholesky vector in this order)
PARAM_ORDER = ('T_lit', 'P_lit', 'Mg#', 'T_core',
               'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb')
# STEP of 26fall_main_mcmc.py (a60); only used for the diagonal blend, which keeps
# the proposal full rank and leaves some step in directions the chains never explored
STEP = {'T_lit': 40.0, 'P_lit': 0.5, 'Mg#': 0.015, 'T_core': 80.0,
        'Mg#_bulk_bml': 0.05, 'BML_thickness': 20.0, 'w_S': 0.015, 'R_cmb': 25.0}


def load_jsonl(fp):
    recs = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


# ── load ──────────────────────────────────────────────────────────────────────
# chain.jsonl stores the state, not accepted-only: rejected steps repeat the
# previous record. Keep all of them; the repeats are the posterior weight.
print(f'chain dir: {a.mcmc_dir}')
dirs = sorted(glob.glob(os.path.join(a.mcmc_dir, f'{a.prefix}_[0-9][0-9]')))
assert dirs, f'no dirs match {a.mcmc_dir}/{a.prefix}_NN'
os.makedirs(START, exist_ok=True)
rows = []
for d in dirs:
    name = os.path.basename(d)
    fp = os.path.join(d, 'chain.jsonl')
    if name in a.exclude or not os.path.exists(fp):
        print(f'  skip {name}')
        continue
    recs = load_jsonl(fp)
    # a70: chains with a 9th parameter (w_C or w_H, appended after R_cmb in the main script)
    for k9, st9 in (('w_C', 0.002), ('w_H', 0.001)):
        if recs and k9 in recs[0]['params'] and k9 not in PARAM_ORDER:
            PARAM_ORDER = PARAM_ORDER + (k9,)
            STEP[k9] = a.wx_step if a.wx_step is not None else st9
            print(f'  {k9} found: {len(PARAM_ORDER)} parameters')
    if len(recs) < a.min_steps:
        print(f'  skip {name}: {len(recs)} steps')
        continue
    kept = recs[int(len(recs) * a.burn_frac):]
    rows += [[r['params'][k] for k in PARAM_ORDER] for r in kept]
    with open(os.path.join(START, f'{name}.json'), 'w') as f:
        json.dump({k: float(recs[-1]['params'][k]) for k in PARAM_ORDER}, f, indent=1)
    print(f'  {name}: {len(recs)} steps, kept {len(kept)}, '
          f'last misfit {recs[-1]["misfit"]:.3f}')

X = np.asarray(rows)
n = len(X)
print(f'\ntotal samples: {n}')
assert n > 500, 'too few samples for a stable 8x8 covariance'

# ── covariance ────────────────────────────────────────────────────────────────
# pooled over chains on purpose: the between-chain spread is the solid <-> molten
# separation, which is the direction the proposal has to learn
cov_post = np.cov(X, rowvar=False)
cov_prop = (1 - a.blend) * cov_post + a.blend * np.diag([STEP[k]**2 for k in PARAM_ORDER])
np.linalg.cholesky(cov_prop)                     # must be positive definite
sd   = np.sqrt(np.diag(cov_post))
corr = cov_post / np.outer(sd, sd)

print('\nposterior sd vs STEP:')
for i, k in enumerate(PARAM_ORDER):
    print(f'  {k:14s}  sd={sd[i]:10.4f}   STEP={STEP[k]:8.3f}   ratio={sd[i]/STEP[k]:5.2f}')

print('\n|corr| > 0.5:')
for i in range(len(PARAM_ORDER)):
    for j in range(i + 1, len(PARAM_ORDER)):
        if abs(corr[i, j]) > 0.5:
            print(f'  {PARAM_ORDER[i]:14s} {PARAM_ORDER[j]:14s} {corr[i, j]:+.3f}')

w, V = np.linalg.eigh(corr)
print('\ntop eigenvector of corr:')
for i, k in enumerate(PARAM_ORDER):
    print(f'  {k:14s} {V[i, -1]:+.3f}')

prop_sd = np.sqrt(np.diag(2.38**2 / len(PARAM_ORDER) * cov_prop))
print('\nproposal step sd  (--prop_scale 1.0 / 0.25):')
for k, s in zip(PARAM_ORDER, prop_sd):
    print(f'  {k:14s} {s:10.4g} / {s/2:10.4g}')

np.savez(OUT_NPZ, cov=cov_prop, cov_post=cov_post,
         param_order=np.array(PARAM_ORDER), n=n, blend=a.blend)
print(f'\nsaved {OUT_NPZ} and {START}/ ({len(os.listdir(START))} files)')

# ── figure ────────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
nP = len(PARAM_ORDER)                                   # 8, or 9 with w_C / w_H
ax.set_xticks(range(nP)); ax.set_yticks(range(nP))
ax.set_xticklabels(PARAM_ORDER, rotation=45, ha='right')
ax.set_yticklabels(PARAM_ORDER)
ax.tick_params(labelsize=14)
for i in range(nP):
    for j in range(nP):
        ax.text(j, i, f'{corr[i, j]:+.2f}', ha='center', va='center',
                fontsize=9, color='k' if abs(corr[i, j]) < 0.6 else 'w')
cb = fig.colorbar(im, ax=ax)
cb.ax.tick_params(labelsize=14)
ax.set_title(f'{a.prefix}  posterior correlation  (n={n})', fontsize=14)
fig.tight_layout()
if SAVE_FIG: fig.savefig(FIG, dpi=150)
print(f'saved {FIG}')
