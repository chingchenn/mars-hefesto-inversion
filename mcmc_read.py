# mcmc_read.py : one shared reader for MCMC chains (chain.jsonl, a34 onward)
#
#   import mcmc_read as mr
#   post = mr.load_run('chain_a60_YM20', exclude=['chain_a60_YM20_03'])
#   post['T_core'], post['misfit'], post['state'], post['chain'], post['profile_step']
#
# Rules (same as 63v2 / 89 / 113, see CLAUDE.md):
#   - chain.jsonl stores the chain STATE at every step; a rejected step repeats the
#     previous state. Every record is kept, repeats included. Never filter to accepted.
#   - per chain: drop the last DROP_LAST lines (may be half written), skip chains with
#     fewer than MIN_STEPS records, then discard the burn-in: burnin < 1 is a fraction
#     of the records, burnin >= 1 is a number of records (per chain, not the 'step' field).
#   - BML state: solid if h_liquid < 1 km, molten if h_solid < 1 km, layered otherwise,
#     'fail' if both < 1 km (or missing).
#   - profiles exist only at accepted steps. Record i uses profile_s{profile_step[i]:05d}.npz,
#     the last accepted step <= i. Drawing records uniformly therefore weights each
#     profile by its dwell time; profile_table() gives the explicit weights.
#
# All arrays in post have the same length N (post burn-in records of all chains), so
# they can be masked together:  m = post['state'] == 'molten'; np.median(post['T_core'][m])

import os, glob, json
import numpy as np
from mars_config import MCMC_DIR

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb']
H_STATE = 1.0          # km, state threshold


def load_run(prefix, exclude=(), burnin=0.4, min_steps=400, drop_last=1,
             chain_dir=MCMC_DIR, verbose=True):
    # burnin < 1  : fraction of each chain's records to discard (0.4 = first 40%)
    # burnin >= 1 : number of records to discard per chain (e.g. 2000)
    # ── read every chain folder prefix_* ────────────────────────────────────────
    chains = []
    for d in sorted(glob.glob(os.path.join(chain_dir, prefix + '_*'))):
        name = os.path.basename(d)
        path = os.path.join(d, 'chain.jsonl')
        if name in exclude or not os.path.exists(path):
            if verbose: print(f'  {name:<24s} skipped')
            continue
        recs = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break                       # half-written line: stop here
        recs = recs[:-drop_last or None]
        if len(recs) < min_steps:
            if verbose: print(f'  {name:<24s} skipped ({len(recs)} steps < {min_steps})')
            continue
        chains.append((d, recs))
    if not chains:
        raise SystemExit(f'no usable chains for {prefix} in {chain_dir}')

    # ── burn-in per chain; profile step = last accepted step <= i ───────────────
    keep, cid, cdir, pstep = [], [], [], []
    for c, (d, recs) in enumerate(chains):
        b = int(len(recs) * burnin) if burnin < 1 else int(burnin)
        if b >= len(recs) and verbose:
            print(f'  {os.path.basename(d):<24s} all {len(recs)} records inside burn-in')
        last_acc = -1                       # -1 = no accepted step yet, no profile
        for i, r in enumerate(recs):
            if r.get('accepted'):
                last_acc = r['step']
            if i >= b:
                keep.append(r); cid.append(c); cdir.append(d); pstep.append(last_acc)
    if verbose:
        unit = 'fraction' if burnin < 1 else 'records'
        print(f'  {prefix}: {len(chains)} chains, {len(keep)} records after burn-in {burnin} ({unit})')

    # ── to arrays: params, then every scalar top-level field (dict fields skipped) ─
    post = {}
    for k in PARAMS:
        post[k] = np.array([np.nan if r['params'].get(k) is None else float(r['params'][k])
                            for r in keep])
    fields = []
    for r in keep:
        for k, v in r.items():
            if k != 'params' and not isinstance(v, (dict, list)) and k not in fields:
                fields.append(k)
    for k in fields:
        vals = [r.get(k) for r in keep]
        if any(isinstance(v, str) for v in vals):
            post[k] = np.array(['' if v is None else str(v) for v in vals])
        else:
            post[k] = np.array([np.nan if v is None else float(v) for v in vals])

    post['chain']        = np.array(cid)
    post['chain_dir']    = np.array(cdir)
    post['chain_name']   = np.array([os.path.basename(d) for d in cdir])
    post['profile_step'] = np.array(pstep)

    # ── derived quantities ──────────────────────────────────────────────────────
    hs = np.nan_to_num(post.get('h_solid_km', np.full(len(keep), np.nan)))
    hl = np.nan_to_num(post.get('h_liquid_km', np.full(len(keep), np.nan)))
    st = np.full(len(keep), 'fail', dtype='<U8')
    st[(hs >= H_STATE) & (hl >= H_STATE)] = 'layered'
    st[(hs <  H_STATE) & (hl >= H_STATE)] = 'molten'
    st[(hs >= H_STATE) & (hl <  H_STATE)] = 'solid'
    post['state']     = st
    post['R_scs']     = post['R_cmb'] + hl                    # ScS reflects at the top of the liquid
    post['R_bml_top'] = post['R_cmb'] + post['BML_thickness']
    return post


def profile_table(post):
    # Unique profiles and their dwell-time weights.
    # weight = number of post burn-in records that use this profile.
    # Returns (paths, weights); records before the first accepted step are dropped.
    m = post['profile_step'] >= 0
    key = np.char.add(np.char.add(post['chain_dir'][m], '/profile_s'),
                      np.char.zfill(post['profile_step'][m].astype(str), 5))
    paths, weights = np.unique(key, return_counts=True)
    return np.char.add(paths, '.npz'), weights
