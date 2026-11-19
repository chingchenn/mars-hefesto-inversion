#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 153: a61t sanity check and first look, compared with a60 chains 00-07
#
#   python 153.check_a61t.py
#
# 1. bookkeeping : step numbers continuous, accept rate, longest stuck stretch, out-of-prior states
# 2. same posterior? shallow parameters (T_lit, P_lit, Mg#) must agree with a60, because the
#    likelihood and prior are identical and only propose() changed. A clear shift here = bug.
# 3. mixing      : MSJD (prior-width units) and state transitions per 1000 steps, a61t vs a60
#                  over the same number of steps
# 4. prior walls : fraction of samples within 2% of each prior wall
# 5. results     : per-chain state counts; medians by state (not citable, mixing still checking)
# 6. H1          : Savage-Dickey BF01 for h -> 0 (prior is effectively [1, 400] km)
# 7. ESS and run length, per run group (a60 00-07, a61t s=1.0, s=0.25), same window:
#    tau  = integrated autocorrelation time (Sokal window M >= 5 tau(M)): steps to forget the start
#    ESS  = m n / tau ; Rhat = split R-hat (well above 1: chains disagree, ESS is optimistic)
#    n < 50 tau: tau is only a lower bound
#    a62 steps per chain = BURN + tau * ESS_TARGET / M_PLAN
#    three states (solid, layered, molten): ESS of each 0/1 indicator -> standard error of the
#    fraction, plus the transition matrix (row = from, column = to) per 1000 steps
# 8. new runs vs a60 (a60, a61t, a62-a66, all YM20): did mixing improve?
#    every run cut to the same window W (last W steps after BURN8): tau, ESS, Rhat for the deep
#    parameters and p(solid); state fractions incl. 'none' (h < 1 km, new prior only);
#    8b: posterior of every run cut to the old prior, so it can be compared with a60
import os, json
import numpy as np
from mars_config import MCMC_DIR

NEW    = 'chain_a61t_YM20'
OLD    = 'chain_a60_YM20'
CHAINS = range(8)
SCALE  = [1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25]
BURN   = 200        # a61t starts from a60 end states, so only a short burn-in
KEYS   = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb']
SHALLOW = ['T_lit', 'P_lit', 'Mg#']
PRIOR  = {'T_lit': (1000.0, 2600.0), 'P_lit': (1.4, 9.0), 'Mg#': (0.50, 0.90),
          'T_core': (1550.0, 3600.0), 'Mg#_bulk_bml': (0.20, 0.80),
          'BML_thickness': (0.0, 400.0), 'w_S': (0.05, 0.365), 'R_cmb': (1400.0, 1900.0)}
WIDTH  = np.array([PRIOR[k][1] - PRIOR[k][0] for k in KEYS])
ESS_TARGET = 400        # per parameter, pooled over chains
M_PLAN     = 16         # chains planned for a62


def load(prefix, i):
    recs = []
    for line in open(os.path.join(MCMC_DIR, f'{prefix}_{i:02d}', 'chain.jsonl')):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


def state(r):
    # solid if h_liquid < 1, molten if h_solid < 1, else layered (same as 63v2)
    hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
    if hs is None or hl is None:
        return 'f'
    if hl < 1:
        return 's'
    if hs < 1:
        return 'm'
    return 'l'


new = {i: load(NEW, i) for i in CHAINS}
old = {i: load(OLD, i) for i in CHAINS}
N = min(len(new[i]) for i in CHAINS)          # compare a60 over the same number of steps
print(f'a61t lengths: {[len(new[i]) for i in CHAINS]}   compare window N = {N}, burn-in {BURN}')

# ── 1. bookkeeping ──────────────────────────────────────────────────────────
print('\n1. bookkeeping')
print(f"{'chain':>5} {'scale':>5} {'steps':>6} {'gaps':>4} {'acc%':>6} {'stuck':>6} "
      f"{'outPrior':>8} {'misfit 5/50/95':>22} {'max':>7}")
for i in CHAINS:
    R = new[i]
    steps = np.array([r['step'] for r in R])
    gaps = int(np.sum(np.diff(steps) != 1)) + int(steps[0] != 1)
    X = np.array([[r['params'][k] for k in KEYS] for r in R])
    same = np.all(np.diff(X, axis=0) == 0, axis=1)
    run = best = 0
    for s in same:                     # longest stretch without an accepted move
        run = run + 1 if s else 0
        best = max(best, run)
    out = int(np.sum([[not (PRIOR[k][0] <= r['params'][k] <= PRIOR[k][1]) for k in KEYS] for r in R]))
    m = np.array([r['misfit'] for r in R])
    q = np.percentile(m, [5, 50, 95])
    print(f'{i:5d} {SCALE[i]:5.2f} {len(R):6d} {gaps:4d} {100*np.mean(same == False):6.1f} '
          f'{best:6d} {out:8d} {q[0]:7.2f}/{q[1]:6.2f}/{q[2]:6.2f} {m.max():7.2f}')
print('  gaps should be 0; stuck = longest run of identical states; outPrior must be 0')

# ── pooled samples ──────────────────────────────────────────────────────────
def pool(chains, take_last):
    X, S = [], []
    for R in chains:
        R = R[-take_last:] if take_last else R
        X += [[r['params'][k] for k in KEYS] for r in R]
        S += [state(r) for r in R]
    return np.array(X), np.array(S)

Xn, Sn = pool([new[i][BURN:] for i in CHAINS], None)
Xo, So = pool([old[i] for i in CHAINS], N - BURN)

# ── 2. same posterior? ──────────────────────────────────────────────────────
print('\n2. posterior, a61t vs a60 (chains 00-07, last N-BURN steps of a60)')
print(f"{'param':>14} {'a60 5/50/95':>30} {'a61t 5/50/95':>30} {'shift/sd':>9}")
for j, k in enumerate(KEYS):
    qo = np.percentile(Xo[:, j], [5, 50, 95])
    qn = np.percentile(Xn[:, j], [5, 50, 95])
    sh = (qn[1] - qo[1]) / Xo[:, j].std()
    flag = '  <-- check' if (k in SHALLOW and abs(sh) > 0.3) else ''
    print(f'{k:>14} {qo[0]:9.4g} {qo[1]:9.4g} {qo[2]:9.4g}  {qn[0]:9.4g} {qn[1]:9.4g} {qn[2]:9.4g} '
          f'{sh:+9.2f}{flag}')
print('  shallow parameters should agree (|shift| < ~0.3 sd); deep ones may differ (a60 mixed poorly)')

# ── 3. mixing ───────────────────────────────────────────────────────────────
print('\n3. mixing over the same window (MSJD in prior-width units, x1e-4)')
rows = {'a60': [old[i][-(N - BURN):] for i in CHAINS],
        'a61t s=1.0': [new[i][BURN:N] for i in CHAINS if SCALE[i] == 1.0],
        'a61t s=0.25': [new[i][BURN:N] for i in CHAINS if SCALE[i] == 0.25]}
print(f"{'':>12} " + ''.join(f'{k[:8]:>9}' for k in KEYS) + f"{'trans/1k':>9}")
for name, chains in rows.items():
    msjd, tr, n = [], 0, 0
    for R in chains:
        U = np.array([[r['params'][k] for k in KEYS] for r in R]) / WIDTH
        msjd.append(np.mean(np.diff(U, axis=0)**2, axis=0))
        st = [state(r) for r in R]
        tr += sum(a != b for a, b in zip(st[:-1], st[1:]))
        n += len(R)
    print(f'{name:>12} ' + ''.join(f'{1e4*v:9.2f}' for v in np.mean(msjd, axis=0))
          + f'{1000*tr/n:9.1f}')

# ── 4. prior walls ──────────────────────────────────────────────────────────
print('\n4. fraction of samples within 2% of a prior wall (a60 / a61t)')
for j, k in enumerate(KEYS):
    lo, hi = PRIOR[k]
    d = 0.02 * (hi - lo)
    print(f'{k:>14}  low {np.mean(Xo[:, j] < lo + d):6.3f} / {np.mean(Xn[:, j] < lo + d):6.3f}'
          f'   high {np.mean(Xo[:, j] > hi - d):6.3f} / {np.mean(Xn[:, j] > hi - d):6.3f}')
print('  uniform posterior would give 0.020; much larger = posterior piles up at the wall')

# ── 5. results by state ─────────────────────────────────────────────────────
print('\n5. state counts per chain (s / l / m), after burn-in')
for i in CHAINS:
    st = [state(r) for r in new[i][BURN:]]
    print(f'  chain {i} (scale {SCALE[i]}): {st.count("s"):5d} / {st.count("l"):5d} / {st.count("m"):5d}'
          f'   solid {st.count("s")/len(st):.2f}')
iR, iw, iT, ih = KEYS.index('R_cmb'), KEYS.index('w_S'), KEYS.index('T_core'), KEYS.index('BML_thickness')
hl_new = np.array([(r.get('h_liquid_km') or 0.0) for i in CHAINS for r in new[i][BURN:]])
print(f"\nmedians by state (a61t pooled, NOT citable)\n{'':>8} {'n':>7} {'R_cmb':>7} {'R_ScS':>7} "
      f"{'w_S':>6} {'T_core':>7} {'h':>6}")
for s, name in (('s', 'solid'), ('l', 'layered'), ('m', 'molten')):
    m = Sn == s
    if m.any():
        print(f'{name:>8} {m.sum():7d} {np.median(Xn[m, iR]):7.0f} {np.median(Xn[m, iR] + hl_new[m]):7.0f} '
              f'{np.median(Xn[m, iw]):6.3f} {np.median(Xn[m, iT]):7.0f} {np.median(Xn[m, ih]):6.0f}')

# ── 6. H1: no BML (Savage-Dickey) ───────────────────────────────────────────
print('\n6. H1 BF01(h -> 0), prior [1, 400] km')
h = Xn[:, ih]
for eps in (10, 20, 40):
    prior = (eps - 1.0) / 399.0
    per = [np.mean(np.array([r['params']['BML_thickness'] for r in new[i][BURN:]]) < eps) / prior
           for i in CHAINS]
    print(f'  eps {eps:3d} km  BF01 {np.mean(h < eps)/prior:5.2f}   per chain '
          f'{min(per):5.2f} / {np.median(per):5.2f} / {max(per):5.2f}')

# ── 7. ESS and run length ───────────────────────────────────────────────────
def tau_int(X):
    # X: (m, n). Chain-averaged normalized autocorrelation, Sokal automatic window.
    m, n = X.shape
    f, used = np.zeros(n), 0
    for x in X:
        x = x - x.mean()
        ft = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(ft * np.conj(ft))[:n]
        if ac[0] > 0:
            f += ac / ac[0]
            used += 1
    if used == 0:
        return np.nan
    f /= used
    taus = 2.0 * np.cumsum(f) - 1.0
    ok = np.arange(n) >= 5.0 * taus
    return taus[np.argmax(ok)] if ok.any() else taus[-1]


def split_rhat(X):
    # split every chain in half, then the classic Gelman-Rubin ratio
    m, n = X.shape
    h = n // 2
    Y = np.vstack([X[:, :h], X[:, h:2 * h]])
    W = Y.var(axis=1, ddof=1).mean()
    B = h * Y.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.nan
    return np.sqrt(((h - 1) / h * W + B / h) / W)



groups = {'a60 00-07':   [old[i][-(N - BURN):] for i in CHAINS],
          'a61t s=1.0':  [new[i][BURN:N] for i in CHAINS if SCALE[i] == 1.0],
          'a61t s=0.25': [new[i][BURN:N] for i in CHAINS if SCALE[i] == 0.25]}
n = N - BURN
print(f'\n7. ESS and run length: {n} steps per chain (ESS target {ESS_TARGET}, '
      f'planning for {M_PLAN} chains)\n')
for name, chains in groups.items():
    m = len(chains)
    print(f'── {name}  ({m} chains x {n} steps) ' + '─' * 40)
    print(f"{'param':>14} {'tau':>8} {'ESS':>7} {'Rhat':>6} {'reliable':>9} {'steps/chain for a62':>21}")
    ST = np.array([[state(r) for r in R] for R in chains])
    for k in KEYS + ['solid', 'layered', 'molten']:
        if k in ('solid', 'layered', 'molten'):
            X = (ST == k[0]).astype(float)
        else:
            X = np.array([[r['params'][k] for r in R] for R in chains])
        t = tau_int(X)
        ess = m * n / t
        rh = split_rhat(X)
        rel = 'yes' if n >= 50 * t else 'no (>=)'
        need = BURN + t * ESS_TARGET / M_PLAN
        print(f'{k:>14} {t:8.0f} {ess:7.0f} {rh:6.2f} {rel:>9} {need:21.0f}')
    for st, label in (('s', 'solid'), ('l', 'layered'), ('m', 'molten')):
        Xi = (ST == st).astype(float)
        p = Xi.mean()
        t = tau_int(Xi)
        se = np.sqrt(p * (1 - p) / (m * n / t)) if np.isfinite(t) else np.nan
        per = Xi.mean(axis=1)
        print(f'  p({label:7s}) = {p:.3f} +- {se:.3f}   per chain {per.min():.2f}-{per.max():.2f}')
    # transition matrix (from row to column), all chains, per 1000 steps
    print('  transitions per 1000 steps (row = from, column = to):')
    print(f"  {'':>8}{'solid':>8}{'layered':>8}{'molten':>8}")
    for a, la in (('s', 'solid'), ('l', 'layered'), ('m', 'molten')):
        row = []
        for b in 'slm':
            c = sum(np.sum((S[:-1] == a) & (S[1:] == b)) for S in ST) if a != b else 0
            row.append(1000.0 * c / (m * n))
        print(f'  {la:>8}' + ''.join(f'{v:8.2f}' for v in row))
    print()


# ══ 8. new runs vs a60: did mixing improve? ══════════════════════════════════
# a60 and a62 start from random draws, so their last W steps are used.
# a61t and a63-a66 start from earlier end states, so BURN8 is enough.
# State 'none' (BML_thickness < 1 km) exists only from a62 on (new prior).
import glob
RUNS8  = ['a60', 'a61t', 'a62', 'a63', 'a64', 'a65', 'a66']
LABEL8 = {'a60': 'Fo diag old', 'a61t': 'Fo cov  old', 'a62': 'Hd diag new',
          'a63': 'Fo cov  new', 'a64': 'Hd cov  new', 'a65': 'Fo cov  new nograv',
          'a66': 'Hd cov  new nograv'}
SKIP8  = {'a61t': [0, 1, 2, 3]}          # a61t: keep only the scale 0.25 chains
BURN8  = 200
MIN8   = 1000                            # skip chains shorter than this after BURN8
DEEP8  = ['T_core', 'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb']
OLD_PRIOR8 = {'Mg#_bulk_bml': (0.20, 0.80), 'w_S': (0.05, 0.365), 'BML_thickness': (1.0, 400.0)}


def state8(r):
    # same as state(), plus 'n' = none (no BML)
    if r['params']['BML_thickness'] < 1.0:
        return 'n'
    return state(r)


print('\n8. loading runs')
runs8 = {}
for run in RUNS8:
    chains = []
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, f'chain_{run}_YM20_*'))):
        i = int(d.split('_')[-1])
        if i in SKIP8.get(run, []) or not os.path.exists(os.path.join(d, 'chain.jsonl')):
            continue
        R = load(f'chain_{run}_YM20', i)[BURN8:]
        if len(R) < MIN8:
            print(f'  {run} chain {i:02d} skipped ({len(R)} steps after burn-in)')
            continue
        chains.append(R)
    if chains:
        runs8[run] = chains
        print(f'  {run}: {len(chains)} chains, {min(map(len, chains))}-{max(map(len, chains))} steps after burn-in')
    else:
        print(f'  {run}: no usable chains in {MCMC_DIR}')

W = min(len(R) for chains in runs8.values() for R in chains)
print(f'\n8. mixing over the same window: last W = {W} steps per chain')
print('   tau: steps to forget the start (smaller = better). ESS = m W / tau. Rhat near 1 = chains agree')
print('   * = W < 50 tau, tau is only a lower bound')

tab = {'tau': [], 'ESS': [], 'Rhat': []}
for run, chains in runs8.items():
    m = len(chains)
    ST = np.array([[state8(r) for r in R[-W:]] for R in chains])
    row = {'tau': '', 'ESS': '', 'Rhat': ''}
    for k in DEEP8 + ['solid']:
        if k == 'solid':
            X = (ST == 's').astype(float)
        else:
            X = np.array([[r['params'][k] for r in R[-W:]] for R in chains])
        t = tau_int(X)
        row['tau'] += f'{t:10.0f}' + ('*' if W < 50 * t else ' ')
        row['ESS'] += f'{m * W / t:11.0f}'
        row['Rhat'] += f'{split_rhat(X):11.2f}'
    for what in tab:
        tab[what].append(f'{run:>5} {LABEL8[run]:>18} {m:3d}' + row[what])

hdr = f"{'run':>5} {'setup':>18} {'m':>3}" + ''.join(f'{k[:10]:>11}' for k in DEEP8 + ['p(solid)'])
for what in tab:
    print(f'\n  {what}')
    print(hdr)
    for line in tab[what]:
        print(line)

print('\n  states (pooled over the window); solid per chain = min-max of the per-chain fraction')
print(f"{'run':>5} {'setup':>18} {'none':>7} {'solid':>7} {'layered':>8} {'molten':>7} "
      f"{'solid per chain':>16} {'trans/1k':>9}")
for run, chains in runs8.items():
    m = len(chains)
    ST = np.array([[state8(r) for r in R[-W:]] for R in chains])
    per = (ST == 's').mean(axis=1)
    tr = 1000.0 * np.sum(ST[:, 1:] != ST[:, :-1]) / (m * (W - 1))
    print(f'{run:>5} {LABEL8[run]:>18} {np.mean(ST == "n"):7.3f} {np.mean(ST == "s"):7.3f} '
          f'{np.mean(ST == "l"):8.3f} {np.mean(ST == "m"):7.3f} {per.min():8.2f}-{per.max():.2f} {tr:9.1f}')
print('  compare runs only within this table: tau and Rhat depend on the window length W')

# ── 8b. posterior cut to the old prior ──────────────────────────────────────
# uniform priors: the new-prior posterior cut to the old ranges = the old-prior posterior
print('\n8b. posterior 5/50/95, every run cut to the old prior (Mg#_BML <= 0.8, w_S >= 0.05, h >= 1 km)')
print('    a60 deep mixing is poor, so a deep shift vs a60 is not by itself an error: compare with a61t too')
print('    Hd runs (a62, a64, a66) are expected to sit ~0.04 lower in Mg#_BML (145)')
print('    a65, a66 are NOT reweighted by exp(-grav_sigma) here')
for k in KEYS:
    print(f'\n  {k}')
    for run, chains in runs8.items():
        recs = [r for R in chains for r in R[-W:]]
        keep = [r for r in recs
                if all(OLD_PRIOR8[q][0] <= r['params'][q] <= OLD_PRIOR8[q][1] for q in OLD_PRIOR8)]
        x = np.array([r['params'][k] for r in keep])
        q = np.percentile(x, [5, 50, 95])
        print(f'  {run:>5} {LABEL8[run]:>18}  kept {len(keep) / len(recs):5.2f}   '
              f'{q[0]:9.4g} {q[1]:9.4g} {q[2]:9.4g}')
