#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 153 (stop check): can a63-a66 be stopped? Same chain reading and state() as 153.check_a61t.py
#
#   python 153.stop_check_a63_a66.py
#
# Goal: report the BML state fractions (solid / layered / molten) with a known Monte Carlo error.
# A run can stop when ALL of these hold over the window after BURN:
#   1. continuous parameters: rank-normalized split R-hat (bulk and tail) < RHAT_MAX,
#      bulk ESS and tail ESS >= ESS_TARGET
#   2. state indicators (0/1 per step): split R-hat < RHAT_MAX, ESS >= ESS_TARGET
#      MCSE of a fraction = sqrt(p (1 - p) / ESS)   e.g. p = 0.2: ESS 100 -> 0.04, ESS 400 -> 0.02
#   3. stability: first half vs second half of the window, |p1 - p2| < 2 sqrt(se1^2 + se2^2)
# 'none' (h < 1 km) is reported but not used for the verdict (too rare to estimate).
# Rank normalization (Vehtari et al. 2021): ranks -> normal scores, then split R-hat.
#   tail R-hat = same on |x - median| (folded). Max of the two is reported.
# ESS = multi-chain ESS of Vehtari et al. (2021) on split chains: the autocorrelation is taken
#   relative to var_plus (within + between chain variance), so chains that disagree LOWER the ESS.
#   (153.check_a61t.py uses m n / tau, which ignores between-chain disagreement and is optimistic
#   when R-hat > 1; that is why both numbers can look fine there while R-hat fails.)
# bulk ESS on rank-normalized draws; tail ESS = min ESS of the indicators x <= q05 and x >= q95.
# Steps still needed = n * ESS_TARGET / ESS - n (worst quantity), assumes ESS grows linearly.
#   R-hat is not extrapolated: if only R-hat fails, rerun this check after more steps.
import os, json
from statistics import NormalDist
import numpy as np
from mars_config import MCMC_DIR

RUNS       = ['a63', 'a64', 'a65', 'a66']
LABEL      = {'a63': 'Fo grav (SI)', 'a64': 'Hd grav (MAIN)', 'a65': 'Fo nograv (SI)', 'a66': 'Hd nograv (SI)'}
ESS_TARGET = {'a63': 150, 'a64': 400, 'a65': 150, 'a66': 150}   # main result stricter
RHAT_MAX   = 1.05
BURN       = 1000       # misfit is stable after ~1000 steps
SUFFIX     = 'YM20'
KEYS  = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness', 'w_S', 'R_cmb']
STATES = [('n', 'none'), ('s', 'solid'), ('l', 'layered'), ('m', 'molten')]
NORM = NormalDist()


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
    # none if h < 1 km; solid if h_liquid < 1, molten if h_solid < 1, else layered (same as 153 section 8)
    if r['params']['BML_thickness'] < 1.0:
        return 'n'
    hs, hl = r.get('h_solid_km'), r.get('h_liquid_km')
    if hs is None or hl is None:
        return 'f'
    if hl < 1:
        return 's'
    if hs < 1:
        return 'm'
    return 'l'


def split_rhat(X):
    # split every chain in half, then the classic Gelman-Rubin ratio (same as 153)
    m, n = X.shape
    h = n // 2
    Y = np.vstack([X[:, :h], X[:, h:2 * h]])
    W = Y.var(axis=1, ddof=1).mean()
    B = h * Y.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.nan
    return np.sqrt(((h - 1) / h * W + B / h) / W)


def ess_v(X):
    # X: (m, n). Split chains, Geyer initial monotone sequence (Vehtari et al. 2021, Stan)
    m, n = X.shape
    h = n // 2
    Y = np.vstack([X[:, :h], X[:, h:2 * h]])
    M = Y.shape[0]
    acov = np.zeros((M, h))
    for j, y in enumerate(Y):
        y = y - y.mean()
        ft = np.fft.rfft(y, 2 * h)
        acov[j] = np.fft.irfft(ft * np.conj(ft))[:h] / h       # biased autocovariance
    chain_var = acov[:, 0] * h / (h - 1)
    W = chain_var.mean()
    var_plus = W * (h - 1) / h + Y.mean(axis=1).var(ddof=1)
    if var_plus <= 0:
        return np.nan
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # sum of pairs, stop at the first negative pair, force monotone decrease
    P = rho[0:h - 1:2] + rho[1:h:2]
    neg = np.where(P < 0)[0]
    P = P[:neg[0]] if neg.size else P
    P = np.minimum.accumulate(P)
    tau = -1.0 + 2.0 * P.sum()
    return M * h / max(tau, 1.0 / np.log10(M * h))


def rank_normal(X):
    # ranks over all draws (ties averaged) -> normal scores (r - 3/8) / (S + 1/4)
    flat = X.ravel()
    order = np.argsort(flat, kind='mergesort')
    ranks = np.empty(flat.size)
    ranks[order] = np.arange(1, flat.size + 1)
    u, inv = np.unique(flat, return_inverse=True)
    ranks = np.bincount(inv, weights=ranks) / np.bincount(inv)
    ranks = ranks[inv]
    z = np.array([NORM.inv_cdf(v) for v in (ranks - 0.375) / (flat.size + 0.25)])
    return z.reshape(X.shape)


summary = []
for run in RUNS:
    chains = []
    i = 0
    while os.path.exists(os.path.join(MCMC_DIR, f'chain_{run}_{SUFFIX}_{i:02d}', 'chain.jsonl')):
        chains.append(load(f'chain_{run}_{SUFFIX}', i)[BURN:])
        i += 1
    m = len(chains)
    n = min(len(R) for R in chains)              # common window: first n steps after BURN
    chains = [R[:n] for R in chains]
    target = ESS_TARGET[run]
    print(f'\n{"=" * 86}\n{run}  {LABEL[run]}   {m} chains x {n} steps after burn-in {BURN}'
          f'   ESS target {target}, R-hat < {RHAT_MAX}\n{"=" * 86}')

    fails, need = [], 0.0

    # ── 1. continuous parameters ─────────────────────────────────────────────
    print(f"{'param':>14} {'Rhat':>6} {'Rhat_rank':>10} {'ESS_bulk':>9} {'ESS_tail':>9}  ok")
    for k in KEYS:
        X = np.array([[r['params'][k] for r in R] for R in chains])
        rh = split_rhat(X)
        rh_bulk = split_rhat(rank_normal(X))
        rh_tail = split_rhat(rank_normal(np.abs(X - np.median(X))))
        rh_rank = max(rh_bulk, rh_tail)
        ess_b = ess_v(rank_normal(X))
        q05, q95 = np.percentile(X, [5, 95])
        ess_t = min(ess_v((X <= q05).astype(float)), ess_v((X >= q95).astype(float)))
        ok = rh_rank < RHAT_MAX and ess_b >= target and ess_t >= target
        if not ok:
            fails.append(k)
        need = max(need, n * target / min(ess_b, ess_t) - n)
        print(f'{k:>14} {rh:6.3f} {rh_rank:10.3f} {ess_b:9.0f} {ess_t:9.0f}  {"yes" if ok else "NO"}')

    # ── 2. state fractions ───────────────────────────────────────────────────
    ST = np.array([[state(r) for r in R] for R in chains])
    print(f"\n{'state':>8} {'p':>6} {'MCSE':>6} {'Rhat':>6} {'ESS':>6} {'per chain':>11}"
          f" {'1st half':>9} {'2nd half':>9} {'diff/se':>8}  ok")
    h = n // 2
    res = {}
    for s, name in STATES:
        Xi = (ST == s).astype(float)
        p = Xi.mean()
        ess = ess_v(Xi)
        mcse = np.sqrt(p * (1 - p) / ess)
        rh = split_rhat(Xi)
        per = Xi.mean(axis=1)
        # stability: first half vs second half of the window, each with its own ESS
        A, B = Xi[:, :h], Xi[:, h:2 * h]
        pa, pb = A.mean(), B.mean()
        sea = np.sqrt(pa * (1 - pa) / ess_v(A))
        seb = np.sqrt(pb * (1 - pb) / ess_v(B))
        z = abs(pa - pb) / np.sqrt(sea**2 + seb**2)
        ok = rh < RHAT_MAX and ess >= target and z < 2.0
        if s != 'n':
            if not ok:
                fails.append(name)
            need = max(need, n * target / ess - n)
        res[name] = (p, mcse)
        print(f'{name:>8} {p:6.3f} {mcse:6.3f} {rh:6.3f} {ess:6.0f} {per.min():5.2f}-{per.max():.2f}'
              f' {pa:9.3f} {pb:9.3f} {z:8.2f}  {"yes" if ok else "NO"}'
              + ('  (not used for verdict)' if s == 'n' else ''))

    verdict = 'STOP' if not fails else 'CONTINUE'
    extra = 0 if not fails else int(np.ceil(max(need, 0)))
    print(f'\nverdict: {verdict}' + ('' if not fails else
          f'   failing: {", ".join(fails)}   ~{extra} more steps per chain for ESS'
          + ('  (ESS already enough: R-hat or stability is the limit)' if extra == 0 else '')))
    summary.append((run, m, n, verdict, extra, res))

# ── summary ─────────────────────────────────────────────────────────────────
print(f'\n{"=" * 86}\nsummary (fractions +- MCSE)')
print(f"{'run':>5} {'setup':>16} {'m':>3} {'n':>6} {'solid':>14} {'layered':>14} {'molten':>14} "
      f"{'verdict':>9} {'more':>6}")
for run, m, n, verdict, extra, res in summary:
    cells = ''.join(f'{res[k][0]:7.3f}+-{res[k][1]:.3f}' for k in ('solid', 'layered', 'molten'))
    print(f'{run:>5} {LABEL[run]:>16} {m:3d} {n:6d} {cells} {verdict:>9} {extra:6d}')
