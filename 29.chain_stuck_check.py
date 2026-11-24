#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 29: find stuck chains in a run
#
#   python 29.chain_stuck_check.py chain_a74_YM20 [chain_a72_YM20 ...]
#
# A step is "moved" if any parameter differs from the previous record (the chain stores the
# state, so a rejected step repeats the previous parameters).
# Per chain: steps, acceptance over all steps and over the last LAST steps, the longest run of
# identical states and the current (ongoing) run at the end, last misfit / misfit_tt, light element.
# Flag STUCK if the ongoing run >= RUN_FLAG steps or the last-LAST acceptance < ACC_FLAG.
import os, sys, glob, json
import numpy as np
from mars_config import MCMC_DIR

LAST     = 200
RUN_FLAG = 100
ACC_FLAG = 0.05


def load_jsonl(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


for prefix in sys.argv[1:]:
    print(f'\n{prefix}')
    print(f'  {"chain":>6} {"steps":>6} {"acc":>6} {"acc_last":>8} {"maxrun":>6} {"now_run":>7} '
          f'{"misfit":>7} {"tt":>6} {"state":>8} {"w_X":>7}  flag')
    for d in sorted(glob.glob(os.path.join(MCMC_DIR, prefix + '_[0-9][0-9]'))):
        fp = os.path.join(d, 'chain.jsonl')
        if not os.path.exists(fp):
            continue
        R = load_jsonl(fp)
        if len(R) < 2:
            continue
        keys = list(R[0]['params'])
        P = np.array([[r['params'].get(k, np.nan) for k in keys] for r in R], float)
        moved = np.any(P[1:] != P[:-1], axis=1)
        run, maxrun = 0, 0
        for m in moved:
            run = 0 if m else run + 1
            maxrun = max(maxrun, run)
        acc_last = moved[-LAST:].mean()
        last = R[-1]
        hs, hl = last.get('h_solid_km'), last.get('h_liquid_km')
        if last['params'].get('BML_thickness', 0) < 1:
            st = 'none'
        elif hs is None or hl is None:
            st = 'fail'
        else:
            st = 'solid' if hl < 1 else ('molten' if hs < 1 else 'layered')
        wx = last['params'].get('w_C', last['params'].get('w_H', np.nan))
        flag = 'STUCK' if (run >= RUN_FLAG or acc_last < ACC_FLAG) else ''
        print(f'  {d[-2:]:>6} {len(R):6d} {moved.mean():6.2f} {acc_last:8.2f} {maxrun:6d} {run:7d} '
              f'{last.get("misfit", np.nan):7.2f} {last.get("misfit_tt", np.nan):6.2f} {st:>8} {wx:7.4f}  {flag}')
