#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:53:18 2026

@author: chingchen
"""

"""
compare_khan_samuel.py

比較：
同一 event
同一 phase

Khan vs Samuel travel times 差多少
"""

import pandas as pd
import numpy as np

# ============================================================
# read comparison table
# ============================================================

df = pd.read_csv('/Users/chingchen/Desktop/Lunar/data/combined_travel_times.csv')

print('\n')
print('='*70)
print('Khan vs Samuel differential travel-time comparison')
print('='*70)

# ============================================================
# loop over events
# ============================================================

events = sorted(df['Event'].unique())

for event in events:

    sub = df[df['Event'] == event]

    print('\n')
    print('='*70)
    print(f'Event: {event}')
    print('='*70)

    for _, row in sub.iterrows():

        phase   = row['Phase']
        khan    = row['Khan']
        samuel  = row['Samuel']
        diff    = row['Samuel_minus_Khan_s']

        print(
            f'{phase:8s} | '
            f'Khan = {khan:8.2f} s | '
            f'Samuel = {samuel:8.2f} s | '
            f'Samuel-Khan = {diff:+7.2f} s'
        )

# ============================================================
# summary statistics
# ============================================================

print('\n')
print('='*70)
print('Overall statistics')
print('='*70)

print('\nMean absolute difference by phase:\n')

phase_stats = (
    df.groupby('Phase')['Samuel_minus_Khan_s']
      .apply(lambda x: np.mean(np.abs(x)))
)

for phase, val in phase_stats.items():

    print(f'{phase:8s} : {val:.2f} s')

print('\nDone.')