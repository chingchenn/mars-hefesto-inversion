# 34.khan_median_traveltime.py
# Khan vs Samuel travel times for each event and phase, and the mean |Samuel - Khan| per phase.
import numpy as np
import pandas as pd

CSV_FILE = '/Users/chingchen/Desktop/Lunar/data/combined_travel_times.csv'

df = pd.read_csv(CSV_FILE)

print('=' * 70)
print('Khan vs Samuel differential travel-time comparison')
print('=' * 70)
for event in sorted(df['Event'].unique()):
    print(f'\nEvent: {event}')
    for _, row in df[df['Event'] == event].iterrows():
        print(f"{row['Phase']:8s} | Khan = {row['Khan']:8.2f} s | Samuel = {row['Samuel']:8.2f} s | "
              f"Samuel-Khan = {row['Samuel_minus_Khan_s']:+7.2f} s")

print('\nMean absolute difference by phase:')
for phase, val in df.groupby('Phase')['Samuel_minus_Khan_s'].apply(lambda x: np.mean(np.abs(x))).items():
    print(f'{phase:8s} : {val:.2f} s')
