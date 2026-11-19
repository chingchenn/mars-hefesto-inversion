"""
plot_fig6ab_style.py

畫類似 Khan 2023 Fig. 6a/b 的走時比較圖：
  Fig. 6a：Y = phase 相對 P 的走時，X = T_S - T_P（近端事件）
  Fig. 6b：Y = phase 相對 PP 的走時，X = T_SS - T_PP（遠端事件）

線 = top N 模型的 TauP 預測（半透明疊加）
點 = Khan 2023 的觀測值（帶誤差棒）

CSV 來源：khan_duran_events.csv
  - 近端事件（T0_ref=P）：深度用 Duran_Depth_km，距離用 Khan_Delta_deg
  - 遠端/撞擊事件（T0_ref=PP）：深度用 0 km（impact）或 10 km（marsquake）
"""

import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel

# ============================================================
# config
MCMC_DIR  = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
TAUP_DIR  = '/net/beno3/data1/jcchen2/mars-hefesto-runs/taup_models'
OUT_DIR   = '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc'
CSV_PATH  = '/net/beno3/data1/jcchen2/Mars_Khan_2023/event_time.csv'

PREFIX      = 'chain_a07'
BURNIN_FRAC = 0.1
STEP_MAX    = 150
N_BEST      = 10

DEFAULT_DEPTH = 10.0  # km default if no data available
EVENT_LABEL_Y = 950

PHASES_6A = {
    'S':   ('S',   'P'),
    'SS':  ('SS',  'P'),
    'SSS': ('SSS', 'P'),
    'ScS': ('ScS', 'P'),
    'PP':  ('PP',  'P'),
    'PPP': ('PPP', 'P'),
}

# 
PHASES_6B = {
    'SS':    ('SS', 'PP'),
    'ScS':   ('ScS', 'PP'),
    'SKS':   ('SKS', 'PP'),
    'Pdiff': ('Pdiff', 'PP'),
}

# 
OBS_COL = {
    'S':   ('T_S',   'T_S_std'),
    'SS':  ('T_SS',  'T_SS_std'),
    'SSS': ('T_SSS', 'T_SSS_std'),
    'ScS': ('T_ScS', 'T_ScS_std'),
    'PP':  ('T_PP',  'T_PP_std'),
    'PPP': ('T_PPP', 'T_PPP_std'),
    'SKS': ('T_SKS', 'T_SKS_std'),
}

#
MARKERS = {
    'S': 's', 'SS': 's', 'SSS': 's',
    'ScS': 'v', 'SKS': 'v',
    'PP': 'o', 'PPP': 'o','Pdiff': '^',
}

# color
PHASE_COLORS = {
    'S':   '#B5D4F4',
    'SS':  '#4C9ECD',
    'SSS': '#185FA5',
    'ScS': '#185FA5',
    'SKS': '#0D3B6E',
    'PP':  '#F5C4B3',
    'PPP': '#DD8452',
    'Pdiff': '#C51B1D',
}

ALL_TAUP_PHASES = ['P', 'S', 'PP', 'PPP', 'SS', 'SSS', 'ScS', 'SKS', 'pP', 'sS', 'Pdiff']

# ============================================================
# best model
def load_best_models(mcmc_dir, prefix, burnin_frac, step_max, n_best):
    chain_dirs = sorted([
        d for d in os.listdir(mcmc_dir)
        if d.startswith(prefix + '_')
        and os.path.isdir(os.path.join(mcmc_dir, d))
    ])
    selected = []
    for chain_dir in chain_dirs:
        chain_file = os.path.join(mcmc_dir, chain_dir, 'chain.json')
        if not os.path.exists(chain_file):
            continue
        with open(chain_file) as f:
            chain = json.load(f)
        burnin_n = int(len(chain) * burnin_frac)
        chain = chain[burnin_n:]
        m = re.search(r'_(\d+)$', chain_dir)
        if m is None:
            continue
        cid = int(m.group(1))
        for entry in chain:
            step = entry['step']
            if step >= step_max:
                continue
            model_name = f'mcmc_c{cid:02d}_s{step:05d}'
            npz_path = os.path.join(TAUP_DIR, f'{model_name}.npz')
            if os.path.exists(npz_path):
                selected.append((model_name, entry['misfit'], npz_path))
    selected.sort(key=lambda x: x[1])
    selected = selected[:n_best]
    print(f"Selected {len(selected)} best models:")
    for name, mf, _ in selected:
        print(f"  {name}  misfit={mf:.4f}")
    return selected

# ============================================================
# CSV

def load_events(csv_path):
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col not in ['Event', 'T0_ref', 'Note', 'Duran_Depth_fixed']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def get_depth(row):
        note = str(row.get('Note', ''))
        if 'impact' in note:
            return 0.0
        d = row.get('Duran_Depth_km', np.nan)
        if pd.notna(d):
            return float(d)
        return DEFAULT_DEPTH

    df['depth_km'] = df.apply(get_depth, axis=1)


    df['delta'] = df['Khan_Delta_deg']

    near = df[df['T0_ref'] == 'P'].copy()
    far  = df[df['T0_ref'] == 'PP'].copy()
    print("\nDepths used:")
    #for _, row in df.iterrows():
    #    print(f"  {row['Event']}: delta={row['delta']:.1f}°, depth={row['depth_km']:.1f} km")

    return near, far

# ============================================================
# TauP 
def get_times(taup, delta, depth):
    try:
        arr = taup.get_travel_times(
            source_depth_in_km=depth,
            distance_in_degree=delta,
            phase_list=ALL_TAUP_PHASES,
        )
    except Exception:
        return {}
    times = {}
    if delta > 120:
        print(f'\nDelta = {delta:.1f} deg')

        for a in arr:
            print(a.name, a.time)

    times = {}

    for a in arr:
        if a.name not in times:
            times[a.name] = a.time
    return times

# ============================================================
# main
best_models = load_best_models(MCMC_DIR, PREFIX, BURNIN_FRAC, STEP_MAX, N_BEST)
near_df, far_df = load_events(CSV_PATH)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
for i, (model_name, misfit, npz_path) in enumerate(best_models):
    is_best = (i == 0)
    alpha = 0.7 if is_best else 0.2
    lw    = 1.8 if is_best else 0.8

    try:
        taup = TauPyModel(model=npz_path)
    except Exception as e:
        print(f"Cannot load {model_name}: {e}")
        continue

    # ── Fig. 6a
    for ph_name, (ph_a, ph_b) in PHASES_6A.items():
        color = PHASE_COLORS.get(ph_name, 'gray')
        pts = []
        for _, row in near_df.iterrows():
            delta = row['delta']
            depth = row['depth_km']
            if pd.isna(delta):
                continue
            times = get_times(taup, delta, depth)
            if ph_a not in times or 'P' not in times:
                continue
            if ph_b not in times:
                continue
            x_pred = times['S'] - times['P'] if 'S' in times else None
            if x_pred is None:
                continue
            y_pred = times[ph_a] - times[ph_b]
            pts.append((x_pred, y_pred))

        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        xs, ys = zip(*pts)
        label = ph_name if is_best else '_nolegend_'
        ax_a.plot(xs, ys, color=color, alpha=alpha, lw=lw, label=label)

    # ── Fig. 6b
    for ph_name, (ph_a, ph_b) in PHASES_6B.items():
        color = PHASE_COLORS.get(ph_name, 'gray')
        pts = []
        for _, row in far_df.iterrows():
            delta = row['delta']
            depth = row['depth_km']
            if pd.isna(delta):
                continue
            times = get_times(taup, delta, depth)
            if ph_a not in times or ph_b not in times:
                continue
            if 'SS' not in times or 'PP' not in times:
                continue
            x_pred = times['SS'] - times['PP']
            y_pred = times[ph_a] - times[ph_b]
            pts.append((x_pred, y_pred))

        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        xs, ys = zip(*pts)
        label = ph_name if is_best else '_nolegend_'
        ax_b.plot(xs, ys, color=color, alpha=alpha, lw=lw, label=label)



# 
for _, row in near_df.iterrows():
    x_obs = row.get('T_S', np.nan)  # T_S = S-P（T_P=0 by definition）
    if pd.isna(x_obs):
        continue
    x_err = row.get('T_S_std', np.nan)

    for ph_name, (t_col, t_std_col) in OBS_COL.items():
        if ph_name not in PHASES_6A:
            continue
        y_obs = row.get(t_col, np.nan)
        y_err = row.get(t_std_col, np.nan)
        if pd.isna(y_obs):
            continue
        color  = PHASE_COLORS.get(ph_name, 'gray')
        marker = MARKERS.get(ph_name, 'o')
        ax_a.errorbar(
            x_obs, y_obs,
            xerr=x_err if pd.notna(x_err) else None,
            yerr=y_err if pd.notna(y_err) else None,
            fmt=marker, color='k', mfc=color, mec='k',
            ms=9, capsize=3, lw=1.2, zorder=5,
        )

    ax_a.axvline(x_obs, color='gray', lw=0.6, ls=':', alpha=0.5, zorder=1)


for _, row in far_df.iterrows():
    x_obs = row.get('T_SS', np.nan)  # T_SS = SS-PP（T_PP=0 by definition）
    if pd.isna(x_obs):
        continue

    for ph_name, (t_col, t_std_col) in OBS_COL.items():
        if ph_name not in PHASES_6B:
            continue
        y_obs = row.get(t_col, np.nan)
        y_err = row.get(t_std_col, np.nan)
        if pd.isna(y_obs):
            continue
        color  = PHASE_COLORS.get(ph_name, 'gray')
        marker = MARKERS.get(ph_name, 'o')
        ax_b.errorbar(
            x_obs, y_obs,
            yerr=y_err if pd.notna(y_err) else None,
            fmt=marker, color='k', mfc=color, mec='k',
            ms=9, capsize=3, lw=1.2, zorder=5,
        )
    if pd.notna(row.get('T_Pdiff_LSL', np.nan)):

        ax_b.errorbar(
            x_obs,
            row['T_Pdiff_LSL'],
            yerr=row.get('T_Pdiff_LSL_std', None),
            fmt='^',
            color='k',
            mfc='#C51B1D',
            mec='k',
            ms=10,
            capsize=3,
            lw=1.2,
            zorder=6,
        )    
    if pd.notna(row.get('T_Pdiff_CMB', np.nan)):

        ax_b.errorbar(
            x_obs,
            row['T_Pdiff_CMB'],
            yerr=row.get('T_Pdiff_CMB_std', None),
            fmt='^',
            color='k',
            mfc='orange',
            mec='k',
            ms=10,
            capsize=3,
            lw=1.2,
            zorder=6,
        )
    

    # 
    ax_b.axvline(x_obs, color='gray', lw=0.6, ls=':', alpha=0.5, zorder=1)
    ax_b.scatter(x_obs,0,marker='o',facecolor='white',edgecolor='k',s=40,zorder=4,)
    ylim = ax_b.get_ylim()
    ax_b.text(x_obs,EVENT_LABEL_Y,row['Event'],ha='center',va='bottom',fontsize=9,color='gray')

# ── 
ax_a.set_xlabel('$T_S - T_P$ (s)', fontsize=12)
ax_a.set_ylabel('Time after P (s)', fontsize=12)
ax_a.set_title('(a)  Near-side events', fontsize=12)
ax_a.grid(True, ls='--', alpha=0.2)
ax_a.legend(fontsize=9, loc='upper left', framealpha=0.8)

ax_b.set_xlabel('$T_{SS} - T_{PP}$ (s)', fontsize=12)
ax_b.set_ylabel('Time after PP (s)', fontsize=12)
ax_b.set_title('(b)  Far-side events', fontsize=12)
ax_b.grid(True, ls='--', alpha=0.2)
ax_b.legend(fontsize=9, loc='upper left', framealpha=0.8)

plt.suptitle(
    f'Travel time comparison  —  top {N_BEST} models  ({PREFIX})\n'
    'Lines = TauP predictions,  Points = Khan 2023 / Durán 2022 observations',
    fontsize=13,
)
plt.tight_layout()

outpath = os.path.join(OUT_DIR, f'fig6ab_style_{PREFIX}.png')
plt.savefig(outpath, dpi=180, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")