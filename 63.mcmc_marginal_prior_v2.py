#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 63 v3: 收斂/後驗診斷,支援 a31(chain.json)到 a51(chain.jsonl)全部 run
#
# 改動 vs v2:
#   - w_S prior 上界依 PREFIX 自動切換:a48/a49 = 0.30,a50/a51 = 0.40。
#     shrink 與 edge check 兩張表完全靠這個數字,寫死會給錯答案。
#   - 三態分類統一走 classify_state();v2 裡 chain_series() 用 hl<1 -> solid,
#     與後驗組裝的 (hs>=1 & hl>=1) -> layered 不同,R-hat 與 transition 表
#     等於在算兩個不同的量。
#   - rhat_table 的 return 從 for 迴圈裡拉出來(v2 在第一個 key 就返回,
#     mats 只有 layered_frac,後面 per-chain means 直接 KeyError);
#     三條 return 路徑統一成 (mats, n2, idx)。
#   - alive 不再被覆蓋三次:長度篩選一次定案,rhat_table 只回報不改寫。
#   - plot_corner / EXTRA_KEYS:整欄 NaN 的鍵(舊 run 沒有 tt_per_datum、
#     fesi_* 等)直接跳過,不再畫出空圖或噴 all-NaN warning。
#   - tick_params(labelsize=14) 全域一致。

import json, glob, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mars_config import MCMC_DIR

# ══════════════════════════════════════════════════════════════════════════
# 設定
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR = MCMC_DIR
OUT_DIR   = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'
PREFIX    = 'chain_a61t'

if PREFIX in ('chain_a36', 'chain_a36_'):
    raise SystemExit("chain_a36 會同抓 A/B 兩套核 EoS,不可合併;指定 chain_a36A 或 chain_a36B")

EXCLUDE = ['chain_a37A_01', 'chain_a38A_01', 'chain_a47B_14',
           'chain_a47B_04', 'chain_a47B_09', 'chain_a47B_10', 'chain_a47B_13',
           'chain_a52B_04']

BURNIN_MODE = 'frac'   # 'switch': per-chain geo<GEO_SWITCH 首達步+PAD(a47 用)
                       # 'frac'  : 全域比例砍頭(a46 以前的 run 用)
BURNIN_FRAC = 0.3
GEO_SWITCH  = 5.0
BURNIN_PAD  = 200
DROP_LAST   = 1
MIN_STEPS   = 5
SAVE        = False
fontsize    = 14
TICKSIZE    = 14

MARS_RADIUS        = 3389.5
FALLBACK_CMB_DEPTH = 1743.3   # a36 之前 R_cmb 固定時的 CMB 深度
ESS_TARGET         = 400
ACC_WINDOW         = 200      # acceptance sliding window
H_STATE_TOL        = 1.0      # km,固/液次層厚度判準(與 70/70b 的 H_MIN_LAYER 對齊)
RHAT_MIN_N         = 50       # 進 R-hat 表所需的最短 post-burn-in 長度

BI_RIC_KM, BI_RIC_ERR, BI_RIC_SIG = 612.0, 50.0, 112.0
RIC_MAX_KM      = 750.0
WS_GEOCHEM      = 0.17
WS_STOICH       = 0.365       # Fe-S 化學計量上限
IRVING_VP_CMB   = (4.9, 5.0)
RHO_MEAN_MSL    = (6.5, 6.65)
T_EXCLUDE_PROBES = (2500.0, 3000.0, 3437.0)

CUR_STEP = {'T_lit': 50.0, 'P_lit': 0.7, 'Mg#': 0.025, 'T_core': 100.0,
            'Mg#_bulk_bml': 0.06, 'BML_thickness': 25.0, 'w_S': 0.02, 'R_cmb': 25.0}

# ── w_S prior 上界:runtime 值隨 run 改過,必須依 PREFIX 取 ────────────────
#   a35 以前            : 0.30
#   a36-a47             : 0.40
#   a48B / a49B         : 0.30   ← 目前主力兩條
#   a50B / a51B         : 0.40
# 表以外的 run 落到 WS_HI_DEFAULT,並且印警告要求人工確認。
WS_HI_BY_RUN = {
    'a48': 0.30, 'a49': 0.30,
    'a50': 0.40, 'a51': 0.40,
    'a36': 0.40, 'a37': 0.40, 'a38': 0.40, 'a39': 0.40,
    'a40': 0.40, 'a41': 0.40, 'a42': 0.40, 'a43': 0.40,
    'a44': 0.40, 'a45': 0.40, 'a46': 0.40, 'a47': 0.40,
    'a31': 0.30, 'a32': 0.30, 'a33': 0.30, 'a34': 0.30, 'a35': 0.30,
}
WS_HI_DEFAULT = 0.40

_run_tag = PREFIX.replace('chain_', '')[:3]
WS_HI = WS_HI_BY_RUN.get(_run_tag, WS_HI_DEFAULT)
if _run_tag not in WS_HI_BY_RUN:
    print(f"WARNING: {PREFIX} 不在 WS_HI_BY_RUN 表內,w_S 上界用預設 {WS_HI_DEFAULT}。"
          f"shrink 與 edge check 會受影響 — 請確認該 run 的 runtime PRIOR。")
print(f"run tag = {_run_tag}   w_S prior = (0.05, {WS_HI})")

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (   0.0,  400.0),
    'w_S':           (  0.05,   WS_HI),
    'R_cmb':         (1400.0, 1900.0),
}
UNITS  = {'T_lit': 'K', 'P_lit': 'GPa', 'Mg#': '', 'T_core': 'K',
          'Mg#_bulk_bml': '', 'BML_thickness': 'km', 'w_S': '', 'R_cmb': 'km'}
PARAMS = list(PRIOR)
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)', 'w$_S$ (core S wt)', 'R$_{CMB}$ (km)']
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}

EXTRA_KEYS = ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
              'upper_contrast', 'lower_contrast', 'Ra', 'h_solid_km', 'h_liquid_km',
              'Mg_solid', 'Mg_liquid', 'T_interface', 'T_mantle_bottom', 'moi_pred',
              'tt_n_ph', 'tt_n_miss', 'tt_n_capped',
              'R_ic_km', 'ric_sigma', 'rho_core_mean', 'Vp_core_cmb',
              'P_center_core', 'T_center_core',
              'P_cmb', 'P_bml_top', 'z_lit_km', 'gap_min', 'bml_n_pass',
              'sc_n_iter', 'sc_dP', 'sc_dz_lit', 'fesi_sigma', 'fesi_pred',
              'tt_per_datum', 'geo_sum', 'misfit_reduced', 'n_data']

TOGGLE = {
    'marginals':            1,
    'corner_misfit':        1,
    'corner_state':         1,
    'corner_misfit_tt':     1,
    'corner_moi_sigma':     1,
    'corner_tt_per':        1,
    'correlations':         1,
    'core_diag':            1,
    'struct_diag':          1,
    'prior_post_shrinkage': 1,
    'convergence_diag':     1,
    'acceptance_window':    1,
}

plt.rcParams['font.family'] = 'DejaVu Sans'
os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''

# ══════════════════════════════════════════════════════════════════════════
# 三態分類:全腳本唯一定義
#   solid   : h_liquid < tol  且 h_solid >= tol
#   molten  : h_solid  < tol  且 h_liquid >= tol
#   layered : 兩者皆 >= tol
#   fail    : 兩者皆 < tol,或有 NaN(forward 失敗的佔位)
# ══════════════════════════════════════════════════════════════════════════
def classify_state(h_sol, h_liq, tol=H_STATE_TOL):
    h_sol = np.atleast_1d(np.asarray(h_sol, float))
    h_liq = np.atleast_1d(np.asarray(h_liq, float))
    st  = np.full(h_sol.shape, 'fail', dtype='<U8')
    ok  = np.isfinite(h_sol) & np.isfinite(h_liq)
    s_y = ok & (h_sol >= tol)
    l_y = ok & (h_liq >= tol)
    st[ s_y &  l_y] = 'layered'
    st[~s_y &  l_y] = 'molten'
    st[ s_y & ~l_y] = 'solid'
    return st

# ══════════════════════════════════════════════════════════════════════════
# 讀檔:chain.jsonl(a34+)優先,退回 chain.json(a31-a33, salvage)
# ══════════════════════════════════════════════════════════════════════════
def load_chain_json(path):
    if not os.path.exists(path):
        return None, 'missing'
    text = open(path).read()
    if not text.strip():
        return None, 'empty'
    try:
        return json.loads(text), 'ok'
    except json.JSONDecodeError:
        pass
    dec, i, recs = json.JSONDecoder(), text.find('[') + 1, []
    while True:
        while i < len(text) and text[i] in ' \t\r\n,':
            i += 1
        if i >= len(text) or text[i] == ']':
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        recs.append(obj)
    return (recs, 'salvaged') if recs else (None, 'unparseable')

def load_chain_jsonl(path):
    if not os.path.exists(path):
        return None, 'missing'
    recs, n_bad = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            n_bad += 1
            break
    if not recs:
        return None, 'empty'
    return recs, ('ok' if n_bad == 0 else 'truncated_last_line')

def load_chain_dir(dd):
    p = os.path.join(dd, 'chain.jsonl')
    if os.path.exists(p):
        return load_chain_jsonl(p)
    return load_chain_json(os.path.join(dd, 'chain.json'))

def burnin_index(ch):
    # 'switch': geo<GEO_SWITCH 首達步 + PAD;缺 mass/moi 欄位的舊 run 自動退回 frac
    if BURNIN_MODE == 'frac':
        return int(len(ch) * BURNIN_FRAC)
    has_geo = any(s.get('mass_sigma') is not None for s in ch[:50])
    if not has_geo:
        return int(len(ch) * BURNIN_FRAC)
    for i, s in enumerate(ch):
        m, q = s.get('mass_sigma'), s.get('moi_sigma')
        if m is not None and q is not None and m < GEO_SWITCH and q < GEO_SWITCH:
            return min(i + BURNIN_PAD, len(ch))
    return len(ch)   # 從未達標 -> 全丟(等效剔除)

def _grid(n, ncols=4, figsize_per=(4.0, 4.0)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]

def has_data(key):
    return key in post and np.isfinite(post[key]).any()

# ══════════════════════════════════════════════════════════════════════════
# load
# ══════════════════════════════════════════════════════════════════════════
chain_dirs = sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*')))
all_chains, chain_names, load_report = [], [], []
for dd in chain_dirs:
    name = os.path.basename(dd)
    if name in EXCLUDE:
        load_report.append((name, 'excluded', 0, 'DROPPED'))
        continue
    ch, status = load_chain_dir(dd)
    n_raw = 0 if ch is None else len(ch)
    if ch is not None and DROP_LAST > 0:
        ch = ch[:-DROP_LAST] if len(ch) > DROP_LAST else []
    if ch is None or len(ch) < MIN_STEPS:
        load_report.append((name, status, n_raw, 'DROPPED'))
        continue
    all_chains.append(ch)
    chain_names.append(name)
    load_report.append((name, status, n_raw, f'kept {len(ch)}'))

print(f"{'chain':<20s} {'status':<18s} {'n_raw':>7s}  action")
for name, status, n_raw, action in load_report:
    print(f"{name:<20s} {status:<18s} {n_raw:>7d}  {action}")
print(f"\nusable chains: {len(all_chains)}")
if not all_chains:
    raise SystemExit("no usable chains")

BURN = [burnin_index(ch) for ch in all_chains]
print(f"\nburn-in mode = {BURNIN_MODE}")
for nm, ch, b in zip(chain_names, all_chains, BURN):
    print(f"  {nm:<20s} burn-in {b:5d} / {len(ch):5d}  kept {len(ch)-b}")

# ══════════════════════════════════════════════════════════════════════════
# 後驗組裝(逐鏈套各自 burn-in)
# ══════════════════════════════════════════════════════════════════════════
samples, chain_id = [], []
for c, (ch, b) in enumerate(zip(all_chains, BURN)):
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"\npost burn-in samples: {len(samples)}")

def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])

post = {p: np.array([float(s['params'].get(p, np.nan)) for s in samples])
        for p in PARAMS}
for k in EXTRA_KEYS:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

_missing = [p for p in PARAMS if np.all(~np.isfinite(post[p]))]
if _missing:
    print(f"NOTE: params absent in these chains: {_missing}")

_absent_extra = [k for k in EXTRA_KEYS if not np.isfinite(post[k]).any()]
if _absent_extra:
    print(f"NOTE: diagnostic keys absent (all NaN): {_absent_extra}")

post['state'] = classify_state(post['h_solid_km'], post['h_liquid_km'])
state = post['state']
print("state counts:", {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})
_n_fail = int((state == 'fail').sum())
if _n_fail:
    print(f"  WARNING: {_n_fail} 個樣本兩層厚度皆 < {H_STATE_TOL} km 或含 NaN,"
          f"歸為 fail,不計入任何三態統計")

PARAMS_LIVE = [p for p in PARAMS if p not in _missing]
LABELS_LIVE = [LABELS[PARAMS.index(p)] for p in PARAMS_LIVE]

if 'R_cmb' in PARAMS_LIVE:
    post['cmb_depth_km'] = MARS_RADIUS - post['R_cmb']
else:
    post['cmb_depth_km'] = np.full(len(samples), FALLBACK_CMB_DEPTH)
post['bml_top_depth_km'] = post['cmb_depth_km'] - post['BML_thickness']

# ══════════════════════════════════════════════════════════════════════════
# 圖4  1D marginals
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['marginals']:
    fig, axes = _grid(len(PARAMS_LIVE), ncols=4, figsize_per=(4.3, 3.8))
    fig.suptitle(f'{PREFIX} Posterior marginals', fontsize=fontsize)
    for ax, param, label in zip(axes, PARAMS_LIVE, LABELS_LIVE):
        ax.hist(post[param], bins=40, color=COLORS[2], edgecolor='none')
        ax.axvline(np.median(post[param]), color='red', lw=1.5, label='median')
        ax.axvline(np.percentile(post[param], 16), color=COLORS[4], lw=2)
        ax.axvline(np.percentile(post[param], 84), color=COLORS[4], lw=2, label='16/84%')
        if param == 'w_S':
            ax.axvline(WS_GEOCHEM, color='k', ls=':', lw=2, label='geochem 17 wt%')
            if PRIOR['w_S'][1] > WS_STOICH:
                ax.axvline(WS_STOICH, color='k', ls='--', lw=1.5, label='stoich 36.5 wt%')
        ax.set_xlabel(label, fontsize=fontsize); ax.set_ylabel('count', fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE); ax.legend(fontsize=9)
    plt.tight_layout()
    if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + '04_marginals.png'), dpi=150)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════
# corner 系列
# ══════════════════════════════════════════════════════════════════════════
def plot_corner(color_key, cmap, cbar_label, suffix, discrete=False):
    if not discrete and not has_data(color_key):
        print(f'corner[{color_key}]: 該 run 無此欄位 -> 跳過')
        return
    n = len(PARAMS_LIVE)
    fig, axes = plt.subplots(n, n, figsize=(2.1*n, 2.0*n))
    fig.suptitle(f'{PREFIX} Corner (colour = {cbar_label})', fontsize=fontsize + 4)
    if discrete:
        cvec = np.array([STATE_COLOR.get(s, '#BBBBBB') for s in post[color_key]])
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                if discrete:
                    for st, c in STATE_COLOR.items():
                        msk = (post[color_key] == st)
                        if msk.sum():
                            ax.hist(post[PARAMS_LIVE[i]][msk], bins=30, color=c,
                                    histtype='step', lw=1.5, density=True)
                else:
                    ax.hist(post[PARAMS_LIVE[i]], bins=30, color=COLORS[2],
                            edgecolor='none')
                    ax.axvline(np.median(post[PARAMS_LIVE[i]]), color='red',
                               ls='--', lw=2)
                ax.set_yticklabels([])
            elif i > j:
                if discrete:
                    ax.scatter(post[PARAMS_LIVE[j]], post[PARAMS_LIVE[i]], c=cvec,
                               s=3, alpha=0.4, linewidths=0)
                else:
                    ax.scatter(post[PARAMS_LIVE[j]], post[PARAMS_LIVE[i]],
                               c=post[color_key], cmap=cmap, s=2, linewidths=0)
            else:
                ax.set_visible(False)
            if i == n - 1: ax.set_xlabel(LABELS_LIVE[j], fontsize=fontsize - 2)
            else:          ax.set_xticklabels([])
            if j == 0 and i != 0: ax.set_ylabel(LABELS_LIVE[i], fontsize=fontsize - 2)
            else:                 ax.set_yticklabels([])
            ax.tick_params(labelsize=TICKSIZE)
    if discrete:
        fig.legend(handles=[mpatches.Patch(color=c, label=s)
                            for s, c in STATE_COLOR.items()], fontsize=fontsize)
    else:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.65, 0.4, 0.05, 0.45])
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=np.nanmin(post[color_key]),
                               vmax=np.nanpercentile(post[color_key], 95)),
            cmap=cmap)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=TICKSIZE)
    if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + suffix + '.png'), dpi=150)
    plt.show()

if TOGGLE['corner_misfit']:
    plot_corner('misfit', 'jet', 'total misfit', '04b_corner')
if TOGGLE['corner_state']:
    plot_corner('state', None, 'BML state', '04c_corner_state', discrete=True)
if TOGGLE['corner_misfit_tt']:
    plot_corner('misfit_tt', 'plasma', 'TT misfit', '04d_corner_misfit_tt')
if TOGGLE['corner_moi_sigma']:
    plot_corner('moi_sigma', 'magma', 'MoI sigma', '04e_corner_moi_sigma')
if TOGGLE['corner_tt_per']:
    plot_corner('tt_per_datum', 'plasma', r'TT misfit / datum ($\sigma$)',
                '04f_corner_tt_per_datum')

# ══════════════════════════════════════════════════════════════════════════
# 相關性
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['correlations']:
    print('\n' + '=' * 64)
    print('Spearman correlation with misfit_tt / moi_sigma')
    print('=' * 64)
    for p in PARAMS_LIVE:
        r_tt, _  = spearmanr(post[p], post['misfit_tt'], nan_policy='omit')
        r_moi, _ = spearmanr(post[p], post['moi_sigma'], nan_policy='omit')
        print(f"{p:15s}  misfit_tt r={r_tt:+.3f}   moi_sigma r={r_moi:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for st in ['molten', 'layered', 'solid']:
        msk = (post['state'] == st)
        if msk.sum():
            ax.scatter(post['misfit_tt'][msk], post['moi_sigma'][msk],
                       c=STATE_COLOR[st], s=6, alpha=0.6, label=st, edgecolors='none')
    ax.set_xlabel('TT misfit', fontsize=fontsize)
    ax.set_ylabel('MoI sigma', fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE); ax.legend(fontsize=12)
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖K  核診斷
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['core_diag']:
    if not has_data('R_ic_km'):
        print('\ncore_diag: no core fields (pre-a35 run) — skipped')
    else:
        Tc, ws, ric = post['T_core'], post['w_S'], post['R_ic_km']
        fin = np.isfinite(Tc)
        print('\n' + '=' * 64)
        print('T_core / w_S exclusion probabilities (posterior mass)')
        print('=' * 64)
        for T0 in T_EXCLUDE_PROBES:
            tag = '  <- Drilleau 2026 BML median' if T0 == 3437.0 else ''
            print(f'  P(T_core > {T0:6.0f} K) = {np.mean(Tc[fin] > T0):6.1%}{tag}')
        ws_f = ws[np.isfinite(ws)]
        print(f'  P(w_S > {WS_GEOCHEM:.2f})       = {np.mean(ws_f > WS_GEOCHEM):6.1%}')
        if PRIOR['w_S'][1] > WS_STOICH:
            print(f'  P(w_S > {WS_STOICH:.3f})      = {np.mean(ws_f > WS_STOICH):6.1%}'
                  f'   <- 超化學計量,報告前需截斷')
        print(f'  P(R_ic > 0)          = {np.mean(ric[np.isfinite(ric)] > 0):6.1%}')
        qs = np.nanpercentile(ric, [5, 50, 95])
        print(f'  R_ic 5/50/95%        = {qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} km'
              f'   (Bi 2025: {BI_RIC_KM:.0f} ± {BI_RIC_ERR:.0f})')

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        (aA, aB, aC), (aD, aE, aF) = axes
        aA.hist(ws_f, bins=40, color=COLORS[2])
        aA.axvline(WS_GEOCHEM, color='k', ls=':', lw=2, label='geochem 17 wt%')
        if PRIOR['w_S'][1] > WS_STOICH:
            aA.axvline(WS_STOICH, color='k', ls='--', lw=1.5, label='stoich 36.5 wt%')
        aA.axvline(np.nanmedian(ws), color='red', lw=1.5, label='median')
        aA.set_xlim(*PRIOR['w_S'])
        aA.set_xlabel('w$_S$', fontsize=fontsize); aA.legend(fontsize=10)

        sc = aB.scatter(Tc, ws, c=post['moi_sigma'], cmap='magma', s=5, linewidths=0)
        fig.colorbar(sc, ax=aB, label='MoI sigma')
        aB.set_xlabel('T$_{core}$ (K)', fontsize=fontsize)
        aB.set_ylabel('w$_S$', fontsize=fontsize)

        for st in ['molten', 'layered', 'solid']:
            msk = (post['state'] == st) & np.isfinite(ric)
            if msk.sum():
                aC.scatter(Tc[msk], ric[msk], c=STATE_COLOR[st], s=6,
                           alpha=0.6, label=st, edgecolors='none')
        aC.axhspan(BI_RIC_KM-BI_RIC_ERR, BI_RIC_KM+BI_RIC_ERR, color='#7B3FA0', alpha=0.20)
        aC.axhspan(BI_RIC_KM-BI_RIC_SIG, BI_RIC_KM+BI_RIC_SIG, color='#7B3FA0', alpha=0.10)
        aC.axhline(RIC_MAX_KM, color='k', ls='--', lw=1, label=f'InSight <{RIC_MAX_KM:.0f} km')
        aC.set_xlabel('T$_{core}$ (K)', fontsize=fontsize)
        aC.set_ylabel('R$_{ic}$ (km)', fontsize=fontsize)
        aC.legend(fontsize=9)

        ric_f = ric[np.isfinite(ric)]
        aD.hist(ric_f, bins=40, color=COLORS[7])
        aD.axvspan(BI_RIC_KM-BI_RIC_ERR, BI_RIC_KM+BI_RIC_ERR, color='#7B3FA0',
                   alpha=0.2, label='Bi 612±50')
        aD.set_xlabel('R$_{ic}$ (km)', fontsize=fontsize)
        aD.set_yscale('log'); aD.legend(fontsize=10)

        vpc = post['Vp_core_cmb']
        aE.hist(vpc[np.isfinite(vpc)], bins=40, color=COLORS[5])
        aE.axvspan(*IRVING_VP_CMB, color='k', alpha=0.15, label='Irving 2023 4.9-5.0')
        aE.set_xlabel('V$_P$(CMB) km/s  [predictive]', fontsize=fontsize - 1)
        aE.legend(fontsize=10)

        rcm = post['rho_core_mean']
        aF.hist(rcm[np.isfinite(rcm)], bins=40, color=COLORS[8])
        aF.axvspan(*RHO_MEAN_MSL, color='k', alpha=0.15, label='MSL 6.5-6.65')
        aF.set_xlabel(r'$\bar\rho_{core}$ (g/cm$^3$)', fontsize=fontsize)
        aF.legend(fontsize=10)

        for ax in axes.ravel():
            ax.tick_params(labelsize=TICKSIZE)
        fig.suptitle(f'{PREFIX}  Core diagnostics', fontsize=fontsize + 1)
        fig.tight_layout()
        if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'K_core_diag.png'), dpi=150)
        plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 圖L  自洽結構診斷
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['struct_diag']:
    if not (has_data('gap_min') or 'R_cmb' in PARAMS_LIVE):
        print('\n[struct_diag] 沒有自洽結構欄位 -> 跳過')
    else:
        def _q(v, name, unit=''):
            v = np.asarray(v, float); v = v[np.isfinite(v)]
            if v.size == 0:
                print(f'  {name:<20s} --'); return
            lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
            print(f'  {name:<20s} {md:9.3f}  [{lo:8.3f}, {hi:8.3f}] {unit}')
        print('\n' + '=' * 96)
        print('self-consistent structure (median [95% CI])')
        print('=' * 96)
        for key, unit in [('R_cmb', 'km'), ('cmb_depth_km', 'km'),
                          ('bml_top_depth_km', 'km'), ('P_cmb', 'GPa'),
                          ('P_bml_top', 'GPa'), ('z_lit_km', 'km'),
                          ('R_ic_km', 'km'), ('gap_min', 'km'),
                          ('sc_n_iter', ''), ('sc_dP', 'GPa'), ('sc_dz_lit', 'km')]:
            _q(post[key], key, unit)
        if has_data('gap_min'):
            g = post['gap_min'][np.isfinite(post['gap_min'])]
            print(f"\n  gap_min < 0 的樣本: {100*np.mean(g < 0):.1f}%")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        (aA, aB), (aC, aD) = axes
        v = post['R_cmb'][np.isfinite(post['R_cmb'])]
        if v.size:
            aA.hist(v, bins=40, color=COLORS[2], edgecolor='none')
            aA.axvline(MARS_RADIUS - FALLBACK_CMB_DEPTH, color=COLORS[0], lw=2,
                       ls='--', label='old fixed')
            aA.axvline(np.median(v), color='k', lw=1.5, label='median')
            aA.legend(fontsize=10)
        aA.set_xlabel(r'R$_{CMB}$ (km)', fontsize=fontsize)

        aB.hist(post['cmb_depth_km'], bins=40, color=COLORS[3], alpha=0.75, label='CMB depth')
        aB.hist(post['bml_top_depth_km'][np.isfinite(post['bml_top_depth_km'])],
                bins=40, color=COLORS[5], alpha=0.75, label='BML top depth')
        aB.set_xlabel('depth (km)', fontsize=fontsize); aB.legend(fontsize=10)

        ni, dp = post['sc_n_iter'], post['sc_dP']
        m = np.isfinite(ni) & np.isfinite(dp)
        if m.any():
            sc = aC.scatter(ni[m], np.maximum(dp[m], 1e-12), s=8,
                            c=post['misfit'][m], cmap='viridis')
            plt.colorbar(sc, ax=aC, label='misfit')
            aC.set_yscale('log')
        aC.set_xlabel('sc_n_iter', fontsize=fontsize)
        aC.set_ylabel('sc_dP (GPa)', fontsize=fontsize)

        g = post['gap_min'][np.isfinite(post['gap_min'])]
        if g.size:
            aD.hist(g, bins=40, color=COLORS[8], edgecolor='none')
            aD.axvline(0, color=COLORS[0], lw=2, ls='--', label='gap = 0')
            aD.legend(fontsize=10)
        aD.set_xlabel('gap_min (km)', fontsize=fontsize)

        for ax in axes.ravel():
            ax.tick_params(labelsize=TICKSIZE)
        fig.suptitle(f'{PREFIX}  structure diagnostics', fontsize=fontsize + 1)
        fig.tight_layout()
        if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'L_struct_diag.png'), dpi=150)
        plt.show()

# ══════════════════════════════════════════════════════════════════════════
# prior -> posterior 收縮 + KL + edge check + 圖J
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['prior_post_shrinkage']:
    print('\n' + '=' * 96)
    print(f'prior vs posterior shrinkage   [w_S prior = {PRIOR["w_S"]}]')
    print('=' * 96)
    print(f"{'param':<15s}{'prior sd':>11s}{'post sd':>11s}{'ratio':>8s}"
          f"{'shrink':>9s}{'bits':>7s}   {'post 90%':<21s}")
    rows = []
    for k in PARAMS_LIVE:
        a, b = PRIOR[k]
        p_sd = (b - a) / np.sqrt(12.0)
        q_sd = np.nanstd(post[k], ddof=1)
        ratio = q_sd / p_sd
        qlo, qhi = np.nanpercentile(post[k], [5, 95])
        print(f'{k:<15s}{p_sd:11.4g}{q_sd:11.4g}{ratio:8.3f}{1-ratio:8.0%}'
              f'{np.log2(p_sd/q_sd):7.2f}   {qlo:8.4g}-{qhi:<11.4g}')
        rows.append((k, ratio))
    best, worst = min(rows, key=lambda r: r[1]), max(rows, key=lambda r: r[1])
    print(f'\n  strongest: {best[0]} ({1-best[1]:.0%})   weakest: {worst[0]} ({1-worst[1]:.0%})')

    print('\n' + '=' * 96)
    print('edge check: posterior density at prior walls (>30% => truncation matters)')
    print('=' * 96)
    for k in PARAMS_LIVE:
        a, b = PRIOR[k]
        vals = post[k][np.isfinite(post[k])]
        n_out = int(np.sum((vals < a) | (vals > b)))
        h, _ = np.histogram(vals, bins=25, range=(a, b))
        h = h / max(h.max(), 1)
        note = '   <-- check' if max(h[0], h[-1]) > 0.30 else ''
        oob  = f'   OUT-OF-PRIOR {n_out}' if n_out else ''
        print(f'  {k:<15s} left {h[0]:6.1%}   right {h[-1]:6.1%}{note}{oob}')
    # 樣本落在宣告的 PRIOR 之外 = WS_HI_BY_RUN 對這個 run 設錯了,或 run 混了兩種設定
    print('  (OUT-OF-PRIOR 非零表示 PRIOR 設定與該 run 的 runtime 值不符)')

    fig, axes = _grid(len(PARAMS_LIVE), ncols=4, figsize_per=(4.3, 4.2))
    for ax, k in zip(axes, PARAMS_LIVE):
        a, b = PRIOR[k]
        nb = 45
        vals = post[k][np.isfinite(post[k])]
        h, e = np.histogram(vals, bins=nb, range=(a, b), density=True)
        dens = h * (b - a)
        ctr = 0.5 * (e[1:] + e[:-1])
        ax.bar(ctr, dens, width=(b - a) / nb, color='#3a7ca5', alpha=0.85, zorder=2)
        ax.axhline(1.0, color='0.35', lw=2, zorder=3)
        ax.axvline(np.nanmedian(post[k]), color='crimson', lw=2, zorder=4)
        if k == 'w_S':
            ax.axvline(WS_GEOCHEM, color='k', ls='-.', lw=1.5, zorder=4)
            if b > WS_STOICH:
                ax.axvline(WS_STOICH, color='k', ls='--', lw=1.5, zorder=4)
        ax.set_xlim(a, b); ax.set_ylim(0, max(1.35, dens.max() * 1.12))
        ax.set_xlabel(f'{k} [{UNITS[k]}]' if UNITS[k] else k, fontsize=fontsize)
        ax.set_ylabel('posterior / prior', fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE)
    fig.suptitle(f'{PREFIX}   grey line = prior', fontsize=fontsize + 1)
    fig.tight_layout()
    if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'J_prior_post.png'), dpi=150)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 收斂診斷(burn-in 與後驗組裝同一把刀)
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['convergence_diag']:
    lens = np.array([len(c) for c in all_chains])
    kept = np.array([len(c) - b for c, b in zip(all_chains, BURN)])
    print('\n' + '=' * 64)
    print('chain lengths (post burn-in)')
    print('=' * 64)
    for nm, L, k2 in zip(chain_names, lens, kept):
        print(f'  {nm:<20s} {L:6d} raw, {k2:6d} kept  '
              + '#' * int(40 * k2 / max(kept.max(), 1)))

    def chain_series(ch, key):
        if key in PARAMS:
            return np.array([float(s['params'].get(key, np.nan)) for s in ch])
        if key.endswith('_frac'):
            hs = np.array([np.nan if s.get('h_solid_km') is None
                           else float(s['h_solid_km']) for s in ch])
            hl = np.array([np.nan if s.get('h_liquid_km') is None
                           else float(s['h_liquid_km']) for s in ch])
            return (classify_state(hs, hl) == key[:-5]).astype(float)
        return np.array([np.nan if s.get(key) is None else float(s[key]) for s in ch])

    def rhat_ess(mat):
        m, n = mat.shape
        if n < 10 or not np.all(np.isfinite(mat)):
            return np.nan, np.nan
        W = mat.var(axis=1, ddof=1).mean()
        B = n * mat.mean(axis=1).var(ddof=1)
        if W <= 0:
            return np.nan, np.nan
        vp = (n - 1) / n * W + B / n
        def acov(c_):
            c_ = c_ - c_.mean()
            f = np.fft.rfft(c_, 2 * n)
            return np.fft.irfft(f * np.conj(f))[:n] / n
        gamma = np.mean([acov(mat[j]) for j in range(m)], axis=0)
        rho = 1.0 - (W - gamma) / vp
        s_, t_ = 0.0, 1
        while t_ + 1 < n:
            pair = rho[t_] + rho[t_ + 1]
            if pair < 0: break
            s_ += pair; t_ += 2
        return np.sqrt(vp / W), m * n / (1.0 + 2.0 * s_)

    KEYS = ['layered_frac', 'solid_frac', 'molten_frac', 'misfit'] + PARAMS_LIVE

    # 每鏈只算一次序列 + 砍頭
    chain_series_cache = []
    for ch, b in zip(all_chains, BURN):
        entry = {k: chain_series(ch, k)[b:] for k in KEYS}
        entry['n'] = len(ch) - b
        chain_series_cache.append(entry)

    def rhat_table(idx, label):
        # 回傳三元組:(mats, 對齊長度, 實際採用的 chain index)
        idx = [i for i in idx if chain_series_cache[i]['n'] >= RHAT_MIN_N]
        if not idx:
            print(f'\n{label}: no chains with >={RHAT_MIN_N} post-burn-in steps')
            return {}, 0, []
        trimmed = [chain_series_cache[i] for i in idx]
        n2 = min(t['n'] for t in trimmed)
        print('\n' + '=' * 64)
        print(f'{label}   {len(idx)} chains, aligned tail n={n2}')
        print('=' * 64)
        print(f"{'param':<16s}{'R-hat':>8s}{'ESS':>9s}{'post-std':>11s}"
              f"{'cur step':>10s}{'suggest':>10s}")
        mats = {}
        for k in KEYS:
            mat = np.array([t[k][-n2:] for t in trimmed])
            mats[k] = mat
            r, e = rhat_ess(mat)
            sd = mat.std()
            cur = f"{CUR_STEP[k]:10.3g}" if k in CUR_STEP else ' ' * 10
            sug = f"{0.5*sd:10.3g}" if k in CUR_STEP else ' ' * 10
            flag = '' if (r == r and r < 1.1) else '  *'
            print(f'{k:<16s}{r:8.3f}{e:9.0f}{sd:11.4f}{cur}{sug}{flag}')
        return mats, n2, idx

    all_idx = list(range(len(all_chains)))
    long_idx = [i for i in all_idx if chain_series_cache[i]['n'] >= 0.5 * np.median(kept)]
    short_names = [chain_names[i] for i in all_idx if i not in long_idx]

    rhat_table(all_idx, 'A. all chains')
    if short_names:
        print(f'\n  short chains excluded: {", ".join(short_names)}')
        mats, n_used, used_idx = rhat_table(long_idx, 'B. long chains only')
    else:
        mats, n_used, used_idx = rhat_table(all_idx, 'B. (no short chains to exclude)')
    if not mats:
        raise SystemExit('convergence_diag: 沒有夠長的鏈可用')

    print('\n' + '=' * 64)
    print('per-chain posterior means')
    print('=' * 64)
    print(f"{'chain':<18s}" + ''.join(f'{k[:9]:>11s}' for k in PARAMS_LIVE))
    for j, i in enumerate(used_idx):
        print(f'{chain_names[i]:<18s}'
              + ''.join(f'{mats[k][j].mean():11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- pooled --":<18s}'
          + ''.join(f'{mats[k].mean():11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- between --":<18s}' +
          ''.join(f'{mats[k].mean(axis=1).std(ddof=1):11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- within --":<18s}' +
          ''.join(f'{mats[k].std(axis=1).mean():11.3f}' for k in PARAMS_LIVE))

    print('\n' + '=' * 64)
    print(f'chain length needed for ESS = {ESS_TARGET}')
    print('=' * 64)
    # tau 與 ESS 必須來自同一個 mats(B 表):m = len(used_idx), n = n_used
    for k in KEYS:
        r, e = rhat_ess(mats[k])
        if e == e and e > 0:
            tau = len(used_idx) * n_used / e
            need = ESS_TARGET * tau / len(used_idx)
            print(f'  {k:<16s} tau ~ {tau:7.0f} steps   need ~ {need:7.0f} post-burn-in '
                  f'per chain (now {n_used})')

    print('\n' + '=' * 64)
    print('state transitions per chain (post burn-in)')
    print('=' * 64)
    tot = 0
    for c in range(len(all_chains)):
        s = state[chain_id == c]
        t = int((s[1:] != s[:-1]).sum()); tot += t
        print(f'{chain_names[c]:<18s} {len(s):6d} {t:6d}  ' +
              str({k: int((s == k).sum()) for k in ['solid', 'layered', 'molten']}))
    print(f'total transitions: {tot}')

    fig, axes = _grid(len(PARAMS_LIVE), ncols=4, figsize_per=(4.3, 3.6))
    for ax, k in zip(axes, PARAMS_LIVE):
        for j in range(len(used_idx)):
            y = mats[k][j]
            ax.plot(np.cumsum(y) / np.arange(1, len(y) + 1), lw=1, alpha=0.8)
        ax.axhline(mats[k].mean(), color='k', ls='--', lw=1.5)
        ax.set_title(k, fontsize=fontsize)
        ax.set_xlabel('step (post burn-in)', fontsize=fontsize)
        ax.set_ylabel('running mean', fontsize=fontsize)
        ax.tick_params(labelsize=TICKSIZE)
    fig.suptitle(f'{PREFIX} running mean per chain', fontsize=fontsize + 2)
    fig.tight_layout()
    if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'I_running_mean.png'), dpi=150)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════
# acceptance sliding window(a47:看 S_PERP 切換台階;舊鏈沒有 accepted 也能跑)
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['acceptance_window']:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c, (ch, nm) in enumerate(zip(all_chains, chain_names)):
        acc = np.array([1.0 if s.get('accepted') else 0.0 for s in ch])
        if len(acc) < ACC_WINDOW:
            continue
        w = np.convolve(acc, np.ones(ACC_WINDOW)/ACC_WINDOW, mode='valid') * 100
        ax.plot(np.arange(len(w)) + ACC_WINDOW, w, lw=1, alpha=0.8,
                color=COLORS[c % len(COLORS)], label=nm.replace(PREFIX + '_', ''))
        ax.axvline(BURN[c], color=COLORS[c % len(COLORS)], ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Step', fontsize=fontsize)
    ax.set_ylabel(f'Acceptance (%, window={ACC_WINDOW})', fontsize=fontsize)
    ax.tick_params(labelsize=TICKSIZE)
    ax.legend(fontsize=8, ncol=4, loc='lower right')
    ax.set_title(f'{PREFIX} sliding-window acceptance (dotted = burn-in cut)',
                 fontsize=fontsize)
    plt.tight_layout()
    if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + 'M_acceptance_window.png'), dpi=150)
    plt.show()