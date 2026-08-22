#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:58:20 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mars BML MCMC — 合併後的後驗分析腳本(a35/core 版)
合併自 a33 版(收斂診斷: R-hat/ESS、prior->posterior 收縮、KL、chain 健檢)
與 a34 版(後驗圖: marginals、corner、misfit/moi 相關性)。
讀檔同時支援 chain.json(a31-a33)與 chain.jsonl(a34 起)。

a35 更新(對應 70.mcmc_core_nGibbs / a26 forward model):
  * 第七個參數 w_S(核 S 重量分率)進 PARAMS/PRIOR/CUR_STEP,
    marginals / corner / shrinkage / R-hat 自動涵蓋
  * PRIOR 同步 runtime:T_core (1550, 3600)、Mg# (0.50, 0.90)
    (舊版分析腳本的 Mg# 0.86 與 a34 runtime 0.90 不一致,已修正)
  * EXTRA_KEYS 加入核診斷量:R_ic_km / ric_sigma / rho_core_mean /
    Vp_core_cmb / P_center_core / T_center_core
  * 新 TOGGLE 'core_diag':T_core 排除機率表、ICB 溫度計後驗圖
    (T_core-R_ic + Bi 帶)、w_S vs 地球化學上限、posterior predictive
    (Vp_cmb vs Irving、rho_mean vs MSL 帶)
  * 讀舊鏈(無 w_S / 核欄位)不會爆:params 用 .get,核圖自動跳過
  * 圖格從 2x3 改 2x4(七參數),多餘的子圖隱藏

a36 更新(對應 70.mcmc_core_nGibbs 自洽結構版):
  * 第八個參數 R_cmb(核半徑)進 PARAMS/PRIOR/CUR_STEP;CMB 深度不再是常數,
    改由 post['cmb_depth_km'] = MARS_RADIUS - R_cmb 逐樣本算
  * PREFIX 必須指定 chain_a36A 或 chain_a36B —— A/B 是兩套不同的核 EoS
    (A = all-Liu Gamma mixing, B = Liu Fe + Sakai FeS Margules),不可合併
  * EXTRA_KEYS 移除 rho_S_interface(a36 不再輸出),加入自洽結構診斷:
    P_cmb / P_bml_top / z_lit_km / gap_min / bml_n_pass / sc_n_iter / sc_dP / sc_dz_lit
  * 新增 TOGGLE 'struct_diag':自洽迭代收斂、gap_min(內核撐滿液核)檢查
  * MIN_STEPS 提高到 300:a36 有一批 30-90 步的短鏈
"""

import json, glob, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════
# 設定
# ══════════════════════════════════════════════════════════════════════════
CHAIN_DIR   = '/Users/chingchen/Desktop/HeFESTo/mcmc/mcmc_chain'
OUT_DIR     = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'
PREFIX      = 'chain_a41B'   # 'chain_a36A' 或 'chain_a36B',不要寫 'chain_a36'
PHASE_TABLE = '/Users/chingchen/Desktop/HeFESTo/mcmcfigures/bml_phase_table_prior_a31.npz'

if PREFIX in ('chain_a36', 'chain_a36_'):
    raise SystemExit(
        "PREFIX='chain_a36' 會同時抓到 a36A 與 a36B。\n"
        "  A = all-Liu (Gamma mixing), B = Liu Fe + Sakai FeS (Margules),\n"
        "  是兩套不同的核 EoS,後驗不可合併。請指定 chain_a36A 或 chain_a36B。")



EXCLUDE = ['chain_a37A_01','chain_a38A_01'] 
plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.4
DROP_LAST   = 1
MIN_STEPS   = 20        
SAVE        = False

fontsize = 14             # 全域字級,tick_params 一律用 labelsize=14

RA_C            = 1.0e5      # 跟 runtime 的 _RA_C 一致
CHENG_TC        = (2200.0, 2400.0)
SEC_YR, G_MARS  = 3.156e7, 3.72
RHO_S_CONST     = 4.097      # g/cc, nGibbs at posterior XS, [3.98, 4.24]

MARS_RADIUS     = 3389.5     # km
# a36 起 R_cmb 是自由參數 -> CMB 深度逐樣本算(post['cmb_depth_km'])。
# 舊鏈沒有 R_cmb 時退回這個常數。
FALLBACK_CMB_DEPTH = 1743.3  # km
K_SOLID         = 4.0        # W/m/K
Q_CMB_RANGE     = (5.0, 20.0)   # mW/m2, 文獻估計
Q_CMB_MAX       = 40.0          # mW/m2, 穩態上限

ESS_TARGET = 400

# 核相關的參考值(core_diag 用)
BI_RIC_KM, BI_RIC_ERR, BI_RIC_SIG = 612.0, 50.0, 112.0   # Bi 2025;SIG 含 OC 速度系統項
RIC_MAX_KM      = 750.0          # InSight SKS 上限
WS_GEOCHEM      = 0.17           # Steenstra & van Westrenen 2018 地球化學上限
IRVING_VP_CMB   = (4.9, 5.0)     # km/s
RHO_MEAN_MSL    = (6.5, 6.65)    # g/cm3, Samuel/Khan MSL 情境核平均密度
T_EXCLUDE_PROBES = (2500.0, 3000.0, 3437.0)   # 3437 = Drilleau 2026 BML Tc 中位

PRIOR_FRAC          = {'solid': 0.284, 'layered': 0.132, 'molten': 0.584}
# a36 又把 R_cmb 放開(1400-1900 km),先驗的三態體積再變 -> 這張表更過期了
PRIOR_FRAC_IS_STALE = True    # T_core 下界 1800->1550、a36 R_cmb 自由,需重算

CUR_STEP = {'T_lit': 50.0, 'P_lit': 0.7, 'Mg#': 0.025,
            'T_core': 100.0, 'Mg#_bulk_bml': 0.06, 'BML_thickness': 25.0,
            'w_S': 0.02, 'R_cmb': 25.0}

PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),    # a34 runtime 值(舊分析腳本誤植 0.86)
    'T_core':        (1550.0, 3600.0),    # a26: 下界 1800->1550(ICB 溫度計臨界區)
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (  0.0,  400.0),
    'w_S':           (  0.05,   0.30),
    'R_cmb':         (1400.0, 1900.0),    # a36 新增
}
UNITS = {'T_lit': 'K', 'P_lit': 'GPa', 'Mg#': '', 'T_core': 'K',
         'Mg#_bulk_bml': '', 'BML_thickness': 'km', 'w_S': '', 'R_cmb': 'km'}

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)', 'w$_S$ (core S wt)',
          'R$_{CMB}$ (km)']
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}

# 從 record 撈出的純量欄位(a35 新增核診斷量;tt_miss_by_event 是 dict,排除)
EXTRA_KEYS = ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
              'upper_contrast', 'lower_contrast', 'Ra', 'h_solid_km', 'h_liquid_km',
              'Mg_solid', 'Mg_liquid', 'T_interface', 'T_mantle_bottom', 'moi_pred',
              'tt_n_ph', 'tt_n_miss', 'tt_n_capped',
              # a35 核診斷
              'R_ic_km', 'ric_sigma', 'rho_core_mean', 'Vp_core_cmb',
              'P_center_core', 'T_center_core',
              # a36 自洽結構診斷(rho_S_interface 在 a36 已不輸出,移除)
              'P_cmb', 'P_bml_top', 'z_lit_km', 'gap_min', 'bml_n_pass',
              'sc_n_iter', 'sc_dP', 'sc_dz_lit']

# ── 各段落開關(1=執行, 0=跳過)─────────────────────────────────────────────
TOGGLE = {
    'marginals':             1,   # 圖4  1D marginals
    'corner_misfit':         1,   # 圖4b corner, colour=misfit
    'corner_state':          1,   # 圖4c corner, colour=BML state
    'corner_misfit_tt':      1,   # 圖4d corner, colour=misfit_tt
    'corner_moi_sigma':      1,   # 圖4e corner, colour=moi_sigma
    'correlations':          1,   # spearman + misfit_tt vs moi_sigma 散點
    'core_diag':             1,   # 圖K  核診斷:T_core 排除、ICB 溫度計、predictive
    'struct_diag':           1,   # 圖L  a36 自洽結構:R_cmb/CMB 深度、迭代收斂、gap_min
    'prior_post_shrinkage':  1,   # 圖J + 收縮量化表 + KL + edge check
    'convergence_diag':      1,   # chain length / R-hat / ESS / running mean / transitions
}

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''


def _grid(n, ncols=4, figsize_per=(4.0, 4.0)):
    """n 個子圖的網格;多餘的隱藏。回傳 fig, 長度 n 的 axes list。"""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0]*ncols, figsize_per[1]*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# ══════════════════════════════════════════════════════════════════════════
# 讀檔:同時支援 chain.json(a31-a33)與 chain.jsonl(a34 起)
# ══════════════════════════════════════════════════════════════════════════
def load_chain_json(path):
    """salvage-capable reader,給舊版整檔 chain.json 用"""
    if not os.path.exists(path):
        return None, 'missing'
    with open(path) as fh:
        text = fh.read()
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
    """逐行讀 chain.jsonl;只有最後一行可能寫到一半,壞掉就丟棄該行"""
    if not os.path.exists(path):
        return None, 'missing'
    recs, n_bad = [], 0
    with open(path) as fh:
        for line in fh:
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


def load_chain_dir(chain_dir):
    """優先找 chain.jsonl(a34+),沒有就退回 chain.json(a31-a33)"""
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    json_path  = os.path.join(chain_dir, 'chain.json')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(json_path)


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


# ══════════════════════════════════════════════════════════════════════════
# 後驗(逐鏈套 burn-in,長度不必相同)
# ══════════════════════════════════════════════════════════════════════════
samples, chain_id = [], []
for c, ch in enumerate(all_chains):
    b = int(len(ch) * BURNIN_FRAC)
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"post burn-in samples: {len(samples)}")


def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])


# 舊鏈(a34 之前)沒有 w_S:用 .get 給 NaN,不炸
post = {p: np.array([float(s['params'].get(p, np.nan)) for s in samples])
        for p in PARAMS}
for k in EXTRA_KEYS:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

_missing = [p for p in PARAMS if np.all(~np.isfinite(post[p]))]
if _missing:
    print(f"NOTE: params absent in these chains (pre-core version?): {_missing}")

h_sol, h_liq = post['h_solid_km'], post['h_liquid_km']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= 1.0) & (h_liq >= 1.0)] = 'layered'
state[(h_sol <  1.0) & (h_liq >= 1.0)] = 'molten'
state[(h_sol >= 1.0) & (h_liq <  1.0)] = 'solid'
post['state'] = state
f_solid = h_sol / np.maximum(h_sol + h_liq, 1e-9)
print("state counts:", {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})

# corner / marginals 只畫實際存在的參數
PARAMS_LIVE = [p for p in PARAMS if p not in _missing]
LABELS_LIVE = [LABELS[PARAMS.index(p)] for p in PARAMS_LIVE]

# ── a36 幾何: CMB 深度是分布不是常數 ──────────────────────────────────────────
if 'R_cmb' in PARAMS_LIVE:
    post['cmb_depth_km'] = MARS_RADIUS - post['R_cmb']
else:
    post['cmb_depth_km'] = np.full(len(samples), FALLBACK_CMB_DEPTH)
post['bml_top_depth_km'] = post['cmb_depth_km'] - post['BML_thickness']
post['bml_top_r_km']     = MARS_RADIUS - post['bml_top_depth_km']


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
        ax.set_xlabel(label, fontsize=fontsize); ax.set_ylabel('count', fontsize=fontsize)
        ax.tick_params(labelsize=14); ax.legend(fontsize=9)
    plt.tight_layout()
    if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + '04_marginals.png'), dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 圖4b-e  corner 系列
# ══════════════════════════════════════════════════════════════════════════
def plot_corner(color_key, cmap, cbar_label, suffix, discrete=False):
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
            ax.tick_params(labelsize=12)
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
        cbar.ax.tick_params(labelsize=14)
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


# ══════════════════════════════════════════════════════════════════════════
# misfit_tt / moi_sigma 相關性
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
    for st in ['molten', 'layered', 'solid']:   # molten 先畫,稀有的畫在最上層
        msk = (post['state'] == st)
        if msk.sum():
            ax.scatter(post['misfit_tt'][msk], post['moi_sigma'][msk],
                       c=STATE_COLOR[st], s=6, alpha=0.6, label=st, edgecolors='none')
    ax.set_xlabel('TT misfit', fontsize=fontsize); ax.set_ylabel('MoI sigma', fontsize=fontsize)
    ax.tick_params(labelsize=14); ax.legend(fontsize=12)
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    sca = ax.scatter(post['misfit_tt'], post['moi_sigma'], c=post['Mg#'],
                     cmap='coolwarm', s=4, alpha=0.6, linewidths=0)
    ax.set_xlabel('TT misfit', fontsize=fontsize); ax.set_ylabel('MoI sigma', fontsize=fontsize)
    ax.tick_params(labelsize=14)
    cbar = fig.colorbar(sca)
    cbar.set_label('Mg# (mantle)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 圖K  核診斷:T_core 排除機率、ICB 溫度計後驗、posterior predictive
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['core_diag']:
    has_core = np.any(np.isfinite(post['R_ic_km']))
    if not has_core:
        print('\ncore_diag: chains have no core fields (pre-a35 run) — skipped')
    else:
        Tc, ws, ric = post['T_core'], post['w_S'], post['R_ic_km']
        fin = np.isfinite(Tc)

        print('\n' + '=' * 64)
        print('T_core / w_S exclusion probabilities (posterior mass)')
        print('=' * 64)
        for T0 in T_EXCLUDE_PROBES:
            tag = '  <- Drilleau 2026 BML median' if T0 == 3437.0 else ''
            print(f'  P(T_core > {T0:6.0f} K) = {np.mean(Tc[fin] > T0):6.1%}{tag}')
        print(f'  P(T_core < 1800 K)   = {np.mean(Tc[fin] < 1800.0):6.1%}'
              f'   (舊 prior 下界之下的質量)')
        print(f'  P(w_S > {WS_GEOCHEM:.2f})       = '
              f'{np.mean(ws[np.isfinite(ws)] > WS_GEOCHEM):6.1%}'
              f'   (超出地球化學上限)')
        print(f'  P(R_ic > 0)          = {np.mean(ric[np.isfinite(ric)] > 0):6.1%}'
              f'   (有內核的樣本比例)')
        qs = np.nanpercentile(ric, [5, 50, 95])
        print(f'  R_ic 5/50/95%        = {qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} km'
              f'   (Bi 2025: {BI_RIC_KM:.0f} ± {BI_RIC_ERR:.0f})')
        with np.errstate(invalid='ignore'):
            in_bi = np.abs(ric - BI_RIC_KM) <= BI_RIC_SIG
        print(f'  P(|R_ic-612|<{BI_RIC_SIG:.0f})   = {np.nanmean(in_bi):6.1%}'
              f'   (落在 Bi 帶內;Run A 這是 predictive,Run B 是 fit)')

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        (aA, aB, aC), (aD, aE, aF) = axes

        # (A) w_S marginal + 上限
        aA.hist(ws[np.isfinite(ws)], bins=40, color=COLORS[2])
        aA.axvline(WS_GEOCHEM, color='k', ls=':', lw=2, label='geochem 17 wt%')
        aA.axvline(np.nanmedian(ws), color='red', lw=1.5, label='median')
        aA.set_xlabel('w$_S$', fontsize=fontsize); aA.legend(fontsize=10)

        # (B) T_core vs w_S, colour = moi_sigma(核密度 trade-off 的主戰場)
        sc = aB.scatter(Tc, ws, c=post['moi_sigma'], cmap='magma', s=5,
                        linewidths=0)
        fig.colorbar(sc, ax=aB, label='MoI sigma')
        aB.set_xlabel('T$_{core}$ (K)', fontsize=fontsize)
        aB.set_ylabel('w$_S$', fontsize=fontsize)

        # (C) ICB 溫度計後驗:T_core vs R_ic,BML 三態上色,Bi 帶
        for st in ['molten', 'layered', 'solid']:
            msk = (post['state'] == st) & np.isfinite(ric)
            if msk.sum():
                aC.scatter(Tc[msk], ric[msk], c=STATE_COLOR[st], s=6,
                           alpha=0.6, label=st, edgecolors='none')
        aC.axhspan(BI_RIC_KM-BI_RIC_ERR, BI_RIC_KM+BI_RIC_ERR,
                   color='#7B3FA0', alpha=0.20)
        aC.axhspan(BI_RIC_KM-BI_RIC_SIG, BI_RIC_KM+BI_RIC_SIG,
                   color='#7B3FA0', alpha=0.10)
        aC.axhline(RIC_MAX_KM, color='k', ls='--', lw=1,
                   label=f'InSight <{RIC_MAX_KM:.0f} km')
        aC.set_xlabel('T$_{core}$ (K)', fontsize=fontsize)
        aC.set_ylabel('R$_{ic}$ (km)', fontsize=fontsize)
        aC.legend(fontsize=9)

        # (D) R_ic 分布(0 與 >0 分開看)
        ric_f = ric[np.isfinite(ric)]
        aD.hist(ric_f, bins=40, color=COLORS[7])
        aD.axvspan(BI_RIC_KM-BI_RIC_ERR, BI_RIC_KM+BI_RIC_ERR,
                   color='#7B3FA0', alpha=0.2, label='Bi 612±50')
        aD.set_xlabel('R$_{ic}$ (km)', fontsize=fontsize)
        aD.set_yscale('log'); aD.legend(fontsize=10)

        # (E) posterior predictive: Vp at CMB vs Irving
        vpc = post['Vp_core_cmb']
        aE.hist(vpc[np.isfinite(vpc)], bins=40, color=COLORS[5])
        aE.axvspan(*IRVING_VP_CMB, color='k', alpha=0.15,
                   label='Irving 2023 4.9-5.0')
        aE.set_xlabel('V$_P$(CMB) km/s  [predictive, not in likelihood]',
                      fontsize=fontsize - 1)
        aE.legend(fontsize=10)

        # (F) posterior predictive: rho_core_mean vs MSL 帶
        rcm = post['rho_core_mean']
        aF.hist(rcm[np.isfinite(rcm)], bins=40, color=COLORS[8])
        aF.axvspan(*RHO_MEAN_MSL, color='k', alpha=0.15,
                   label='Samuel/Khan MSL 6.5-6.65')
        aF.set_xlabel(r'$\bar\rho_{core}$ (g/cm$^3$)', fontsize=fontsize)
        aF.legend(fontsize=10)

        for ax in axes.ravel():
            ax.tick_params(labelsize=13)
        fig.suptitle(f'{PREFIX}  Core diagnostics '
                     f'(ric_sigma in likelihood: '
                     f'{"yes (Run B)" if np.nanmax(post["ric_sigma"]) > 0 else "no (Run A)"})',
                     fontsize=fontsize + 1)
        fig.tight_layout()
        if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'K_core_diag.png'), dpi=150)
        plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 圖L  a36 自洽結構診斷
#   (A) R_cmb 後驗 vs 舊的固定值 1743.3 km 對應的 R_cmb
#   (B) CMB 深度 / BML 頂部深度
#   (C) 自洽迭代步數 sc_n_iter 與殘差 sc_dP / sc_dz_lit
#   (D) gap_min: 負值代表內核幾乎撐滿液核,是 a36 新的物理約束訊號
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['struct_diag']:
    _have_struct = np.isfinite(post['gap_min']).any() or 'R_cmb' in PARAMS_LIVE
    if not _have_struct:
        print('\n[struct_diag] 這批鏈沒有 a36 自洽結構欄位 -> 跳過')
    else:
        def _q(v, name, unit=''):
            v = np.asarray(v, float); v = v[np.isfinite(v)]
            if v.size == 0:
                print(f'  {name:<20s} --'); return
            lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
            print(f'  {name:<20s} {md:9.3f}  [{lo:8.3f}, {hi:8.3f}] {unit}')

        print('\n' + '=' * 96)
        print('a36 self-consistent structure (median [95% CI])')
        print('=' * 96)
        _q(post['R_cmb'],            'R_cmb',          'km')
        _q(post['cmb_depth_km'],     'CMB depth',      'km')
        _q(post['bml_top_depth_km'], 'BML top depth',  'km')
        _q(post['P_cmb'],            'P_cmb',          'GPa')
        _q(post['P_bml_top'],        'P_bml_top',      'GPa')
        _q(post['z_lit_km'],         'z_lit',          'km')
        _q(post['R_ic_km'],          'R_ic',           'km')
        _q(post['gap_min'],          'gap_min',        'km')
        _q(post['sc_n_iter'],        'sc_n_iter',      '')
        _q(post['sc_dP'],            'sc_dP',          'GPa')
        _q(post['sc_dz_lit'],        'sc_dz_lit',      'km')

        if np.isfinite(post['gap_min']).any():
            g = post['gap_min'][np.isfinite(post['gap_min'])]
            print(f"\n  gap_min < 0 的樣本: {100*np.mean(g < 0):.1f}%  "
                  f"(內核與 CMB 之間沒有留下液態外核)")
        _cd = post['cmb_depth_km']
        print(f"  舊腳本固定的 CMB 深度 {FALLBACK_CMB_DEPTH:.1f} km 落在後驗的 "
              f"{100*np.mean(_cd <= FALLBACK_CMB_DEPTH):.1f}% 分位")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        (aA, aB), (aC, aD) = axes

        # (A) R_cmb
        v = post['R_cmb'][np.isfinite(post['R_cmb'])]
        if v.size:
            aA.hist(v, bins=40, color=COLORS[2], edgecolor='none')
            aA.axvline(MARS_RADIUS - FALLBACK_CMB_DEPTH, color=COLORS[0], lw=2,
                       ls='--', label=f'old fixed = {MARS_RADIUS-FALLBACK_CMB_DEPTH:.1f} km')
            aA.axvline(np.median(v), color='k', lw=1.5, label='median')
            aA.legend(fontsize=10)
        aA.set_xlabel(r'R$_{CMB}$ (km)', fontsize=fontsize)
        aA.set_ylabel('count', fontsize=fontsize)

        # (B) CMB 深度 vs BML 頂部深度
        aB.hist(post['cmb_depth_km'], bins=40, color=COLORS[3],
                alpha=0.75, label='CMB depth')
        aB.hist(post['bml_top_depth_km'][np.isfinite(post['bml_top_depth_km'])],
                bins=40, color=COLORS[5], alpha=0.75, label='BML top depth')
        aB.set_xlabel('depth (km)', fontsize=fontsize)
        aB.legend(fontsize=10)

        # (C) 自洽迭代收斂
        ni, dp = post['sc_n_iter'], post['sc_dP']
        m = np.isfinite(ni) & np.isfinite(dp)
        if m.any():
            sc = aC.scatter(ni[m], np.maximum(dp[m], 1e-12), s=8,
                            c=post['misfit'][m], cmap='viridis')
            plt.colorbar(sc, ax=aC, label='misfit')
            aC.set_yscale('log')
        aC.set_xlabel('sc_n_iter', fontsize=fontsize)
        aC.set_ylabel('sc_dP (GPa)', fontsize=fontsize)

        # (D) gap_min
        g = post['gap_min'][np.isfinite(post['gap_min'])]
        if g.size:
            aD.hist(g, bins=40, color=COLORS[8], edgecolor='none')
            aD.axvline(0, color=COLORS[0], lw=2, ls='--', label='gap = 0')
            aD.legend(fontsize=10)
        aD.set_xlabel('gap_min (km)   [<0 = inner core fills the liquid core]',
                      fontsize=fontsize - 1)
        aD.set_ylabel('count', fontsize=fontsize)

        for ax in axes.ravel():
            ax.tick_params(labelsize=14)
        fig.suptitle(f'{PREFIX}  self-consistent structure diagnostics',
                     fontsize=fontsize + 1)
        fig.tight_layout()
        if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'L_struct_diag.png'), dpi=150)
        plt.show()


# ══════════════════════════════════════════════════════════════════════════
# prior -> posterior 收縮量化 + KL + edge check + 圖J
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['prior_post_shrinkage']:
    print('\n' + '=' * 96)
    print('prior vs posterior: how much did the data shrink each parameter?')
    print('=' * 96)
    print(f"{'param':<15s}{'prior sd':>11s}{'post sd':>11s}{'ratio':>8s}"
          f"{'shrink':>9s}{'bits':>7s}   {'prior 90%':<21s}{'post 90%':<21s}")
    rows = []
    for k in PARAMS_LIVE:
        a, b = PRIOR[k]
        p_sd = (b - a) / np.sqrt(12.0)
        q_sd = np.nanstd(post[k], ddof=1)
        ratio = q_sd / p_sd
        bits = np.log2(p_sd / q_sd)
        plo, phi_ = a + 0.05 * (b - a), a + 0.95 * (b - a)
        qlo, qhi = np.nanpercentile(post[k], [5, 95])
        print(f'{k:<15s}{p_sd:11.4g}{q_sd:11.4g}{ratio:8.3f}{1-ratio:8.0%}{bits:7.2f}   '
              f'{plo:8.4g}-{phi_:<11.4g}{qlo:8.4g}-{qhi:<11.4g}')
        rows.append((k, ratio, bits))
    print('\n  ratio = post sd / prior sd  (1.0 = data said nothing)')
    best, worst = min(rows, key=lambda r: r[1]), max(rows, key=lambda r: r[1])
    print(f'  strongest: {best[0]}  (shrunk {1-best[1]:.0%}, {best[2]:.2f} bits)')
    print(f'  weakest:   {worst[0]}  (shrunk {1-worst[1]:.0%}, {worst[2]:.2f} bits)'
          f'  <- state this honestly in the paper')

    print('\n' + '=' * 96)
    print('information: sd-ratio bits vs KL divergence (mismatch => non-Gaussian posterior)')
    print('=' * 96)
    GAUSS_OFFSET = np.log2(np.sqrt(12.0)) - np.log2(np.sqrt(2 * np.pi * np.e))
    print(f"{'param':<15s}{'sd bits':>10s}{'KL bits':>10s}{'gauss-eq':>10s}{'excess':>9s}   note")
    for k in PARAMS_LIVE:
        a, b = PRIOR[k]
        nb = 40
        vals = post[k][np.isfinite(post[k])]
        h, _ = np.histogram(vals, bins=nb, range=(a, b), density=True)
        q = h * (b - a) / nb
        p_ = np.full(nb, 1.0 / nb)
        m = q > 0
        kl = float(np.sum(q[m] * np.log2(q[m] / p_[m])))
        sd_bits = np.log2(((b - a) / np.sqrt(12.0)) / np.nanstd(post[k], ddof=1))
        geq = sd_bits + GAUSS_OFFSET
        exc = kl - geq
        note = ('non-Gaussian (ramp/one-sided), sd underestimates info' if exc > 0.10
                else 'flatter or multimodal than Gaussian' if exc < -0.10
                else 'near-Gaussian, sd ratio is fine')
        print(f'{k:<15s}{sd_bits:10.2f}{kl:10.2f}{geq:10.2f}{exc:+9.2f}   {note}')

    print('\n' + '=' * 96)
    print('edge check: posterior density at the prior walls (>30% => prior truncation matters)')
    print('=' * 96)
    for k in PARAMS_LIVE:
        a, b = PRIOR[k]
        vals = post[k][np.isfinite(post[k])]
        h, _ = np.histogram(vals, bins=25, range=(a, b))
        h = h / max(h.max(), 1)
        print(f'  {k:<15s} left {h[0]:6.1%}   right {h[-1]:6.1%}'
              + ('   <-- check' if max(h[0], h[-1]) > 0.30 else ''))

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
        lo, hi = np.nanpercentile(post[k], [5, 95])
        ax.axvline(lo, color='k', ls=':', lw=1.2, zorder=4)
        ax.axvline(hi, color='k', ls=':', lw=1.2, zorder=4)
        ax.axvline(np.nanmedian(post[k]), color='crimson', lw=2, zorder=4)
        if k == 'w_S':
            ax.axvline(WS_GEOCHEM, color='k', ls='-.', lw=1.5, zorder=4)
        ax.set_xlim(a, b); ax.set_ylim(0, max(1.35, dens.max() * 1.12))
        ax.set_xlabel(f'{k} [{UNITS[k]}]' if UNITS[k] else k, fontsize=fontsize)
        ax.set_ylabel('posterior / prior density', fontsize=fontsize)
        r = np.nanstd(post[k], ddof=1) / ((b - a) / np.sqrt(12.0))
        warn = '  [EDGE]' if max(dens[0], dens[-1]) > 1.0 else ''
        ax.set_title(f'{k}: shrink {1-r:.0%}{warn}', fontsize=fontsize)
        ax.tick_params(labelsize=14)
    fig.suptitle('grey line = prior (1.0), bars above 1 = data moved probability here',
                 fontsize=fontsize + 1)
    fig.tight_layout()
    if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'J_prior_post.png'), dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 收斂診斷: chain length / R-hat / ESS / per-chain means / running mean /
#           state transitions
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['convergence_diag']:
    lens = np.array([len(c) for c in all_chains])

    print('\n' + '=' * 64)
    print('chain lengths')
    print('=' * 64)
    for nm, L in zip(chain_names, lens):
        print(f'  {nm:<18s} {L:6d}  ' + '#' * int(40 * L / lens.max()))
    print(f'\n  min={lens.min()}  median={int(np.median(lens))}  max={lens.max()}')
    print(f'  truncating to min discards {lens.sum() - lens.min()*len(lens)} steps '
          f'({100*(1 - lens.min()*len(lens)/lens.sum()):.1f}%)')

    def chain_series(ch, key):
        if key in PARAMS:
            return np.array([float(s['params'].get(key, np.nan)) for s in ch])
        if key in ('misfit', 'misfit_tt', 'moi_sigma'):
            return np.array([np.nan if s.get(key) is None else float(s[key]) for s in ch])
        hs = np.array([np.nan if s.get('h_solid_km') is None else float(s['h_solid_km'])
                       for s in ch])
        hl = np.array([np.nan if s.get('h_liquid_km') is None else float(s['h_liquid_km'])
                       for s in ch])
        st = np.where(hl < 1.0, 'solid', np.where(hs < 1.0, 'molten', 'layered'))
        return (st == key.replace('_frac', '')).astype(float)

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

    KEYS = ['layered_frac', 'solid_frac', 'misfit'] + PARAMS_LIVE

    def rhat_table(idx, label):
        sel = [all_chains[i] for i in idx]
        n = min(len(c) for c in sel)
        b = int(BURNIN_FRAC * n)
        print('\n' + '=' * 64)
        print(f'{label}   {len(sel)} chains, n={n}, burn-in={b}, using last {n-b}')
        print('=' * 64)
        print(f"{'param':<16s}{'R-hat':>8s}{'ESS':>9s}{'post-std':>11s}"
              f"{'cur step':>10s}{'suggest':>10s}")
        mats = {}
        for k in KEYS:
            mat = np.array([chain_series(c, k)[-n:][b:] for c in sel])
            mats[k] = mat
            r, e = rhat_ess(mat)
            sd = mat.std()
            cur = f"{CUR_STEP[k]:10.3g}" if k in CUR_STEP else ' ' * 10
            sug = f"{0.5*sd:10.3g}" if k in CUR_STEP else ' ' * 10
            flag = '' if (r == r and r < 1.1) else '  *'
            print(f'{k:<16s}{r:8.3f}{e:9.0f}{sd:11.4f}{cur}{sug}{flag}')
        return mats, n - b

    all_idx = list(range(len(all_chains)))
    alive = [i for i, L in enumerate(lens) if L >= 0.8 * np.median(lens)]
    dead = [chain_names[i] for i in all_idx if i not in alive]

    rhat_table(all_idx, 'A. all chains (truncated to shortest)')
    if dead:
        print(f'\n  short chains excluded: {", ".join(dead)}')
        mats, n_used = rhat_table(alive, 'B. long chains only')
    else:
        mats, n_used = rhat_table(all_idx, 'B. (no short chains to exclude)')
        alive = all_idx

    print('\n' + '=' * 64)
    print('per-chain posterior means  -- few stuck chains, or nothing mixing?')
    print('=' * 64)
    print(f"{'chain':<18s}" + ''.join(f'{k[:9]:>11s}' for k in PARAMS_LIVE))
    for j, i in enumerate(alive):
        print(f'{chain_names[i]:<18s}'
              + ''.join(f'{mats[k][j].mean():11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- pooled --":<18s}'
          + ''.join(f'{mats[k].mean():11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- between --":<18s}' +
          ''.join(f'{mats[k].mean(axis=1).std(ddof=1):11.3f}' for k in PARAMS_LIVE))
    print(f'{"-- within --":<18s}' +
          ''.join(f'{mats[k].std(axis=1).mean():11.3f}' for k in PARAMS_LIVE))
    print('\n  between/within >> 1 => chains have not met yet (run longer)')
    print('  between/within ~  0 => mixed; a bad R-hat comes from something else')

    print('\n' + '=' * 64)
    print(f'chain length needed for ESS = {ESS_TARGET}')
    print('=' * 64)
    for k in KEYS:
        r, e = rhat_ess(mats[k])
        if e == e and e > 0:
            tau = len(alive) * n_used / e
            need = ESS_TARGET * tau / len(alive) / (1 - BURNIN_FRAC)
            print(f'  {k:<16s} tau ~ {tau:7.0f} steps   need ~ {need:7.0f} per chain '
                  f'(now {n_used/(1-BURNIN_FRAC):.0f})')

    print('\n' + '=' * 64)
    print('state transitions per chain')
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
        for j in range(len(alive)):
            y = mats[k][j]
            ax.plot(np.cumsum(y) / np.arange(1, len(y) + 1), lw=1, alpha=0.8)
        ax.axhline(mats[k].mean(), color='k', ls='--', lw=1.5)
        ax.set_title(k, fontsize=fontsize)
        ax.set_xlabel('step (post burn-in)', fontsize=fontsize)
        ax.set_ylabel('running mean', fontsize=fontsize)
        ax.tick_params(labelsize=14)
    fig.suptitle('running mean per chain: should converge onto the dashed line',
                 fontsize=fontsize + 2)
    fig.tight_layout()
    if SAVE: fig.savefig(os.path.join(OUT_DIR, TAG + 'I_running_mean.png'), dpi=150)
    plt.show()