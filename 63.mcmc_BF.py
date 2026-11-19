#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:44:18 2026

@author: chingchen

三態後驗比例 + Bayes factor。
讀檔支援 chain.json(a31-a33)與 chain.jsonl(a34 起)。

a36 更新:
  * 參數 8 個(a36 起 R_cmb 為自由參數,a35 起有 w_S)
  * PREFIX 必須指定 chain_a36A 或 chain_a36B —— A/B 是兩套不同的核 EoS
    (A = all-Liu Gamma mixing, B = Liu Fe + Sakai FeS Margules),
    寫 'chain_a36' 會把兩套模型的後驗混在一起
  * CMB 深度不再是常數 1743.3 km,改成逐樣本 MARS_RADIUS - R_cmb
  * PRIOR_FRAC 仍是 a34 的 6 維表 -> 標記為 stale,BF 只當相對指標
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from mars_config import MCMC_DIR, FIG_DIR

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR   = MCMC_DIR
OUT_DIR     = FIG_DIR
PREFIX      = 'chain_a52B'      # 'chain_a36A' 或 'chain_a36B',不要寫 'chain_a36'

if PREFIX.startswith('chain_a36') and PREFIX in ('chain_a36', 'chain_a36_'):
    raise SystemExit(
        "PREFIX='chain_a36' 會同時抓到 a36A 與 a36B。\n"
        "  A = all-Liu (Gamma mixing), B = Liu Fe + Sakai FeS (Margules),\n"
        "  是兩套不同的核 EoS,後驗不可合併。請指定 chain_a36A 或 chain_a36B。")

plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 300         # a36 有一批 30-90 步的短鏈,門檻拉高避免污染後驗
RA_C        = 1.0e5
CHENG_TC    = (2200.0, 2400.0)
SAVE_FIG    = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
SEC_YR, G_MARS = 3.156e7, 3.72

RHO_S_CONST = 4.097      # g/cc, nGibbs at posterior XS, [3.98, 4.24]

# 先驗三態比例(由 test_ratio_state.py 的相圖表 + N=200k 抽樣算出)
#PRIOR_FRAC = {'solid': 0.284, 'layered': 0.132, 'molten': 0.584}  # a31 先驗
PRIOR_FRAC  = {'solid': 0.214, 'layered': 0.096, 'molten': 0.690}  # a34 先驗, 21x71
# a36 把 R_cmb 放開(1400-1900 km),先驗的三態體積跟著改變,而這張表是在
# R_cmb 固定 1743.3 km 下抽出來的 -> 分母過期。BF 仍會算,但只當相對指標,
# 重算 bml_phase_table 之前不要引用絕對值。
PRIOR_FRAC_IS_STALE = True

CUR_STEP = {'T_lit': 50.0, 'P_lit': 0.7, 'Mg#': 0.025,
            'T_core': 100.0, 'Mg#_bulk_bml': 0.06, 'BML_thickness': 25.0,
            'w_S': 0.02, 'R_cmb': 25.0}
ESS_TARGET = 400

# a36 先驗,對照 70.mcmc_core_nGibbs.py 的 PRIOR
PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (   0.0,  400.0),
    'w_S':           (  0.05,   0.30),
    'R_cmb':         (1400.0, 1900.0),
}
UNITS = {'T_lit':'K', 'P_lit':'GPa', 'Mg#':'', 'T_core':'K',
         'Mg#_bulk_bml':'', 'BML_thickness':'km', 'w_S':'', 'R_cmb':'km'}

A_GRAIN, C_KC, MU_MELT, ETA_MTX, N_RZ = 1e-3, 1000.0, 1.0, 1e19, 4.65
RCMF = (0.25, 0.40)

# molten 態的 TBL（沒有固態次層可用，改由邊際穩定估厚度）
ALPHA_TH, RHO_TH, KAPPA_TH, RA_C_TBL, ETA_TBL = 2e-5, 4000.0, 1e-6, 1000.0, 1e20
Q_CMB_MAX = 40.0        # mW/m2, 穩態上限（Q_CMB_RANGE 是文獻估計 5-20）

# 熱流檢查用
MARS_RADIUS     = 3389.5     # km
# a36 起 R_cmb 是自由參數,CMB 深度逐樣本算(見下方 post['cmb_depth_km'])。
# 舊鏈沒有 R_cmb 時退回這個常數。
FALLBACK_CMB_DEPTH = 1743.3  # km
K_SOLID         = 4.0        # W/m/K
Q_CMB_RANGE     = (5.0, 20.0)   # mW/m2, 文獻估計

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']
LABELS = ['T_lit (K)', 'P_lit (GPa)', 'Mg# (mantle)', 'T_core (K)',
          'Mg#_bulk_bml', 'BML thickness (km)', 'w_S (wt frac)', 'R_cmb (km)']
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
bwith, fontsize = 1.5, 12

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''


# ── chain 載入: chain.json(salvage) + chain.jsonl(逐行) ─────────────────────
def load_chain_json(path):
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
    """逐行讀;只有最後一行可能寫到一半,壞掉就丟棄該行"""
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
    """優先 chain.jsonl(a34+),沒有才退回 chain.json(a31-a33)"""
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(os.path.join(chain_dir, 'chain.json'))


chain_dirs = sorted(glob.glob(os.path.join(CHAIN_DIR, f'{PREFIX}*')))
all_chains, chain_names, load_report = [], [], []

for dd in chain_dirs:
    name = os.path.basename(dd)
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

# ── 載入報告 + 失敗守衛(原版刪掉了這段,導致 0 條鏈時靜默往下跑)───────────
print(f"{'chain':<20s} {'status':<18s} {'n_raw':>7s}  action")
for name, status, n_raw, action in load_report:
    print(f"{name:<20s} {status:<18s} {n_raw:>7d}  {action}")
print(f"\nusable chains: {len(all_chains)}")

if not all_chains:
    raise SystemExit(
        f"no usable chains for prefix '{PREFIX}'.\n"
        f"  檢查: (1) {CHAIN_DIR} 下是否有 chain.json 或 chain.jsonl\n"
        f"        (2) MIN_STEPS={MIN_STEPS} 是否高於實際鏈長")
if len(all_chains) < 2:
    raise SystemExit(
        f"只有 {len(all_chains)} 條鏈。ESS/R-hat 用鏈間變異估計,至少需要 2 條。")


# ── 後驗(逐鏈套 burn-in,長度不必相同) ───────────────────────────────────────
samples, chain_id = [], []
for c, ch in enumerate(all_chains):
    b = int(len(ch) * BURNIN_FRAC)
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"post burn-in samples: {len(samples)}")
if len(samples) == 0:
    raise SystemExit("burn-in 之後沒有樣本,降低 BURNIN_FRAC 或等鏈跑更長")


def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])


# 舊鏈缺 w_S / R_cmb:用 .get 給 NaN,不炸
post = {p: np.array([float(s['params'].get(p, np.nan)) for s in samples])
        for p in PARAMS}
_missing = [p for p in PARAMS if not np.isfinite(post[p]).any()]
if _missing:
    print(f"  這批鏈沒有 {_missing} 欄位(a36 之前的鏈),相關診斷會跳過")
PARAMS_LIVE = [p for p in PARAMS if p not in _missing]

for k in ['misfit', 'misfit_tt', 'misfit_solidus', 'mass_sigma', 'moi_sigma',
          'upper_contrast', 'lower_contrast', 'Ra', 'h_solid_km', 'h_liquid_km',
          'Mg_solid', 'Mg_liquid', 'T_interface', 'T_mantle_bottom', 'moi_pred',
          'tt_n_ph', 'tt_n_miss',
          # a35+ 核診斷
          'R_ic_km', 'ric_sigma', 'rho_core_mean', 'Vp_core_cmb',
          'P_center_core', 'T_center_core',
          # a36 自洽結構診斷
          'P_cmb', 'P_bml_top', 'z_lit_km', 'gap_min', 'bml_n_pass',
          'sc_n_iter', 'sc_dP', 'sc_dz_lit']:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

# CMB 深度: a36 逐樣本由 R_cmb 決定;舊鏈退回常數
if 'R_cmb' in PARAMS_LIVE:
    post['cmb_depth_km'] = MARS_RADIUS - post['R_cmb']
else:
    post['cmb_depth_km'] = np.full(len(samples), FALLBACK_CMB_DEPTH)
# BML 頂部半徑 / 深度(取代舊的 TRUE_CMB_DEPTH - D_bml 寫法)
post['bml_top_depth_km'] = post['cmb_depth_km'] - post['BML_thickness']
post['bml_top_r_km']     = MARS_RADIUS - post['bml_top_depth_km']

h_sol, h_liq = post['h_solid_km'], post['h_liquid_km']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= 1.0) & (h_liq >= 1.0)] = 'layered'
state[(h_sol <  1.0) & (h_liq >= 1.0)] = 'molten'
state[(h_sol >= 1.0) & (h_liq <  1.0)] = 'solid'
post['state'] = state
f_solid = h_sol / np.maximum(h_sol + h_liq, 1e-9)
print("state counts:", {s: int((state == s).sum())
                        for s in ['solid', 'layered', 'molten', 'fail']})

# ── a36 幾何摘要: R_cmb 自由後,CMB 深度是分布不是常數 ────────────────────────
def _q(v, name, unit=''):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f'  {name:<18s} --');  return
    lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
    print(f'  {name:<18s} {md:9.2f}  [{lo:8.2f}, {hi:8.2f}] {unit}')

print('\ngeometry / core (median [95% CI]):')
_q(post['R_cmb'],            'R_cmb',        'km')
_q(post['cmb_depth_km'],     'CMB depth',    'km')
_q(post['bml_top_depth_km'], 'BML top depth','km')
_q(post['w_S'],              'w_S',          'wt frac')
_q(post['R_ic_km'],          'R_ic',         'km')
_q(post['gap_min'],          'gap_min',      'km')
if np.isfinite(post['gap_min']).any():
    _bad = np.nanmean(post['gap_min'] < 0) * 100
    print(f'  gap_min < 0 的樣本: {_bad:.1f}%  (內核幾乎撐滿液核)')

ok = state != 'fail'
n_ok = int(ok.sum())
if n_ok == 0:
    raise SystemExit(
        "所有樣本都是 'fail'(h_solid 與 h_liquid 都 < 1 km)。\n"
        "  a36 的 BML_thickness 先驗下界為 0,極薄層樣本會落入這類;\n"
        "  若比例過高,需檢視分類門檻是否適用。")

n_fail = len(state) - n_ok
if n_fail > 0:
    print(f"\n  警告: {n_fail} 個樣本 ({100*n_fail/len(state):.2f}%) 分類為 'fail'。")
    print(f"  a36 的 BML_thickness 下界為 0,而分類用 1 km 門檻,極薄層會落入此類;")
    print(f"  但 PRIOR_FRAC 來自純溫度分類(無厚度門檻)→ BF 分子分母慣例不完全一致。")


# ── 三態比例 + ESS/R-hat(鏈間變異法)────────────────────────────────────────
def _state_frac_per_chain(name):
    m = len(all_chains)
    p = np.array([(state[chain_id == c] == name).mean() for c in range(m)])
    n = np.mean([(chain_id == c).sum() for c in range(m)])
    return p, n, m


def ess_state(name):
    """由鏈間變異估狀態比例的 ESS（Gelman-Rubin）。
    稀有二元事件不能用鏈內自相關，會嚴重高估。"""
    p, n, m = _state_frac_per_chain(name)
    if m < 2:
        return np.nan
    W, Bn = np.mean(p * (1 - p)), p.var(ddof=1)
    if W <= 0 or Bn <= 0:
        return np.nan
    return float(m * ((n - 1) / n * W + Bn) / Bn)


def rhat_state(name):
    p, n, m = _state_frac_per_chain(name)
    if m < 2:
        return np.nan
    W, Bn = np.mean(p * (1 - p)), p.var(ddof=1)
    if W <= 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + Bn) / W))


STL = ['solid', 'layered', 'molten']
pst = {s: (state == s).sum() / n_ok for s in STL}

ESS_S = {s: ess_state(s) for s in STL}
RH_S  = {s: rhat_state(s) for s in STL}

_finite = [v for v in ESS_S.values() if np.isfinite(v)]
if not _finite:
    raise SystemExit(
        "三個狀態的 ESS 都無法估計(鏈間變異為零)。\n"
        "  常見原因: 所有鏈都停在同一狀態,或鏈太短還沒發生任何狀態轉換。")
floor = float(np.nanmin(_finite))
for s in STL:                       # 全零狀態沒有鏈間變異 -> 借最小的
    if not np.isfinite(ESS_S[s]):
        ESS_S[s] = floor

print(f"\n{'state':<10s}{'count':>8s}{'prior':>9s}{'post':>9s}"
      f"{'R-hat':>8s}{'ESS':>8s}   95% CI")
for s in STL:
    p, e, c = pst[s], ESS_S[s], int((state == s).sum())
    if c == 0:
        ci = f'[0, {3/e:.2e}]  (rule of 3)'
    else:
        se = np.sqrt(p * (1 - p) / e)
        ci = f'[{max(p-1.96*se,0):.4f}, {min(p+1.96*se,1):.4f}]'
    rh = RH_S[s]
    rh_str = f'{rh:8.3f}' if np.isfinite(rh) else f"{'--':>8s}"
    print(f'{s:<10s}{c:8d}{PRIOR_FRAC[s]:9.3f}{p:9.4f}{rh_str}{e:8.0f}   {ci}')
print(f'  ESS 由鏈間變異估（raw n = {n_ok}，差 {n_ok/floor:.0f} 倍）')
if PRIOR_FRAC_IS_STALE:
    print('  警告: PRIOR_FRAC 標記為過期,BF 不可引用')


# ── Bayes factor ─────────────────────────────────────────────────────────────
JEFF = [(1, 3, 'barely'), (3, 10, 'substantial'), (10, 30, 'strong'),
        (30, 100, 'very strong'), (100, np.inf, 'decisive')]


def _lab(b):
    if not np.isfinite(b) or b <= 0:
        return '-'
    b = b if b >= 1 else 1 / b
    for lo, hi, t in JEFF:
        if lo <= b < hi:
            return t
    return '-'


print()
BF = {}
for a, b in [('molten', 'solid'), ('molten', 'layered'), ('layered', 'solid')]:
    ca, cb = int((state == a).sum()), int((state == b).sum())
    pa = 3 / ESS_S[a] if ca == 0 else pst[a]
    pb = 3 / ESS_S[b] if cb == 0 else pst[b]
    bf = (pa / pb) / (PRIOR_FRAC[a] / PRIOR_FRAC[b])
    bnd = (ca == 0) or (cb == 0)
    BF[(a, b)] = (bf, bnd)
    print(f'  BF({a:>7s} : {b:<7s}) {">" if bnd else "="} {bf:10.1f}   '
          f'log10 = {np.log10(bf):+6.2f}   {_lab(bf)}'
          + ('   [下界，零計數]' if bnd else ''))


# ── Cheng 2025 reweighting ───────────────────────────────────────────────────
w_cheng = ((post['T_core'] >= CHENG_TC[0]) &
           (post['T_core'] <= CHENG_TC[1])).astype(float)
has_cheng = w_cheng.sum() > 0

if has_cheng:
    print(f'\nreweighted to Cheng T_core {CHENG_TC}   '
          f'ESS = {w_cheng.sum()**2/(w_cheng**2).sum():.0f}')
    for s in STL:
        print(f'   {s:<8s} {100*w_cheng[state==s].sum()/w_cheng.sum():5.1f}%')
else:
    print(f'\n沒有樣本落在 Cheng T_core {CHENG_TC} 區間 -> 跳過 reweighting')


# ── 圖B: prior -> posterior + Bayes factor ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
x, w = np.arange(3), 0.36
err = [3/ESS_S[s] if (state == s).sum() == 0
       else 1.96*np.sqrt(pst[s]*(1-pst[s])/ESS_S[s]) for s in STL]
ax.bar(x - w/2, [PRIOR_FRAC[s] for s in STL], w, color='#B0A99F', label='prior')
ax.bar(x + w/2, [pst[s] for s in STL], w, yerr=err, capsize=5,
       color=COLORS[3], label='posterior')
for i, s in enumerate(STL):
    if (state == s).sum() == 0:
        ax.text(x[i]+w/2, .05, f'0 / {ESS_S[s]:.0f} ESS\n< {3/ESS_S[s]:.1e}',
                ha='center', fontsize=10, color=COLORS[0])
    else:
        ax.text(x[i]+w/2, pst[s]+err[i]+.02, f'{pst[s]:.3f}',
                ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(STL, fontsize=fontsize)
ax.set_ylabel('probability', fontsize=fontsize); ax.set_ylim(0, 1.15)
ax.set_title('prior vs posterior (95% CI from ESS)', fontsize=fontsize)

ax = axes[1]
keys = list(BF.keys()); vals = [np.log10(BF[k][0]) for k in keys]
ax.barh(range(len(keys)), vals, color=COLORS[8], height=.55)
for v in [3, 10, 30, 100]:
    ax.axvline(np.log10(v), color='k', lw=.8, ls=':')
ax.axvline(0, color='k', lw=1.5)
for i, k in enumerate(keys):
    bf, bd = BF[k]
    ax.text(vals[i]+.06, i, f'{">" if bd else "="} {bf:,.0f}  {_lab(bf)}',
            va='center', fontsize=fontsize-1)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([f'{a} : {b}' for a, b in keys], fontsize=fontsize)
ax.invert_yaxis()
# BF < 1 時 log10 為負,原版 set_xlim(0, max*1.9) 會反轉軸 -> 用對稱邊界
_lo, _hi = min(vals + [0.0]), max(vals + [0.0])
_pad = max(0.1, 0.25 * (_hi - _lo))
ax.set_xlim(_lo - _pad, _hi + _pad * 2.5)
ax.set_xlabel('log10 Bayes factor  (dotted = 3/10/30/100)', fontsize=fontsize)
ax.set_title('how much the DATA moved the odds', fontsize=fontsize)
for a_ in axes:
    a_.tick_params(labelsize=14)
    # 右圖(BF barh)沒有 labelled artist,無條件 legend() 會噴 UserWarning
    if a_.get_legend_handles_labels()[1]:
        a_.legend(fontsize=fontsize-1)
    for sp in ['top', 'bottom', 'left', 'right']:
        a_.spines[sp].set_linewidth(bwith)
plt.tight_layout()
if SAVE_FIG: plt.savefig(os.path.join(OUT_DIR, TAG + 'B_bayes.png'), dpi=150)
plt.show()


# ── Cheng-reweighted BF ──────────────────────────────────────────────────────
if has_cheng:
    pst_w = {s: w_cheng[state == s].sum() / w_cheng.sum() for s in STL}
    print(f"\nCheng-reweighted posterior: solid={pst_w['solid']:.3f}  "
          f"layered={pst_w['layered']:.3f}  molten={pst_w['molten']:.3f}")

    bf_ml_w = None
    if pst_w['layered'] > 0:
        bf_ml_w = ((pst_w['molten'] / pst_w['layered'])
                   / (PRIOR_FRAC['molten'] / PRIOR_FRAC['layered']))
        print(f"Cheng-reweighted BF(molten:layered) = {bf_ml_w:.2f}  ({_lab(bf_ml_w)})")
        print(f"  (unweighted was {BF[('molten','layered')][0]:.2f}  "
              f"{_lab(BF[('molten','layered')][0])})")
    else:
        print("Cheng 區間內沒有 layered 樣本 -> BF(molten:layered) 無法計算")

    if pst_w['solid'] > 0:
        bf_ms_w = ((pst_w['molten'] / pst_w['solid'])
                   / (PRIOR_FRAC['molten'] / PRIOR_FRAC['solid']))
        print(f"Cheng-reweighted BF(molten:solid)   = {bf_ms_w:.2f}  ({_lab(bf_ms_w)})")
    else:
        print("Cheng 區間內沒有 solid 樣本 -> BF(molten:solid) 無法計算")

    if bf_ml_w is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ['unweighted', 'Cheng-reweighted']
        vals = [BF[('molten', 'layered')][0], bf_ml_w]
        bars = ax.bar(labels, vals, color=[COLORS[8], COLORS[3]])
        for v in [3, 10]:
            ax.axhline(v, color='gray', ls=':', lw=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.2f}',
                    ha='center', fontsize=12)
        ax.set_ylabel('BF(molten : layered)', fontsize=14)
        ax.tick_params(labelsize=14)
        plt.tight_layout(); plt.show()