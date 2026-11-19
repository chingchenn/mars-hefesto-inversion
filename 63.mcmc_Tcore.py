#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:02:53 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:02:53 2026
@author: chingchen

合併版 —  a34
  圖E   相圖三態區域圖(固定 P 切面)+ 兩相窗寬度
  圖H2  code 分類 vs 相圖表邊界(逐樣本 P 對齊)+ 誤分類統計
  圖G   固態 BML 傳導熱流,k 掃描 + h_solid 下界  -> 回應 Duncan 2026
  圖G2  molten 態 TBL 熱流(方向與主結論相反,引用前必掛敏感度)

a34 相關修正:
  T_LO/T_HI 與 sample_P 改為從 PRIOR 取值(原本寫死 3200 / 50-400)
  逐樣本查表優先使用 record 的 P_interface,而非由 D 反推
  四態分類(含 no_bml),因 BML_thickness 先驗下界為 0
  讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)

a36 更新:
  * PARAMS 補 w_S / R_cmb;PRIOR 同步 70.mcmc_core_nGibbs 的 runtime 值
    (T_core 下界 1800 -> 1550,新增 w_S、R_cmb)
  * PREFIX 必須是 chain_a36A 或 chain_a36B —— A/B 是兩套核 EoS,不可合併
  * **修正實質 bug**: 圖G 的 BML 頂部半徑原本寫成
        r_top = MARS_RADIUS - (TRUE_CMB_DEPTH - D)
    這在 CMB 深度固定時等價於 R_cmb_fixed + D。a36 起 R_cmb 是自由參數,
    必須逐樣本用 r_top = R_cmb + D,否則 Q_TW(= q x 4 pi r^2)整批算錯。
    a36A 的 R_cmb 中位數 1610 km vs 舊常數 1646 km,面積差約 4%。
  * 圖E 的固定 P 切面改成取最接近 P_interface 後驗中位數的格點,
    而不是表格的正中央(a36 的 P_interface 分布比 a34 寬)
  * MIN_STEPS 提高到 300(a36 有一批 30-90 步的短鏈)
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mars_config import MCMC_DIR, RUN_BASE

# ── 設定 ─────────────────────────────────────────────────────────────────────
CHAIN_DIR   = MCMC_DIR
OUT_DIR     = '/Users/chingchen/Desktop/HeFESTo/mcmc/figures'
PREFIX      = 'chain_a38B'   # 'chain_a36A' 或 'chain_a36B',不要寫 'chain_a36'
# 9x25 / 13x35 / 21x71 三種格點的三態比例差異 < 0.05%,任一皆可
# 注意: 這張表是在 R_cmb 固定 1743.3 km 下建的。a36 把 R_cmb 放開後,
#       表格只用於「一致性檢查」(圖E / H2),不要拿來算先驗比例或 BF。
PHASE_TABLE = (f'{RUN_BASE}/mcmcfigures/'
               'bml_phase_table_prior_a34_13x35.npz')

if PREFIX in ('chain_a36', 'chain_a36_'):
    raise SystemExit(
        "PREFIX='chain_a36' 會同時抓到 a36A 與 a36B。\n"
        "  A = all-Liu (Gamma mixing), B = Liu Fe + Sakai FeS (Margules),\n"
        "  是兩套不同的核 EoS,後驗不可合併。請指定 chain_a36A 或 chain_a36B。")

plt.rcParams['font.family'] = 'DejaVu Sans'

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 300        # a36 有一批 30-90 步的短鏈,門檻拉高避免污染後驗
CHENG_TC    = (2200.0, 2400.0)
SAVE        = False
G_MARS      = 3.72

D_NO_BML = 2.0           # km, D 小於此值視為「沒有 BML」
H_MIN    = 1.0           # km, 三態分類門檻

# a36 先驗,對照 70.mcmc_core_nGibbs.py 的 PRIOR
PRIOR = {
    'T_lit':         (1000.0, 2600.0),
    'P_lit':         (   1.4,    9.0),
    'Mg#':           (  0.50,   0.90),
    'T_core':        (1550.0, 3600.0),   # a26: 下界 1800 -> 1550
    'Mg#_bulk_bml':  (  0.20,   0.80),
    'BML_thickness': (   0.0,  400.0),
    'w_S':           (  0.05,   0.30),   # a35 新增
    'R_cmb':         (1400.0, 1900.0),   # a36 新增
}

# ── 熱流 ─────────────────────────────────────────────────────────────────────
MARS_RADIUS    = 3389.5      # km
# a36 起 R_cmb 是自由參數,CMB 深度逐樣本算。舊鏈沒有 R_cmb 時退回這個常數。
FALLBACK_CMB_DEPTH = 1743.3  # km
K_SOLID_LIST   = [1.0, 1.6, 2.5, 4.0]   # W/m/K; 1.6 = Drilleau 2026 kd 後驗中位
Q_BAND         = (10.0, 60.0)           # mW/m2, Samuel 2021 Fig 5a INTERFACE flux
                                        # (不是 Fig 5b 的 CMB flux -60~+20)
Q_DRILLEAU     = 12.6        # mW/m2, Drilleau 2026 BML 自洽值
                             # kd=1.6, Dd=162 km, Td-Tm = 3051-1778 = 1273 K

# ── molten TBL:依賴 ETA_TBL / RA_C_TBL 兩個寫死、無後驗不確定性的常數,
#    且結論方向與主結果相反。要用就單獨做並掛敏感度,不要跟 layered 的圖並排。
ALPHA_TH, RHO_TH, KAPPA_TH, RA_C_TBL, ETA_TBL = 2e-5, 4000.0, 1e-6, 1000.0, 1e20

# ── 開關 ─────────────────────────────────────────────────────────────────────
fig_E_region      = 1    # 相圖區域圖            (需要 PHASE_TABLE)
code_vs_table     = 1    # code 分類 vs 表邊界   (需要 PHASE_TABLE)
fig_G_heat_flux   = 1
fig_G2_molten_tbl = 1

PARAMS = ['T_lit', 'P_lit', 'Mg#', 'T_core', 'Mg#_bulk_bml', 'BML_thickness',
          'w_S', 'R_cmb']
POST_KEYS = ['misfit', 'misfit_tt', 'h_solid_km', 'h_liquid_km',
             'T_interface', 'T_mantle_bottom', 'P_interface',
             'P_cmb', 'P_bml_top', 'R_ic_km', 'gap_min']   # a35/a36 新增
COLORS = ['#CD5C5C', '#35838D', '#849DAB', '#414F67', '#97795D', '#7B9E87',
          '#9B6B8A', '#4E6E8E', '#C47F3E', '#5C7A5C', '#8B6F6F', '#4A7C7C',
          '#7A6B9B', '#6B8E7A']
STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87',
               'molten': '#CD5C5C', 'no_bml': '#97795D'}
bwith, fontsize = 1.5, 12

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f'{PREFIX}_' if PREFIX else ''


def style(ax):
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    for sp in ['top', 'bottom', 'left', 'right']:
        ax.spines[sp].set_linewidth(bwith)


# ── chain 載入: chain.jsonl(逐行) + chain.json(salvage) ─────────────────────
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

print(f"{'chain':<20s}{'status':<18s}{'n_raw':>8s}   result")
for name, status, n_raw, result in load_report:
    print(f"{name:<20s}{status:<18s}{n_raw:8d}   {result}")
print(f"\nusable chains: {len(all_chains)}")
if not all_chains:
    raise SystemExit(
        f"no usable chains for prefix '{PREFIX}'.\n"
        f"  檢查: (1) {CHAIN_DIR} 下是否有 chain.jsonl\n"
        f"        (2) MIN_STEPS={MIN_STEPS} 是否高於實際鏈長")


# ── 後驗(逐鏈套 burn-in,長度不必相同) ───────────────────────────────────────
samples, chain_id = [], []
for c, ch in enumerate(all_chains):
    b = int(len(ch) * BURNIN_FRAC)
    samples.extend(ch[b:])
    chain_id.extend([c] * (len(ch) - b))
chain_id = np.array(chain_id)
print(f"post burn-in samples: {len(samples)}  from {len(all_chains)} chains")
if not samples:
    raise SystemExit("burn-in 之後沒有樣本,降低 BURNIN_FRAC 或等鏈跑更長")


def col(key):
    return np.array([np.nan if s.get(key) is None else float(s.get(key, np.nan))
                     for s in samples])


# 舊鏈缺 w_S / R_cmb:用 .get 給 NaN,不炸
post = {p: np.array([float(s['params'].get(p, np.nan)) for s in samples])
        for p in PARAMS}
for k in POST_KEYS:
    post[k] = col(k)
post['thermal_state'] = np.array([s.get('thermal_state', '') for s in samples])

_missing = [p for p in PARAMS if not np.isfinite(post[p]).any()]
if _missing:
    print(f"  這批鏈沒有 {_missing}(a36 之前的鏈),相關量會用固定值退回")

# ── a36 幾何: CMB 深度是分布不是常數 ──────────────────────────────────────────
HAS_RCMB = 'R_cmb' not in _missing
if HAS_RCMB:
    post['cmb_depth_km'] = MARS_RADIUS - post['R_cmb']
else:
    post['R_cmb']        = np.full(len(samples), MARS_RADIUS - FALLBACK_CMB_DEPTH)
    post['cmb_depth_km'] = np.full(len(samples), FALLBACK_CMB_DEPTH)


# ── 四態分類 ─────────────────────────────────────────────────────────────────
# a34 的 BML_thickness 先驗下界為 0,所以 D 很小的模型會讓 h_solid 和 h_liquid
# 都小於 1 km 而被誤判成 solid。單獨拉出來當 no_bml,它同時是 Savage-Dickey 的零點。
h_sol, h_liq, D_bml = post['h_solid_km'], post['h_liquid_km'], post['BML_thickness']
state = np.full(len(h_sol), 'fail', dtype='<U8')
state[(h_sol >= H_MIN) & (h_liq >= H_MIN)] = 'layered'
state[(h_sol <  H_MIN) & (h_liq >= H_MIN)] = 'molten'
state[(h_sol >= H_MIN) & (h_liq <  H_MIN)] = 'solid'
state[D_bml < D_NO_BML]                    = 'no_bml'
post['state'] = state

print("\nstate counts:")
for s in ['no_bml', 'solid', 'layered', 'molten', 'fail']:
    m = state == s
    if m.sum():
        print(f"   {s:8s} {m.sum():6d}  {100*m.mean():5.2f}%   "
              f"misfit med={np.nanmedian(post['misfit'][m]):6.2f}  "
              f"min={np.nanmin(post['misfit'][m]):6.2f}  "
              f"tt med={np.nanmedian(post['misfit_tt'][m]):6.2f}")
    else:
        print(f"   {s:8s} {0:6d}")


# ── 相圖表(圖E 與 code_vs_table 共用)─────────────────────────────────────────
tbl = None
if fig_E_region or code_vs_table:
    if not os.path.exists(PHASE_TABLE):
        raise SystemExit(
            f"PHASE_TABLE 不存在: {PHASE_TABLE}\n"
            f"  由 test_ratio_state.py 在 beno3 建表後 rsync 過來,"
            f"或把 fig_E_region / code_vs_table 關掉。")
    z = np.load(PHASE_TABLE)
    tbl = {'P': z['P_grid'], 'Mg': z['Mg_grid'], 'Tsol': z['Tsol'], 'Tliq': z['Tliq']}
    tbl['f_ts'] = RegularGridInterpolator((tbl['P'], tbl['Mg']), tbl['Tsol'],
                                          bounds_error=False)
    tbl['f_tl'] = RegularGridInterpolator((tbl['P'], tbl['Mg']), tbl['Tliq'],
                                          bounds_error=False)
    # 固定 P 切面: 取最接近 P_interface 後驗中位數的格點。
    # 原本用表格正中央 len(P)//2,a36 的 P_interface 分布比 a34 寬,
    # 中央格點不一定代表後驗。
    _Pi = post['P_interface'][np.isfinite(post['P_interface'])]
    if _Pi.size:
        tbl['i'] = int(np.argmin(np.abs(tbl['P'] - np.median(_Pi))))
        print(f"\nfig-E slice: P_interface median = {np.median(_Pi):.2f} GPa "
              f"-> table node {tbl['i']} @ {tbl['P'][tbl['i']]:.2f} GPa "
              f"(舊版固定用中央節點 {len(tbl['P'])//2} @ "
              f"{tbl['P'][len(tbl['P'])//2]:.2f} GPa)")
    else:
        tbl['i'] = len(tbl['P']) // 2
    print(f"\nphase table: P {tbl['P'][0]:.2f}-{tbl['P'][-1]:.2f} GPa "
          f"({len(tbl['P'])} nodes)   Mg {tbl['Mg'][0]:.2f}-{tbl['Mg'][-1]:.2f} "
          f"({len(tbl['Mg'])} nodes)")
    # 表格的 Mg 軸必須對齊當前先驗,否則是舊表
    mg_lo, mg_hi = PRIOR['Mg#_bulk_bml']
    if not (np.isclose(tbl['Mg'][0], mg_lo, atol=1e-6) and
            np.isclose(tbl['Mg'][-1], mg_hi, atol=1e-6)):
        print(f"   !! 表格 Mg 軸與 PRIOR['Mg#_bulk_bml'] {PRIOR['Mg#_bulk_bml']} "
              f"不一致 -> 這是舊表,請重建")


# ── 圖E: 相圖三態區域圖(固定 P 切面)────────────────────────────────────────
if fig_E_region:
    i = tbl['i']
    Mg_grid = tbl['Mg']
    ts_line, tl_line = tbl['Tsol'][i, :], tbl['Tliq'][i, :]
    T_LO, T_HI = PRIOR['T_core']          # 原本寫死 1800/3200,a34 上界是 3600

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.fill_between(Mg_grid, T_LO, ts_line, color='tab:blue', alpha=.30, label='solid')
    ax.fill_between(Mg_grid, ts_line, tl_line, color='tab:green', alpha=.50,
                    label='layered')
    ax.fill_between(Mg_grid, tl_line, T_HI, color='tab:red', alpha=.30, label='molten')
    ax.plot(Mg_grid, ts_line, 'k-', lw=1.6)
    ax.plot(Mg_grid, tl_line, 'k-', lw=1.6)
    ax.axhspan(*CHENG_TC, color='k', alpha=.20, zorder=5)
    ax.text(Mg_grid[1], np.mean(CHENG_TC), 'Cheng 2025 CMB', fontsize=11,
            va='center', zorder=6)
    ax.scatter(post['Mg#_bulk_bml'], post['T_core'], s=3, c='k', alpha=.10, zorder=4)
    ax.set_xlim(Mg_grid[0], Mg_grid[-1]); ax.set_ylim(T_LO, T_HI)
    ax.set_xlabel('Mg# bulk BML', fontsize=fontsize)
    ax.set_ylabel('T_core (K)', fontsize=fontsize)
    ax.set_title(f"state regions at P = {tbl['P'][i]:.2f} GPa "
                 f"(boundaries shift with P; scatter spans all P)", fontsize=fontsize)
    style(ax); ax.legend(fontsize=11, loc='lower right')

    w_win = tl_line - ts_line
    print(f"\ntwo-phase window: min={np.nanmin(w_win):.0f} K  "
          f"max={np.nanmax(w_win):.0f} K  mean={np.nanmean(w_win):.0f} K  "
          f"({100*np.nanmean(w_win)/(T_HI-T_LO):.1f}% of T_core prior range)")

    plt.tight_layout()
    if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + 'E_region_map.png'), dpi=150)
    plt.show()


# ── 圖H2: code 分類 vs 相圖表邊界(逐樣本 P 對齊)──────────────────────────────
if code_vs_table:
    P_lo_t, P_hi_t = tbl['P'][0] + 0.05, tbl['P'][-1] - 0.05
    D_lo, D_hi = PRIOR['BML_thickness']

    def sample_P_from_D(D_km):
        # D -> 層中點壓力的線性近似(D 大 -> 中點淺 -> P 低)
        # 原本寫死 50/400,a34 的下界是 0
        return P_hi_t + (P_lo_t - P_hi_t) * (D_km - D_lo) / (D_hi - D_lo)

    # T_interface 是在 P_interface 求出來的,所以查表應該用 P_interface;
    # 沒有這個欄位時才退回由 D 反推。
    P_rec = post['P_interface']
    P_all = np.where(np.isfinite(P_rec), P_rec, sample_P_from_D(D_bml))
    n_rec = int(np.isfinite(P_rec).sum())
    print(f"\ntable lookup pressure: {n_rec} samples from record P_interface, "
          f"{len(P_all)-n_rec} from D-based linear map")

    ts_all = tbl['f_ts'](np.column_stack([P_all, post['Mg#_bulk_bml']]))
    tl_all = tbl['f_tl'](np.column_stack([P_all, post['Mg#_bulk_bml']]))
    n_oob = int(np.sum(~np.isfinite(ts_all) | ~np.isfinite(tl_all)))
    if n_oob:
        print(f"   {n_oob} samples ({100*n_oob/len(P_all):.1f}%) fall outside the "
              f"table grid -> NaN, excluded from the check below")

    denom = tl_all - ts_all
    xrel  = np.where(np.abs(denom) > 1e-9,
                     (post['T_interface'] - ts_all) / denom, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    for s in ['molten', 'layered', 'solid', 'no_bml']:
        m = (state == s)
        if m.sum():
            ax.scatter(post['Mg#_bulk_bml'][m], xrel[m], s=4, alpha=.3,
                       color=STATE_COLOR[s], label=f'{s} (code)')
    ax.axhline(0, color='k', lw=2, label='solidus (per-sample P)')
    ax.axhline(1, color='k', lw=2, ls='--', label='liquidus (per-sample P)')
    ax.set_xlabel('Mg# bulk BML', fontsize=fontsize)
    ax.set_ylabel('(T_interface - T_sol) / (T_liq - T_sol)', fontsize=fontsize)
    ax.set_title('code state vs table boundaries (P matched per sample)',
                 fontsize=fontsize)
    style(ax); ax.legend(fontsize=10)
    plt.tight_layout()
    if SAVE:
        plt.savefig(os.path.join(OUT_DIR, TAG + 'H2_code_vs_table_Pmatched.png'),
                    dpi=150)
    plt.show()

    # 逐樣本量化驗證:solidus=0、liquidus=1 都是該樣本自己的邊界
    # no_bml 不參與(層太薄,界面位置沒有意義)
    print(f"\n{'state':<10s}{'n':>7s}{'n_finite':>10s}{'misclassified':>15s}   expected")
    for s, expect in [('layered', '0 < xrel < 1'),
                      ('molten',  'xrel >= 1'),
                      ('solid',   'xrel <= 0')]:
        m = (state == s)
        if m.sum() == 0:
            print(f'{s:<10s}{0:7d}')
            continue
        xr = xrel[m]
        fin = np.isfinite(xr)
        if fin.sum() == 0:
            print(f'{s:<10s}{int(m.sum()):7d}{0:10d}{"--":>15s}   {expect}')
            continue
        v = xr[fin]
        if s == 'layered':
            bad = np.mean((v <= 0) | (v >= 1))
        elif s == 'molten':
            bad = np.mean(v < 1)
        else:
            bad = np.mean(v > 0)
        print(f'{s:<10s}{int(m.sum()):7d}{int(fin.sum()):10d}'
              f'{100*bad:14.2f}%   {expect}')
    print('   誤分類率高不一定是 bug:表格用的是層中點 P,code 用的是界面 P,')
    print('   兩者在厚層時差異可達數 GPa。這張圖是一致性檢查,不是真值比對。')


# ── 圖G: 固態 BML 傳導熱流,k 掃描 + h_solid 下界 ────────────────────────────
if fig_G_heat_flux:
    sel = ((state == 'layered') & np.isfinite(post['T_interface'])
           & np.isfinite(post['T_mantle_bottom']) & (h_sol >= H_MIN))
    if sel.sum() == 0:
        print('\n[圖G] 沒有分層態樣本,跳過固態 BML 熱流')
    else:
        dT       = post['T_interface'][sel] - post['T_mantle_bottom'][sel]
        h_m      = h_sol[sel] * 1000.0
        # BML 頂部半徑。幾何: z_cmb = R_mars - R_cmb, z_bml_top = z_cmb - D
        #   => r_top = R_mars - z_bml_top = R_cmb + D
        # 舊版寫 MARS_RADIUS - (TRUE_CMB_DEPTH - D),在 CMB 固定時等價,
        # 但 a36 的 R_cmb 是自由參數,必須逐樣本取。
        r_top_km = post['R_cmb'][sel] + D_bml[sel]

        print(f"\nsolid-BML conductive heat flux   n={int(sel.sum())}")
        print(f"   dT median={np.nanmedian(dT):.0f} K   "
              f"h_solid median={np.nanmedian(h_sol[sel]):.1f} km")
        _r_old = MARS_RADIUS - (FALLBACK_CMB_DEPTH - D_bml[sel])
        print(f"   r_top median={np.nanmedian(r_top_km):.1f} km "
              f"(舊常數 CMB 寫法會給 {np.nanmedian(_r_old):.1f} km, "
              f"面積差 {100*abs(np.nanmedian(r_top_km)**2/np.nanmedian(_r_old)**2 - 1):.1f}%)")
        n_neg = int(np.sum(dT <= 0))
        if n_neg:
            print(f"   警告: {n_neg} 個樣本 dT <= 0(界面比地函底部冷),"
                  f"其正熱流無定義")
        print(f"   對照帶 = Samuel 2021 Fig 5a interface flux "
              f"{Q_BAND[0]:.0f}-{Q_BAND[1]:.0f} mW/m2")
        print(f"\n{'k':>6}{'q_med':>9}{'q05':>8}{'q95':>10}{'Q_TW':>8}"
              f"{'in band':>10}{'above':>8}{'h_min_med':>11}")

        BINS = np.logspace(np.log10(2), np.log10(3000), 50)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for mm, K_SOLID in enumerate(K_SOLID_LIST):
            q_mW = K_SOLID * dT / h_m * 1000.0
            Q_TW = q_mW * 1e-3 * 4 * np.pi * (r_top_km * 1e3)**2 / 1e12
            ok   = np.isfinite(q_mW) & (q_mW > 0)
            if ok.sum() == 0:
                print(f'{K_SOLID:6.1f}      (no positive flux)')
                continue
            q, Q = q_mW[ok], Q_TW[ok]

            q5, q50, q95 = np.percentile(q, [5, 50, 95])
            in_band = ((q >= Q_BAND[0]) & (q <= Q_BAND[1])).mean()
            above   = (q > Q_BAND[1]).mean()

            # 穩態能量收支要求的固態次層厚度下界: q <= Q_BAND[1]
            h_min = K_SOLID * dT[ok] / (Q_BAND[1] * 1e-3) / 1e3      # km

            print(f"{K_SOLID:6.1f}{q50:9.1f}{q5:8.1f}{q95:10.1f}"
                  f"{np.median(Q):8.2f}{100*in_band:9.1f}%{100*above:7.1f}%"
                  f"{np.median(h_min):11.1f}")

            axes[0].hist(q, bins=BINS, color=COLORS[mm], alpha=0.55,
                         edgecolor='none',
                         label=f'k = {K_SOLID:.1f}   med {q50:.0f}   '
                               f'{100*in_band:.0f}% in band')
            axes[0].axvline(q50, color=COLORS[mm], lw=2.0)

        axes[0].axvspan(*Q_BAND, color='k', alpha=0.10, zorder=0)
        axes[0].axvline(Q_DRILLEAU, color='#C47F3E', ls='--', lw=2.0,
                        label=f'Drilleau 2026 BML  {Q_DRILLEAU:.1f}')
        axes[0].set_xscale('log')
        axes[0].set_xlabel(r'$q = k\,\Delta T / h_{solid}$  (mW/m$^2$)',
                           fontsize=fontsize)
        axes[0].set_ylabel('count', fontsize=fontsize)
        axes[0].set_title(f'layered n={int(sel.sum())}', fontsize=fontsize)
        axes[0].legend(fontsize=9, loc='upper right')
        style(axes[0])

        # 右圖:散布由 h_solid 主導,不是 k
        q_ref = 1.6 * dT / h_m * 1000.0
        vmax = np.nanpercentile(q_ref, 95)
        sc = axes[1].scatter(h_sol[sel], dT, c=q_ref, cmap='magma', s=8,
                             norm=plt.Normalize(0, vmax if vmax > 0 else 1.0))
        cb = plt.colorbar(sc, ax=axes[1])
        cb.set_label('q at k=1.6 (mW/m$^2$)', fontsize=fontsize)
        cb.ax.tick_params(labelsize=14)
        axes[1].set_xlabel('h_solid (km)', fontsize=fontsize)
        axes[1].set_ylabel('T_interface - T_mantle_bottom (K)', fontsize=fontsize)
        axes[1].set_title('spread is driven by h_solid, not k', fontsize=fontsize)
        style(axes[1])

        plt.tight_layout()
        if SAVE: plt.savefig(os.path.join(OUT_DIR, TAG + 'G_heat_flux.png'), dpi=150)
        plt.show()

        print("\n判讀:k 只在 log 軸上平移分布,5-95% 跨度遠大於 k 的 4 倍變化")
        print("     -> q 的不確定度主要來自 h_solid。結論應寫成「能量收支對 h_solid")
        print("        設下界」,而不是「排除了多少百分比的樣本」。")


# ── 圖G2: molten 態 TBL 熱流 ─────────────────────────────────────────────────
if fig_G2_molten_tbl:
    selm = ((state == 'molten') & np.isfinite(post['T_interface'])
            & np.isfinite(post['T_mantle_bottom']))
    if selm.sum() == 0:
        print('\n[圖G2] 沒有 molten 態樣本,跳過')
    else:
        dTm     = post['T_interface'][selm] - post['T_mantle_bottom'][selm]
        dTm_pos = np.maximum(dTm, 1.0)
        delta_m = (RA_C_TBL * KAPPA_TH * ETA_TBL
                   / (ALPHA_TH * RHO_TH * G_MARS * dTm_pos))**(1/3)     # m
        qm      = 1.6 * dTm_pos / delta_m * 1000.0                      # mW/m2

        print(f"\nmolten-state TBL heat flux (n={int(selm.sum())}, "
              f"eta={ETA_TBL:.0e}, Ra_c={RA_C_TBL:.0f})")
        print(f"   delta median={np.nanmedian(delta_m)/1e3:.1f} km")
        print(f"   q     median={np.nanmedian(qm):.1f} mW/m2  "
              f"5/95% = {np.nanpercentile(qm,5):.1f} / {np.nanpercentile(qm,95):.1f}")
        print("   警告:delta 由 ETA_TBL 與 RA_C_TBL 反推,兩者無後驗不確定性;")
        print("        delta ~ dT^(-1/3) -> q ~ dT^(4/3),加厚 BML 救不了。")
        print("        此結果方向與主結論相反,引用前務必掛敏感度。")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(qm[np.isfinite(qm)], bins=40, color=STATE_COLOR['molten'])
        axes[0].axvspan(*Q_BAND, color='k', alpha=.10)
        axes[0].set_xlabel('implied flux, molten BML (mW/m$^2$)', fontsize=fontsize)
        axes[0].set_ylabel('count', fontsize=fontsize)
        style(axes[0])

        sc = axes[1].scatter(post['T_core'][selm], qm, c=D_bml[selm],
                             cmap='magma', s=8)
        cb = plt.colorbar(sc, ax=axes[1])
        cb.set_label('BML thickness (km)', fontsize=fontsize)
        cb.ax.tick_params(labelsize=14)
        axes[1].axhspan(*Q_BAND, color='k', alpha=.10)
        axes[1].set_xlabel('T_core (K)', fontsize=fontsize)
        axes[1].set_ylabel('q (mW/m$^2$)', fontsize=fontsize)
        axes[1].set_yscale('log')
        style(axes[1])

        plt.tight_layout()
        if SAVE:
            plt.savefig(os.path.join(OUT_DIR, TAG + 'G2_heat_flux_molten.png'), dpi=150)
        plt.show()