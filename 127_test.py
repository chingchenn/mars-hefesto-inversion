#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 127: Fo-Fa 相圖 -- 後驗的界面點落在相界的哪一側
#
# 在 beno3 上跑,因為要 import 70.mcmc_core_nGibbs 取得 bml_phase_diagram,
# 而該模組的相依(相圖表、core_fes)都在 server 上。
#
#   P2  127_phase_TP.png   solidus / liquidus vs P,固定數個 Mg#,
#                          疊上後驗的界面點 (T_int, P_int),依三態上色
#   P1  127_phase_TX.png   T-X 二元 loop(預設關閉,見下)
#
# P1 預設關閉:投影片上已經有一張更好的 Fo-Fa 等壓截面(含 tie line、
#   XS / XL 標記、三態方框)。這裡不重複做。要對照時把 TOGGLE['TX'] 開起來。
#
# P2 做的是不同的事:那張等壓截面說「給定 P 與 bulk 組成,槓桿定則決定三態」,
#   P2 說「後驗實際落在哪裡」。一個是機制,一個是結果。
#
# ⚠️ bml_phase_diagram 的 T 引數只用來判斷狀態;solidus / liquidus 本身
#   只是 (P, X) 的函數,所以掃描時 T 給定值即可。
# ⚠️ 相圖表綁定先驗的 Mg 軸(bml_phase_table_prior_*.npz)。改先驗必須重建。
# ⚠️ 70.mcmc_core_nGibbs.py 的模組名不合法(開頭數字 + 點),
#   不能用 import_module,要用 spec_from_file_location 吃路徑。

import os, sys, json, glob
import importlib.util
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# 路徑:自動判斷跑在 beno3 還是本機 Mac
# 第一個存在的候選就採用,解析結果會 print 出來。
# 猜錯就直接改 CHAIN_CANDS / PHASE_CANDS 的第一項。
# ══════════════════════════════════════════════════════════════════════════
CHAIN_CANDS = [
    '/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc',
    '/Users/chingchen/Desktop/HeFESTo/mcmc/mcmc_chain',
]
PHASE_CANDS = [
    os.path.expanduser('~/mars-hefesto-inversion'),
    '/net/beno3/data1/jcchen2/mars-hefesto-inversion',
    '/net/beno3/data1/jcchen2/mars-hefesto-runs',
    '/Users/chingchen/Desktop/HeFESTo/python',
]
PHASE_GLOB = '70*mcmc_core_nGibbs.py'      # 同時吃 '70.' 與 '70_' 兩種命名
OUT_DIR    = '.'                            # 圖存在當前目錄

PREFIX  = 'chain_a52B'
EXCLUDE = ['chain_a52B_04']

BURNIN_FRAC = 0.3
DROP_LAST   = 1
MIN_STEPS   = 5
H_STATE_TOL = 1.0
STATE_MIN_N = 50
SAVE        = True
FS   = 14
TICK = 14

TOGGLE = {'TX': False, 'TP': True}

KEY_TSOL = 'T_solidus'
KEY_TLIQ = 'T_liquidus'

PRIOR_MG_BML = (0.20, 0.80)

# P1:T-X loop 的壓力(GPa)。TX 關閉時不用。
TX_PRESSURES = [16.5, 20.0]
TX_MG_GRID   = np.linspace(0.05, 0.95, 91)
TX_T_PROBE   = 2000.0        # 只是 bml_phase_diagram 的必填引數

# P2:T-P。掃描範圍貼著後驗的界面壓力(5/95 = 18.1 - 21.2 GPa),
# 原本 12-24 會讓散點擠在中間一小條。
TP_MGS    = [0.30, 0.45, 0.60, 0.75]
TP_P_GRID = np.linspace(16.0, 23.0, 36)

SCAT_N    = 6000
SCAT_SEED = 0

STATE_COLOR = {'solid': '#4E6E8E', 'layered': '#7B9E87', 'molten': '#CD5C5C'}
ORDER = ['molten', 'solid', 'layered']
SZ    = {'molten': 2.5, 'solid': 2.5, 'layered': 9.0}
AL    = {'molten': 0.20, 'solid': 0.25, 'layered': 0.55}

# beno3 沒有 X display,而且沒有 PingFang TC。無 DISPLAY 就用 Agg,
# 否則 plt.show() 會擋住或直接報錯。
import matplotlib
HEADLESS = (os.environ.get('DISPLAY', '') == '' and sys.platform != 'darwin')
if HEADLESS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
print('headless = %s   backend = %s' % (HEADLESS, matplotlib.get_backend()))


def first_existing(cands, what, must_glob):
    """只檢查目錄存在不夠 -- 父目錄常常存在但裡面沒有目標檔案,
    那樣會選中後才失敗。這裡要求目錄內確實有 must_glob 的匹配。"""
    for c in cands:
        if not os.path.isdir(c):
            continue
        if not glob.glob(os.path.join(c, must_glob)):
            continue
        return c
    print('%s 候選都不合格 (需要目錄內有 %s):' % (what, must_glob))
    for c in cands:
        isd = os.path.isdir(c)
        n = len(glob.glob(os.path.join(c, must_glob))) if isd else 0
        print('    %-62s dir=%-5s match=%d' % (c, isd, n))
    raise SystemExit('請直接改 %s 的第一項' % what)


CHAIN_DIR    = first_existing(CHAIN_CANDS, 'CHAIN_CANDS', PREFIX + '*')
PHASE_MODULE = first_existing(PHASE_CANDS, 'PHASE_CANDS', PHASE_GLOB)
print('CHAIN_DIR    =', CHAIN_DIR)
print('PHASE_MODULE =', PHASE_MODULE)

# ══════════════════════════════════════════════════════════════════════════
# 三態分類(與 113 / 122 / 124-129 逐字相同)
# ══════════════════════════════════════════════════════════════════════════
def classify_state(h_sol, h_liq, tol=H_STATE_TOL):
    h_sol = np.atleast_1d(np.asarray(h_sol, float))
    h_liq = np.atleast_1d(np.asarray(h_liq, float))
    st = np.full(h_sol.shape, 'fail', dtype='<U8')
    ok = np.isfinite(h_sol) & np.isfinite(h_liq)
    s_y = ok & (h_sol >= tol)
    l_y = ok & (h_liq >= tol)
    st[s_y & l_y]  = 'layered'
    st[~s_y & l_y] = 'molten'
    st[s_y & ~l_y] = 'solid'
    return st

# ══════════════════════════════════════════════════════════════════════════
# 讀鏈(全記錄含重複 = 停留時間權重)
# ══════════════════════════════════════════════════════════════════════════
all_chains = []
print('')
print(f"{'chain':<22s} {'n_raw':>7s}  action")
for dd in sorted(glob.glob(os.path.join(CHAIN_DIR, PREFIX + '*'))):
    name = os.path.basename(dd)
    if name in EXCLUDE:
        print(f'{name:<22s} {0:>7d}  EXCLUDED')
        continue
    path = os.path.join(dd, 'chain.jsonl')
    if not os.path.exists(path):
        print(f'{name:<22s} {0:>7d}  DROPPED (no chain.jsonl)')
        continue
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    n_raw = len(recs)
    recs = recs[:-DROP_LAST] if len(recs) > DROP_LAST else []
    if len(recs) < MIN_STEPS:
        print(f'{name:<22s} {n_raw:>7d}  DROPPED (too short)')
        continue
    all_chains.append(recs)
    print(f'{name:<22s} {n_raw:>7d}  kept {len(recs)}')

if not all_chains:
    print('CHAIN_DIR 內 chain* 開頭的項目:',
          sorted(d for d in os.listdir(CHAIN_DIR) if d.startswith('chain')))
    raise SystemExit('no usable chains -- 檢查 CHAIN_DIR 與 PREFIX')

BURN = [int(len(ch) * BURNIN_FRAC) for ch in all_chains]
samples = []
for c in range(len(all_chains)):
    samples.extend(all_chains[c][BURN[c]:])
print(f'\nusable chains: {len(all_chains)}   post burn-in samples: {len(samples)}')


def top(k):
    return np.array([np.nan if s.get(k) is None else float(s.get(k, np.nan))
                     for s in samples])


Mg_bml = np.array([float(s['params']['Mg#_bulk_bml']) for s in samples])
h_sol  = top('h_solid_km')
h_liq  = top('h_liquid_km')
T_int  = top('T_interface')
P_int  = top('P_interface')

state = classify_state(h_sol, h_liq)
good  = (state != 'fail')

print('\nMg#_BML 5/50/95 : %s'
      % np.array2string(np.percentile(Mg_bml[good], [5, 50, 95]), precision=3))
m_int = good & np.isfinite(T_int) & np.isfinite(P_int)
print('interface points available : %d / %d' % (int(m_int.sum()), int(good.sum())))
if m_int.sum() > 0:
    print('  T_interface 5/50/95 : %s'
          % np.array2string(np.percentile(T_int[m_int], [5, 50, 95]), precision=0))
    print('  P_interface 5/50/95 : %s'
          % np.array2string(np.percentile(P_int[m_int], [5, 50, 95]), precision=1))
    for stn in ['solid', 'layered', 'molten']:
        sel = m_int & (state == stn)
        if sel.sum() < STATE_MIN_N:
            continue
        print('  %-8s n=%6d  T %s  P %s'
              % (stn, int(sel.sum()),
                 np.array2string(np.percentile(T_int[sel], [5, 50, 95]),
                                 precision=0),
                 np.array2string(np.percentile(P_int[sel], [5, 50, 95]),
                                 precision=1)))

# ══════════════════════════════════════════════════════════════════════════
# 載入相圖
# 70.mcmc_core_nGibbs.py 的模組名不合法(開頭數字 + 中間有點),
# import_module('70.mcmc_core_nGibbs') 會把點當套件分隔而失敗。
# ⚠️ 這會執行該檔的模組層程式碼。若它在模組層跑 argparse 或重建相圖表,
#    這裡會很慢或直接失敗 -- 那就把 bml_phase_diagram 及其相依抽成獨立檔。
# ══════════════════════════════════════════════════════════════════════════
cand = sorted(glob.glob(os.path.join(PHASE_MODULE, PHASE_GLOB)))
if not cand:
    print('PHASE_MODULE 內容:', sorted(os.listdir(PHASE_MODULE))[:40])
    raise SystemExit('找不到 %s' % PHASE_GLOB)
print('\nloading phase module:', cand[0])

sys.path.insert(0, PHASE_MODULE)          # 讓它自己 import core_fes 等相依
spec = importlib.util.spec_from_file_location('phase_mod', cand[0])
mod = importlib.util.module_from_spec(spec)
sys.modules['phase_mod'] = mod
spec.loader.exec_module(mod)
bml_phase_diagram = mod.bml_phase_diagram

probe = bml_phase_diagram(2200.0, 17.7, 0.50)
print('bml_phase_diagram keys :', sorted(probe.keys()))
print('probe (2200 K, 17.7 GPa, Mg# 0.50) :', probe)


def sol_liq(T, P, X):
    d = bml_phase_diagram(float(T), float(P), float(X))
    return d[KEY_TSOL], d[KEY_TLIQ]

# ══════════════════════════════════════════════════════════════════════════
# P1: T-X 二元 loop(預設關閉)
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['TX']:
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for P, ls in zip(TX_PRESSURES, ['-', '--', ':']):
        ts, tl = [], []
        for X in TX_MG_GRID:
            a, b = sol_liq(TX_T_PROBE, P, X)
            ts.append(a)
            tl.append(b)
        ax.plot(TX_MG_GRID, ts, ls, lw=2.2, color=STATE_COLOR['solid'],
                label='solidus  %.1f GPa' % P)
        ax.plot(TX_MG_GRID, tl, ls, lw=2.2, color=STATE_COLOR['molten'],
                label='liquidus %.1f GPa' % P)
        print('  %.1f GPa : T_sol %.0f-%.0f K   T_liq %.0f-%.0f K'
              % (P, min(ts), max(ts), min(tl), max(tl)))

    lo, hi = np.percentile(Mg_bml[good], [5, 95])
    ax.axvspan(lo, hi, color='0.75', alpha=0.45, zorder=0,
               label='posterior bulk Mg#$_{BML}$ (5-95%)')
    ax.set_xlim(TX_MG_GRID[0], TX_MG_GRID[-1])
    ax.set_xlabel('Mg#   (Fo-Fa binary)', fontsize=FS)
    ax.set_ylabel('temperature (K)', fontsize=FS)
    ax.tick_params(labelsize=TICK)
    ax.legend(loc='lower right', fontsize=FS - 4, framealpha=0.92)
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(OUT_DIR, '127_phase_TX.png'), dpi=200)
        print('saved 127_phase_TX.png')

# ══════════════════════════════════════════════════════════════════════════
# P2: solidus / liquidus vs P + 後驗界面點
# 界面點落在 solidus 左側 -> 全固;liquidus 右側 -> 全熔;之間 -> 分層。
# ══════════════════════════════════════════════════════════════════════════
if TOGGLE['TP']:
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    cmap = plt.get_cmap('viridis')
    for i, X in enumerate(TP_MGS):
        col = cmap(i / max(len(TP_MGS) - 1, 1))
        ts, tl = [], []
        for P in TP_P_GRID:
            a, b = sol_liq(TX_T_PROBE, P, X)
            ts.append(a)
            tl.append(b)
        ax.plot(ts, TP_P_GRID, '-',  lw=2.2, color=col, label='Mg# = %.2f' % X)
        ax.plot(tl, TP_P_GRID, '--', lw=2.2, color=col)

    if m_int.sum() > 0:
        rng = np.random.default_rng(SCAT_SEED)
        for stn in ORDER:
            sel = m_int & (state == stn)
            if sel.sum() < STATE_MIN_N:
                continue
            k = np.where(sel)[0]
            if len(k) > SCAT_N:
                k = rng.choice(k, SCAT_N, replace=False)
            ax.scatter(T_int[k], P_int[k], s=SZ[stn], color=STATE_COLOR[stn],
                       alpha=AL[stn], lw=0, zorder=4)
        for stn in ['solid', 'layered', 'molten']:
            if (m_int & (state == stn)).sum() >= STATE_MIN_N:
                ax.scatter([], [], s=34, color=STATE_COLOR[stn],
                           label=stn + ' (interface)')
    else:
        print('T_interface / P_interface 不在 chain 裡 -> P2 只畫相界')

    ax.set_ylim(TP_P_GRID[-1], TP_P_GRID[0])
    ax.set_xlabel('temperature (K)', fontsize=FS)
    ax.set_ylabel('pressure (GPa)', fontsize=FS)
    ax.tick_params(labelsize=TICK)
    ax.legend(loc='lower right', fontsize=FS - 4, framealpha=0.92)
    ax.set_title('solid line: solidus    dashed: liquidus', fontsize=FS - 1)
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(OUT_DIR, '127_phase_TP.png'), dpi=200)
        print('saved 127_phase_TP.png')

if not HEADLESS:
    plt.show()

for T in [1800.0, 2400.0, 3000.0]:
    d = bml_phase_diagram(T, 19.5, 0.50)
    print('T=%.0f -> T_sol %.2f  T_liq %.2f' % (T, d['T_solidus'], d['T_liquidus']))