#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:11:41 2026

@author: chingchen

66.plot_result_profile.py
Plot top-N best-fit nGibbs MCMC profiles vs HeFESTo vs Samuel 2023 vs Khan 2023
四個 panel: Vp / Vs / rho / T,右軸是壓力。
預設一路畫到地心(3389.5 km),含核心;要只看地函把 PLOT_CORE 關掉。

讀檔支援 chain.json(a31-a33,整檔 array + salvage)與
chain.jsonl(a34 起,逐行 append,寫壞只丟最後一行)。

a36 / a37 更新 —— profile npz 格式換掉了:
  a35 及之前: depth_km / Vp / ... + bml_* / core_*(三段分開存)
  a36 / a37:  z_km / layer / P_GPa / T_K / rho / Vp / Vs / g 單一全行星陣列
              + z_bml_top / z_cmb / h_solid_km / h_liquid_km / R_ic_km
              layer 編碼 0=固定殼層 1=mantle 2=BML 3=core
  用 read_profile() 正規化,跟 64.BML_profiles.py 同一套(兩支各自帶一份,
  維持每支 script 可獨立執行)。

其他修正:
  * 原本細線迴圈的淺部內插誤用 best_npz,導致 top-N 每一條的 0-100 km
    都畫成最佳模型的形狀。改成各自的 npz。
  * 深度視窗由實際 z_cmb 決定;a36/a37 的 CMB 可深到 1940 km,
    寫死 1800 會把 BML 底部切掉。
  * a35 的 bml_outer_core_depth_abs 在 a36+ 沒有;改標 z_cmb(核頂)與
    z_cmb - h_liquid_km(BML 內固/液次層界面)。
  * PREFIX 必須指定 A 或 B —— 兩者是不同的核 EoS,剖面不可混畫。
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mars_config import DATA_BASE, KHAN_MODEL_DIR, MCMC_DIR, FIG_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
MCMC_BASE_DIR  = MCMC_DIR
CHAIN_PREFIX   = 'chain_a60_TA'
HEF_PREFIX     = 'chain_a21'
SAMUEL_BASE    = (f'{DATA_BASE}/Mars_Samuel_2023/'
                  'Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1')

OUT_DIR        = FIG_DIR

DEPTH_MAX      = 1800        # Khan / Samuel 對照資料的深度上限(Z 網格用)
SPLICE_DEPTH   = 100.0
MARS_RADIUS    = 3389.5
# a36 起 R_cmb 是自由參數 -> CMB 深度逐模型不同。這個常數只給 HeFESTo 分支退回用。
FALLBACK_CMB_DEPTH = 1743.3
PHI_OUTER_CORE = 0.30
MG_DUNCAN_REF  = 0.75
TOP_N          = 50

# 圖的深度視窗。None = 自動:PLOT_CORE 開就畫到地心,關就畫到最深的 CMB
PLOT_DEPTH_MAX = None
PLOT_PAD       = 60.0        # km

# 要不要把核心(layer 3)也畫進去。預設開:一路畫到地心 3389.5 km。
#   開: 看得到 (a) 核心絕熱線 —— R_ic 由它跟 Fe-S 熔點的交點決定
#              (b) CMB 的密度跳躍 rho ~4.2 -> ~7.1,那是質量/MoI 真正在約束的量
#              (c) nGibbs 的核心密度 vs Samuel(6.6)/Khan(6.4)的系統性偏移
#       代價是地函被壓扁約 1.7 倍。
#   關: 深度只到最深的 CMB,地函細節看得清楚。
#   Vs 在液態核是 0,畫出來就是 S 波陰影,不是 bug。
PLOT_CORE    = True

PLOT_HEFESTO = False
SAVE_FIG     = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

if CHAIN_PREFIX in ('chain_a36', 'chain_a36_', 'chain_a37', 'chain_a37_'):
    raise SystemExit(
        f"CHAIN_PREFIX='{CHAIN_PREFIX}' 會同時抓到 A 與 B。\n"
        "  A = all-Liu (Gamma mixing), B = Liu Fe + Sakai FeS (Margules),\n"
        "  是兩套不同的核 EoS,剖面不可混畫。請加上 A 或 B。")

Z = np.linspace(0, DEPTH_MAX, 500)
os.makedirs(OUT_DIR, exist_ok=True)

# ── CHAIN READER (json + jsonl) ────────────────────────────────────────────────
def load_chain_json(path):
    """salvage-capable reader,給舊版整檔 chain.json 用"""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return None
    try:
        return json.loads(text)
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
    return recs if recs else None


def load_chain_jsonl(path):
    """逐行讀 chain.jsonl;只有最後一行可能寫到一半,壞掉就丟棄該行"""
    if not os.path.exists(path):
        return None
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs if recs else None


def load_chain_dir(chain_dir):
    """優先找 chain.jsonl(a34+),沒有就退回 chain.json(a31-a33)"""
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    json_path  = os.path.join(chain_dir, 'chain.json')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(json_path)


def iter_chain_dirs(base_dir, prefix):
    return sorted(glob.glob(f'{base_dir}/{prefix}_*'))


# ── PROFILE npz READER: a35 舊格式 / a36+ 新格式正規化 ─────────────────────────
LAY_FIXED, LAY_MANTLE, LAY_BML, LAY_CORE = 0, 1, 2, 3


def _seg(z, npz, mask, keymap):
    """由遮罩切出一段;段為空回 None"""
    if not np.any(mask):
        return None
    out = {'depth': z[mask]}
    for k_out, k_in in keymap.items():
        out[k_out] = np.asarray(npz[k_in])[mask] if k_in in npz.files else None
    return out


def read_profile(npz):
    """回傳 dict:
        fmt / mantle / bml / core / z_bml_top / z_cmb / z_sl / R_ic_km
    每段是 {'depth','Vp','Vs','rho','T','P'}(缺的欄位為 None)。
    z_sl = BML 內固/液次層界面深度(a35 用 bml_outer_core_depth_abs 取代)。
    """
    f = set(npz.files)

    def _s(key):
        return float(np.asarray(npz[key]).ravel()[0]) if key in f else None

    out = {'mantle': None, 'bml': None, 'core': None,
           'z_bml_top': None, 'z_cmb': None, 'z_sl': None, 'R_ic_km': None}

    if 'z_km' in f and 'layer' in f:                       # ── a36 / a37 ──
        out['fmt'] = 'a36'
        z, lay = np.asarray(npz['z_km']), np.asarray(npz['layer'])
        km = {'Vp': 'Vp', 'Vs': 'Vs', 'rho': 'rho', 'T': 'T_K', 'P': 'P_GPa'}
        out['mantle'] = _seg(z, npz, lay <= LAY_MANTLE, km)
        out['bml']    = _seg(z, npz, lay == LAY_BML,    km)
        out['core']   = _seg(z, npz, lay == LAY_CORE,   km)
        out['z_bml_top'] = _s('z_bml_top')
        out['z_cmb']     = _s('z_cmb')
        out['R_ic_km']   = _s('R_ic_km')
        h_liq = _s('h_liquid_km')
        if out['z_cmb'] is not None and h_liq is not None and h_liq > 0:
            out['z_sl'] = out['z_cmb'] - h_liq
        return out

    out['fmt'] = 'a35'                                     # ── a35 及之前 ──
    if 'depth_km' in f:
        out['mantle'] = {'depth': np.asarray(npz['depth_km']),
                         'Vp': npz['Vp'] if 'Vp' in f else None,
                         'Vs': npz['Vs'] if 'Vs' in f else None,
                         'rho': npz['rho'] if 'rho' in f else None,
                         'T': npz['T_K'] if 'T_K' in f else None,
                         'P': npz['P_GPa'] if 'P_GPa' in f else None}
    if 'bml_depth_km' in f:
        out['bml'] = {'depth': np.asarray(npz['bml_depth_km']),
                      'Vp': npz['bml_Vp'] if 'bml_Vp' in f else None,
                      'Vs': npz['bml_Vs'] if 'bml_Vs' in f else None,
                      'rho': npz['bml_rho'] if 'bml_rho' in f else None,
                      'T': npz['bml_T_K'] if 'bml_T_K' in f else None,
                      'P': npz['bml_P_GPa'] if 'bml_P_GPa' in f else None}
        out['z_bml_top'] = float(out['bml']['depth'][0])
        out['z_cmb']     = float(out['bml']['depth'][-1])
        out['z_sl']      = _s('bml_outer_core_depth_abs')
    if 'core_depth_km' in f:
        out['core'] = {'depth': np.asarray(npz['core_depth_km']),
                       'Vp': npz['core_Vp'] if 'core_Vp' in f else None,
                       'Vs': None,
                       'rho': npz['core_rho'] if 'core_rho' in f else None,
                       'T': npz['core_T_K'] if 'core_T_K' in f else None,
                       'P': npz['core_P_GPa'] if 'core_P_GPa' in f else None}
        out['R_ic_km'] = _s('core_R_ic_km')
    return out


# ── PIERRU CORRECTION ─────────────────────────────────────────────────────────
def solidus_duncan2018(P):
    P = np.atleast_1d(np.asarray(P, dtype=float))
    T_C = np.where(P <= 10.0,
            -4.877*P**2 + 120.2*P + 1088.0,
            np.where(P <= 23.0,
                -1.323*(P-10)**2 + 38.18*(P-10) + 1802.0,
                77.75*(P-23) + 2075.0))
    return T_C + 273.15

def compute_phi(T_K, P_GPa, Mg_bml):
    dT   = -6.0 * (100*(1-Mg_bml) - 100*(1-MG_DUNCAN_REF))
    Tsol = solidus_duncan2018(P_GPa) + dT
    Tliq = (2160.6 + 64.7109*P_GPa - 3.97463*P_GPa**2 + 0.0957894*P_GPa**3) + dT
    Tliq = np.where(Tliq <= Tsol, Tsol + 200, Tliq)
    return np.clip((T_K - Tsol) / (Tliq - Tsol), 0, 1)

def apply_pierru(Vp, Vs, phi):
    phi_n = [0.0, 0.10, 0.20, 0.30, 1.0]
    dVp_n = [0.0, 0.10, 0.175, 0.20, 0.20]
    dVs_n = [0.0, 0.20, 0.40,  1.00, 1.00]
    dVp = np.interp(phi, phi_n, dVp_n)
    dVs = np.interp(phi, phi_n, dVs_n)
    return Vp*(1-dVp), np.maximum(Vs*(1-dVs), 0.0)



def fmt(p):
    if not p: return ''
    s = (f"T_lit={p.get('T_lit',0):.0f}K  P_lit={p.get('P_lit',0):.2f}GPa  "
         f"Mg#={p.get('Mg#',0):.3f}  T_core={p.get('T_core',0):.0f}K  "
         f"Mg#_bulk={p.get('Mg#_bulk_bml',0):.3f}  "
         f"BML={p.get('BML_thickness',0):.0f}km")
    if 'w_S' in p:                       # a35+
        s += f"  w_S={p['w_S']:.3f}"
    if 'R_cmb' in p:                     # a36+
        s += f"  R_cmb={p['R_cmb']:.0f}km"
    return s
# ── READ FORT.56 ──────────────────────────────────────────────────────────────
def read_fort56(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            f.readline(); cols = f.readline().split()
        df = pd.read_csv(path, sep=r'\s+', skiprows=2, names=cols)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['P(GPa)', 'T(K)', 'rho(g/cm^3)', 'VS(km/s)', 'VP(km/s)'])
        if df.empty: return None
        P   = df['P(GPa)'].values
        rho = df['rho(g/cm^3)'].values
        dP  = np.diff(P) * 1e9
        dz  = dP / ((rho[:-1]+rho[1:])/2 * 1000 * 3.72) / 1000
        depth = np.zeros(len(P)); depth[1:] = np.cumsum(dz)
        return {'depth': depth, 'P': P, 'T': df['T(K)'].values,
                'rho': rho, 'Vp': df['VP(km/s)'].values, 'Vs': df['VS(km/s)'].values}
    except:
        return None

# ── FIND TOP-N nGibbs MODELS ──────────────────────────────────────────────────
def find_top_n(base_dir, prefix, n=20):
    results = []
    for chain_dir in iter_chain_dirs(base_dir, prefix):
        chain = load_chain_dir(chain_dir)
        if not chain:
            continue
        for s in chain:
            if s.get('accepted'):
                results.append({
                    'misfit':    s['misfit'],
                    'step':      s['step'],
                    'chain_dir': chain_dir,
                    'params':    s['params'],
                })
    results.sort(key=lambda x: x['misfit'])
    return results[:n]

# ── FIND BEST HeFESTo MODEL ───────────────────────────────────────────────────
def find_best(base_dir, prefix):
    best = {'misfit': 999.0, 'step': None, 'chain_dir': None, 'params': None}
    for chain_dir in iter_chain_dirs(base_dir, prefix):
        chain = load_chain_dir(chain_dir)
        if not chain:
            continue
        for s in chain:
            if s.get('accepted') and s['misfit'] < best['misfit']:
                best = {'misfit': s['misfit'], 'step': s['step'],
                        'chain_dir': chain_dir,
                        'params':    s['params']}
    return best

# ── LOAD SAMUEL 2023 ──────────────────────────────────────────────────────────
print("Loading Samuel 2023...")
def r2d(r): return MARS_RADIUS - r

sam_vp,  dep_vp  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vp_profile.dat').T
sam_vs,  dep_vs  = np.loadtxt(f'{SAMUEL_BASE}/PANEL_K/vs_profile.dat').T
sam_rho, dep_rho = np.loadtxt(f'{SAMUEL_BASE}/PANEL_J/rho_profile.dat').T
sam_T,   dep_T   = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tprofile.dat').T
sam_sol, dep_sol = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tsol.dat').T
sam_liq, dep_liq = np.loadtxt(f'{SAMUEL_BASE}/PANEL_I/Tliq.dat').T

# ── LOAD KHAN ENSEMBLE ────────────────────────────────────────────────────────

print("Loading Khan models...")
# 欄位: 0=Depth 1=Vp 2=Vs 3=Rho 4=T 5=g 6=P
COL_DEPTH, COL_VP, COL_VS, COL_RHO, COL_T, COL_G, COL_P = range(7)

khan_d, khan_vp, khan_vs, khan_rho, khan_T = [], [], [], [], []
for fpath in sorted(glob.glob(f'{KHAN_MODEL_DIR}/Model_*.txt')):
    try:
        d = np.loadtxt(fpath, comments='#')
        mask = d[:, COL_DEPTH] <= DEPTH_MAX 
        if mask.sum() < 5: continue
        khan_d.append(d[mask, COL_DEPTH])
        khan_vp.append(d[mask, COL_VP])
        khan_vs.append(d[mask, COL_VS])
        khan_rho.append(d[mask, COL_RHO])
        khan_T.append(d[mask, COL_T])
    except:
        continue
print(f"  {len(khan_d)} Khan models")
def khan_stats(vals):
    mat = np.full((len(khan_d), len(Z)), np.nan)
    for i, (z, v) in enumerate(zip(khan_d, vals)):
        if z.max() > DEPTH_MAX*0.8:
            mat[i] = np.interp(Z, z, v, left=np.nan, right=np.nan)
    return np.nanmedian(mat,0), np.nanpercentile(mat,16,0), np.nanpercentile(mat,84,0)

vp_med,  vp_p16,  vp_p84  = khan_stats(khan_vp)
vs_med,  vs_p16,  vs_p84  = khan_stats(khan_vs)
rho_med, rho_p16, rho_p84 = khan_stats(khan_rho)

# ── LOAD TOP-20 nGibbs PROFILES ───────────────────────────────────────────────
print(f"\nSearching top {TOP_N} in {CHAIN_PREFIX}...")
top_n = find_top_n(MCMC_BASE_DIR, CHAIN_PREFIX, n=TOP_N)
ng_profiles = []
for r in top_n:
    npz_path = f"{r['chain_dir']}/profile_s{r['step']:05d}.npz"
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        ng_profiles.append({
            'npz':       npz,
            'prof':      read_profile(npz),
            'misfit':    r['misfit'],
            'params':    r['params'],
            'chain_dir': r['chain_dir'],
            'step':      r['step'],
        })
print(f"  Loaded {len(ng_profiles)} nGibbs profiles")
if not ng_profiles:
    raise SystemExit(
        f"No accepted+profiled steps found for prefix '{CHAIN_PREFIX}' under {MCMC_BASE_DIR}.\n"
        f"檢查: (1) chain 目錄是否存在 chain.json 或 chain.jsonl "
        f"(2) profile_s*.npz 是否有跟著存 (3) 該鏈是否還太短、還沒有 accepted 步")

_fmts = {p['prof']['fmt'] for p in ng_profiles}
print(f"  npz format: {sorted(_fmts)}")
if len(_fmts) > 1:
    print("  !! 同時載到 a35 與 a36+ 兩種格式,幾何定義不同,不應混畫")

# top-N 全來自同一條鏈時,圖上的「族群」其實只是那條鏈的自相關軌跡
from collections import Counter
_byc = Counter(os.path.basename(p['chain_dir']) for p in ng_profiles)
print(f"  來自 {len(_byc)} 條鏈: " + ', '.join(f'{k}:{v}' for k, v in _byc.most_common()))
if len(_byc) == 1:
    print("  !! 全部來自同一條鏈 -> 不是後驗的代表性抽樣,只是最佳鏈的局部軌跡")

best_ng   = ng_profiles[0]
best_npz  = best_ng['npz']
best_prof = best_ng['prof']

# pressure axis reference from best model (地函段)
ng_P     = best_prof['mantle']['P']
ng_depth = best_prof['mantle']['depth']

m_shal = Z <= SPLICE_DEPTH

# BML 幾何(取自最佳模型)
z_bml_top = best_prof['z_bml_top']
z_cmb     = best_prof['z_cmb']
z_sl      = best_prof['z_sl']

# 深度視窗:確保 a36/a37 較深的 CMB 不會被切掉
if PLOT_DEPTH_MAX is None:
    _bots = [p['prof']['z_cmb'] for p in ng_profiles if p['prof']['z_cmb'] is not None]
    if PLOT_CORE:
        PLOT_DEPTH_MAX = MARS_RADIUS
    else:
        PLOT_DEPTH_MAX = max(DEPTH_MAX,
                             (max(_bots) + PLOT_PAD) if _bots else DEPTH_MAX)
    if _bots:
        print(f"  深度視窗(自動): 0 - {PLOT_DEPTH_MAX:.0f} km  "
              f"[z_cmb {min(_bots):.0f}-{max(_bots):.0f}]"
              f"{'  (含核心)' if PLOT_CORE else ''}")

# CMB 溫度連續性檢查。BML 底就是核頂,溫度應該接得上;
# 密度則本來就該跳(那是質量/MoI 在約束的量)。這裡不連續代表 forward
# model 或 npz 拼接出了問題,比任何圖都早該發現。
_bp = best_prof
if _bp['bml'] is not None and _bp['core'] is not None:
    _dT = float(_bp['core']['T'][0] - _bp['bml']['T'][-1])
    _dr = float(_bp['core']['rho'][0] - _bp['bml']['rho'][-1])
    print(f"  CMB 接合檢查(最佳模型): dT = {_dT:+.3f} K   "
          f"drho = {_dr:+.3f} g/cm3")
    if abs(_dT) > 1.0:
        print("  !! CMB 溫度不連續超過 1 K,檢查 forward model 的核心邊界條件")


def _spliced(prof, key):
    """地函段:淺於 SPLICE_DEPTH 用共用 Z 網格內插,深部用原節點"""
    man = prof['mantle']
    if man is None or man.get(key) is None:
        return None, None
    zm, vm = man['depth'], man[key]
    deep = zm > SPLICE_DEPTH
    d = np.concatenate([Z[m_shal], zm[deep]])
    v = np.concatenate([np.interp(Z[m_shal], zm, vm), vm[deep]])
    return v, d


def mark_bml(ax):
    if z_bml_top is not None:
        ax.axhline(z_bml_top, color='orange', ls='--', lw=1.0, alpha=0.6)
    if z_cmb is not None:
        ax.axhline(z_cmb, color='orange', ls='--', lw=1.0, alpha=0.6)
    if z_sl is not None:
        ax.axhline(z_sl, color='orange', ls=':', lw=1.0, alpha=0.5)

# ── LOAD HeFESTo BEST ─────────────────────────────────────────────────────────

hf = None
hf_man = None
has_hf_bml = False

if PLOT_HEFESTO:
    print(f"\nSearching best in {HEF_PREFIX}...")
    hf = find_best(MCMC_BASE_DIR, HEF_PREFIX)
    print(f"  {os.path.basename(hf['chain_dir'])}  step={hf['step']}  misfit={hf['misfit']:.4f}")

    hf_man     = read_fort56(f"{hf['chain_dir']}/step_{hf['step']:05d}/s3_final/fort.56")
    hf_bml_raw = read_fort56(f"{hf['chain_dir']}/step_{hf['step']:05d}/bml/fort.56")
    has_hf_bml = hf_bml_raw is not None and hf['params'] is not None

    if has_hf_bml:
        bml_thick    = hf['params'].get('BML_thickness', 168.7)
        hf_bml_depth = (hf_bml_raw['depth'] - hf_bml_raw['depth'][0]
                        + FALLBACK_CMB_DEPTH - bml_thick)
        Mg_bml       = hf['params'].get('Mg#_bml', 0.7)
        phi          = compute_phi(hf_bml_raw['T'], hf_bml_raw['P'], Mg_bml)
        hf_bml_Vp, hf_bml_Vs = apply_pierru(hf_bml_raw['Vp'], hf_bml_raw['Vs'], phi)
        hf_bml_rho   = hf_bml_raw['rho']
        hf_bml_T     = hf_bml_raw['T']
        oc_mask      = phi >= PHI_OUTER_CORE
        hf_oc        = hf_bml_depth[np.argmax(oc_mask)] if oc_mask.any() else hf_bml_depth[-1]
# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 10), gridspec_kw={'wspace': 0.45})

# colours
C_NG  = '#C0392B'   # nGibbs
C_HF  = '#2C3E50'   # HeFESTo
C_SAM = '#708090'   # Samuel
C_KH  = '#4198B9'   # Khan Vp
C_RHO = '#35838D'   # Khan rho

panel_cfg = [
    # (xlabel, khan_vals, khan_med, p16, p84, khan_color, key)
    ('Vp (km/s)',  khan_vp,  vp_med,  vp_p16,  vp_p84,  C_KH,  'Vp'),
    ('Vs (km/s)',  khan_vs,  vs_med,  vs_p16,  vs_p84,  C_KH,  'Vs'),
    ('ρ (g/cm³)', khan_rho, rho_med, rho_p16, rho_p84, C_RHO, 'rho'),
]
sam_vals = [sam_vp/1000, sam_vs/1000, sam_rho/1000]
sam_deps = [r2d(dep_vp),  r2d(dep_vs),  r2d(dep_rho)]
hf_vals  = ([hf_man['Vp'], hf_man['Vs'], hf_man['rho']] if (PLOT_HEFESTO and hf_man) else [None]*3)
hf_bml_vals = ([hf_bml_Vp, hf_bml_Vs, hf_bml_rho] if (PLOT_HEFESTO and has_hf_bml) else [None]*3)

for i, (ax, (xlabel, kvals, med, p16, p84, kc, key),
        sv, sd, hv, hbv) in enumerate(
            zip(axes[:3], panel_cfg, sam_vals, sam_deps, hf_vals, hf_bml_vals)):

    # Khan ensemble
    for kd, kv in zip(khan_d, kvals):
        ax.plot(kv, kd, color=kc, alpha=0.04, lw=0.5, rasterized=True)
    ax.fill_betweenx(Z, p16, p84, color=kc, alpha=0.20, label='Khan 16–84%')
    ax.plot(med, Z, color=kc, lw=2.0, label='Khan median')

    # Samuel
    ax.plot(sv, sd, color=C_SAM, lw=2.0, label='Samuel 2023')

    # nGibbs top-N (thin + transparent)
    # 原本這裡的淺部內插用的是 best_npz,等於把最佳模型的 0-100 km
    # 複製到每一條細線上。改成各自的剖面。
    for p in ng_profiles[1:]:
        pr = p['prof']
        v, d = _spliced(pr, key)
        if v is not None:
            ax.plot(v, d, color=C_NG, lw=0.8, alpha=0.25, zorder=4)
        for _s in (['bml', 'core'] if PLOT_CORE else ['bml']):
            seg = pr[_s]
            if seg is not None and seg.get(key) is not None:
                ax.plot(seg[key], seg['depth'],
                        color=C_NG, lw=0.8, alpha=0.25, zorder=4)

    # nGibbs best (bold)
    v_best, d_best = _spliced(best_prof, key)
    if v_best is not None:
        ax.plot(v_best, d_best, color=C_NG, lw=2.5, zorder=5,
                label=f'nGibbs best (misfit={best_ng["misfit"]:.2f})')
    for _s in (['bml', 'core'] if PLOT_CORE else ['bml']):
        seg = best_prof[_s]
        if seg is not None and seg.get(key) is not None:
            ax.plot(seg[key], seg['depth'], color=C_NG, lw=2.5, zorder=5)

    # HeFESTo mantle + BML
    if hv is not None:
        ax.plot(hv, hf_man['depth'], color=C_HF, lw=2.0, ls='-.', zorder=6,
                label=f'HeFESTo best (misfit={hf["misfit"]:.2f})')
    if hbv is not None:
        ax.plot(hbv, hf_bml_depth, color=C_HF, lw=1.8, ls=':', zorder=6)

    mark_bml(ax)
    ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
    ax.set_ylim(PLOT_DEPTH_MAX, 0)
    ax.set_ylabel('Depth (km)' if i == 0 else '', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.grid(True, ls='--', alpha=0.3, lw=0.8)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    for sp in ax.spines.values(): sp.set_linewidth(2.0)

    # pressure axis
    axr = ax.twinx()
    axr.set_ylim(PLOT_DEPTH_MAX, 0)
    p_ticks = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24])
    p_ticks = p_ticks[p_ticks <= ng_P.max()*1.05]
    axr.set_yticks(np.interp(p_ticks, ng_P, ng_depth))
    axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray', fontsize=9)
    axr.tick_params(axis='y', colors='gray', length=4)
    for sp in axr.spines.values(): sp.set_linewidth(2.0)

# ── Panel 4: Temperature ──────────────────────────────────────────────────────
ax = axes[3]
ax.plot(sam_T,   r2d(dep_T),   color='#2980B9', lw=2.0, label='Samuel 2023 T')
ax.plot(sam_sol, r2d(dep_sol), color='orange',   lw=1.5, ls='--', label='Solidus')
ax.plot(sam_liq, r2d(dep_liq), color='brown',    lw=1.5, ls='--', label='Liquidus')

T_med, T_p16, T_p84 = khan_stats(khan_T)
# 在 Panel 4 迴圈前加：
ax.fill_betweenx(Z, T_p16, T_p84, color='#4198B9', alpha=0.15, label='Khan T 16-84%')
ax.plot(T_med, Z, color='#4198B9', lw=1.5, ls='-', label='Khan T median')

# 地函 + BML 的連續溫度剖面(兩種 npz 格式共用)
def full_T_depth(prof, with_core=False):
    segs = ['mantle', 'bml'] + (['core'] if with_core else [])
    T, d = [], []
    for s in segs:
        seg = prof.get(s)
        if seg is None or seg.get('T') is None:
            continue
        T.append(np.asarray(seg['T'])); d.append(np.asarray(seg['depth']))
    if not T:
        return np.array([]), np.array([])
    return np.concatenate(T), np.concatenate(d)


# nGibbs top-N T (thin)
for p in ng_profiles[1:]:
    T_full, d_full = full_T_depth(p['prof'], with_core=PLOT_CORE)
    ax.plot(T_full, d_full, color='#E67E22', lw=0.8, alpha=0.25, zorder=4)

# nGibbs best T (bold)
T_best, d_best = full_T_depth(best_prof, with_core=PLOT_CORE)
ax.plot(T_best, d_best, color='#E67E22', lw=2.5, zorder=5,
        label='nGibbs best T')

# 內核邊界(如果有)。R_ic 由核心絕熱線跟 Fe-S 熔點的交點決定,
# 目前後驗裡 P(R_ic > 0) 只有約 16%,所以多數模型這條線不會出現。
if PLOT_CORE and best_prof['R_ic_km'] and best_prof['R_ic_km'] > 0:
    _z_ic = MARS_RADIUS - best_prof['R_ic_km']
    ax.axhline(_z_ic, color='#8B0000', ls='--', lw=1.2, alpha=0.8)
    ax.text(0.02, _z_ic - 20,
            f"ICB  R_ic={best_prof['R_ic_km']:.0f} km",
            transform=ax.get_yaxis_transform(), fontsize=7,
            color='#8B0000', va='bottom')

# HeFESTo T
if PLOT_HEFESTO and hf_man:
    hf_T_full = np.concatenate([hf_man['T'], hf_bml_T]) if has_hf_bml else hf_man['T']
    hf_d_full = np.concatenate([hf_man['depth'], hf_bml_depth]) if has_hf_bml else hf_man['depth']
    ax.plot(hf_T_full, hf_d_full, color=C_HF, lw=2.0, ls='-.', zorder=6, label='HeFESTo T')

if PLOT_HEFESTO and has_hf_bml:
    ax.axhline(hf_oc, color=C_HF, ls=':', lw=1.5, alpha=0.8)
    ax.text(0.02, hf_oc+10, f'HeFESTo outer core\n{hf_oc:.0f} km',
            transform=ax.get_yaxis_transform(), fontsize=7, color=C_HF, va='top')

# CMB + BML 固/液界面標註
if z_cmb is not None:
    ax.axhline(z_cmb, color='#E67E22', ls=':', lw=1.5, alpha=0.8)
    ax.text(0.02, z_cmb - 20, f'CMB (core top)\n{z_cmb:.0f} km',
            transform=ax.get_yaxis_transform(), fontsize=7,
            color='#E67E22', va='bottom')
if z_sl is not None:
    ax.axhline(z_sl, color='#E67E22', ls='-.', lw=1.2, alpha=0.7)
    ax.text(0.02, z_sl + 10, f'BML solid/liquid\n{z_sl:.0f} km',
            transform=ax.get_yaxis_transform(), fontsize=7,
            color='#E67E22', va='top')
if z_bml_top is not None:
    ax.axhline(z_bml_top, color='orange', ls='--', lw=1.0, alpha=0.6)

ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_ylim(PLOT_DEPTH_MAX, 0)
ax.set_xlabel('T (K)', fontsize=13)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.tick_params(labelsize=14)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
for sp in ax.spines.values(): sp.set_linewidth(2.0)

axr = axes[3].twinx()
axr.set_ylim(PLOT_DEPTH_MAX, 0)
p_ticks = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24])
p_ticks = p_ticks[p_ticks <= ng_P.max()*1.05]
axr.set_yticks(np.interp(p_ticks, ng_P, ng_depth))
axr.set_yticklabels([f'{p:.0f}' for p in p_ticks], color='gray', fontsize=9)
axr.set_ylabel('Pressure (GPa)', fontsize=10, color='gray')
axr.tick_params(axis='y', colors='gray', length=4)
for sp in axr.spines.values(): sp.set_linewidth(2.0)

# ── TITLE ─────────────────────────────────────────────────────────────────────
title_str = (
    f'Mars Mantle+BML: nGibbs (top {len(ng_profiles)}) vs Samuel 2023 vs Khan 2023\n'
    f"nGibbs best : {os.path.basename(ng_profiles[0]['chain_dir'])}  "
    f"step {ng_profiles[0]['step']}  \n misfit={ng_profiles[0]['misfit']:.4f}  "
    f"|  {fmt(ng_profiles[0]['params'])}"
)
if PLOT_HEFESTO and hf is not None:
    title_str += (f"\nHeFESTo best: {os.path.basename(hf['chain_dir'])}  step {hf['step']}  "
                  f"misfit={hf['misfit']:.4f}  |  {fmt(hf['params'])}")
fig.suptitle(title_str, fontsize=14)
if SAVE_FIG:
    out = os.path.join(OUT_DIR, f'best_fit_{CHAIN_PREFIX}_top{TOP_N}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved: {out}")
plt.show()




pr = read_profile(npz)
m = pr['mantle']
r = (MARS_RADIUS - m['depth'])*1e3
o = np.argsort(r); ra, rhoa = r[o], m['rho'][o]*1e3
rhm = 0.5*(rhoa[1:]+rhoa[:-1])
dV  = (4*np.pi/3)*(ra[1:]**3 - ra[:-1]**3)
print('mantle rho_mean =', float(np.sum(rhm*dV)/np.sum(dV)), ' vs Samuel 3628.6')