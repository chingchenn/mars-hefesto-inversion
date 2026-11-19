
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:16:09 2026
@author: chingchen

Plot top-N best-fit nGibbs MCMC profiles vs HeFESTo vs Samuel 2023 vs Khan 2023
  Figure 1: Vp + Vs 合併 panel + 密度 panel(縮放到 BML 區間)
  Figure 2: 溫度剖面(全深度)

讀檔支援 chain.jsonl(a34 起)與 chain.json(舊版,含 salvage)。

a36 更新 —— profile npz 格式整個換掉了:
  a35 及之前: depth_km / Vp / Vs / rho / T_K / P_GPa
              + bml_depth_km / bml_Vp / ... + core_depth_km / core_Vp / ...
              (地函、BML、核三段分開存)
  a36 起:     z_km / layer / P_GPa / T_K / rho / Vp / Vs / g 單一全行星陣列
              + z_bml_top / z_cmb / h_solid_km / h_liquid_km / R_ic_km 等純量
              layer 編碼 0=固定殼層 1=mantle 2=BML 3=core

  本檔用 read_profile() 把兩種格式正規化成同一個 dict,其餘繪圖碼共用。
  另外:
  * CMB 深度不再固定 1743.3 km -> Figure 1 的深度視窗由實際 z_cmb 自動決定
  * a35 的 bml_outer_core_depth_abs 在 a36 沒有對應欄位;改用 z_cmb(BML 底/
    核頂)與 z_cmb - h_liquid_km(固/液次層界面),後者物理意義更明確
  * PREFIX 必須是 chain_a36A 或 chain_a36B —— A/B 是兩套核 EoS,不可混
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mars_config import DATA_BASE, KHAN_MODEL_DIR, MCMC_DIR, FIG_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
MCMC_BASE_DIR  = MCMC_DIR
CHAIN_PREFIX   = 'chain_a52B'
HEF_PREFIX     = 'chain_a21'
SAMUEL_BASE    = (f'{DATA_BASE}/Mars_Samuel_2023/'
                  'Nature_Samuel_s41586-023-06601-8/METADATA_BML/DATA_FIG1')
OUT_DIR        = FIG_DIR

DEPTH_MAX      = 1800        # Khan / Samuel 對照資料的深度上限(Z 網格用)
SPLICE_DEPTH   = 100.0
MARS_RADIUS    = 3389.5
# a36 起 R_cmb 是自由參數 -> CMB 深度逐模型不同。這個常數只給舊格式/HeFESTo 分支退回用。
FALLBACK_CMB_DEPTH = 1743.3
PHI_OUTER_CORE = 0.30
MG_DUNCAN_REF  = 0.75
TOP_N          = 100

# Figure 1 的深度視窗(BML 區間)。None = 由實際載入的 z_bml_top / z_cmb 自動決定,
# 因為 a36 的 CMB 深度是分布(a36A 約 1700-1940 km),寫死 1400-1800 會切掉。
FIG1_YLIM = None
FIG1_PAD  = 80.0             # km, 自動視窗上下各留的邊界

PLOT_HEFESTO = False
SAVE_FIG     = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR

if CHAIN_PREFIX in ('chain_a36', 'chain_a36_'):
    raise SystemExit(
        "CHAIN_PREFIX='chain_a36' 會同時抓到 a36A 與 a36B。\n"
        "  A = all-Liu (Gamma mixing), B = Liu Fe + Sakai FeS (Margules),\n"
        "  是兩套不同的核 EoS,剖面不可混畫。請指定 chain_a36A 或 chain_a36B。")

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
    """逐行讀;只有最後一行可能寫到一半,壞掉就丟棄該行"""
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
    """優先 chain.jsonl(a34+),沒有才退回 chain.json(a31-a33)"""
    jsonl_path = os.path.join(chain_dir, 'chain.jsonl')
    if os.path.exists(jsonl_path):
        return load_chain_jsonl(jsonl_path)
    return load_chain_json(os.path.join(chain_dir, 'chain.json'))


# ── PROFILE npz READER: a35 舊格式 / a36 新格式正規化 ─────────────────────────
LAY_FIXED, LAY_MANTLE, LAY_BML, LAY_CORE = 0, 1, 2, 3
_FIELDS = ('depth', 'Vp', 'Vs', 'rho', 'T', 'P')


def _seg(z, npz, mask, keymap):
    """由遮罩切出一段,回傳 {'depth','Vp','Vs','rho','T','P'};段為空回 None"""
    if not np.any(mask):
        return None
    out = {'depth': z[mask]}
    for k_out, k_in in keymap.items():
        out[k_out] = np.asarray(npz[k_in])[mask] if k_in in npz.files else None
    return out


def read_profile(npz):
    """把 profile npz 正規化。回傳 dict:
        fmt        'a36' | 'a35'
        mantle     {'depth','Vp','Vs','rho','T','P'}          (a36: layer 0+1)
        bml        同上或 None
        core       同上或 None (a35 的 core 段沒有 Vs)
        z_bml_top  BML 頂部深度 (km) 或 None
        z_cmb      CMB 深度 = BML 底 = 核頂 (km) 或 None
        z_sl       BML 內固/液次層界面深度 (km) 或 None
        R_ic_km    內核半徑 (km) 或 None
    a35 的 bml_outer_core_depth_abs 在 a36 沒有對應欄位;a36 用 z_cmb 取代,
    另外由 h_liquid_km 導出更有物理意義的固/液界面 z_sl。
    """
    f = set(npz.files)

    def _s(key):                       # 純量欄位(存成長度 1 的陣列)
        return float(np.asarray(npz[key]).ravel()[0]) if key in f else None

    out = {'mantle': None, 'bml': None, 'core': None,
           'z_bml_top': None, 'z_cmb': None, 'z_sl': None, 'R_ic_km': None}

    if 'z_km' in f and 'layer' in f:                      # ── a36 ──
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
            out['z_sl'] = out['z_cmb'] - h_liq            # 固/液次層界面
        return out

    # ── a35 及之前 ──
    out['fmt'] = 'a35'
    if 'depth_km' in f:
        zm = np.asarray(npz['depth_km'])
        out['mantle'] = {'depth': zm,
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


# ── PIERRU CORRECTION(僅 HeFESTo 分支使用;nGibbs 的 npz 已含 bml_Vp/bml_Vs)──
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


# ── READ FORT.56(僅 HeFESTo 分支)────────────────────────────────────────────
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
    except Exception:
        return None


# ── FIND TOP-N nGibbs MODELS ──────────────────────────────────────────────────
def find_top_n(base_dir, prefix, n=20):
    results = []
    for chain_dir in sorted(glob.glob(f'{base_dir}/{prefix}_*')):
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
    for chain_dir in sorted(glob.glob(f'{base_dir}/{prefix}_*')):
        chain = load_chain_dir(chain_dir)
        if not chain:
            continue
        for s in chain:
            if s.get('accepted') and s['misfit'] < best['misfit']:
                best = {'misfit': s['misfit'], 'step': s['step'],
                        'chain_dir': chain_dir, 'params': s['params']}
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
    except Exception:
        continue
print(f"  {len(khan_d)} Khan models")


def khan_stats(vals):
    mat = np.full((len(khan_d), len(Z)), np.nan)
    for i, (z, v) in enumerate(zip(khan_d, vals)):
        if z.max() > DEPTH_MAX*0.8:
            mat[i] = np.interp(Z, z, v, left=np.nan, right=np.nan)
    return np.nanmedian(mat, 0), np.nanpercentile(mat, 16, 0), np.nanpercentile(mat, 84, 0)


vp_med,  vp_p16,  vp_p84  = khan_stats(khan_vp)
vs_med,  vs_p16,  vs_p84  = khan_stats(khan_vs)
rho_med, rho_p16, rho_p84 = khan_stats(khan_rho)

# ── LOAD TOP-N nGibbs PROFILES ────────────────────────────────────────────────
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
        f"No accepted+profiled steps found for prefix '{CHAIN_PREFIX}' "
        f"under {MCMC_BASE_DIR}.\n"
        f"  檢查: (1) chain 目錄下是否有 chain.jsonl 或 chain.json\n"
        f"        (2) profile_s*.npz 是否有跟著存\n"
        f"        (3) 該鏈是否還太短、尚無 accepted 步")

_fmts = {p['prof']['fmt'] for p in ng_profiles}
print(f"  npz format: {sorted(_fmts)}")
if len(_fmts) > 1:
    print("  !! 同時載到 a35 與 a36 兩種格式的 npz,幾何定義不同,不應混畫")

# top-N 全部來自同一條鏈時,圖上看到的「族群」其實只是一條鏈的自相關軌跡
from collections import Counter
_byc = Counter(os.path.basename(p['chain_dir']) for p in ng_profiles)
print(f"  top-{len(ng_profiles)} 來自 {len(_byc)} 條鏈: "
      + ', '.join(f'{k}:{v}' for k, v in _byc.most_common()))
if len(_byc) == 1:
    print("  !! 全部來自同一條鏈 -> 這不是後驗的代表性抽樣,只是最佳鏈的局部軌跡")

best_ng   = ng_profiles[0]
best_npz  = best_ng['npz']
best_prof = best_ng['prof']

# pressure/depth axis reference from best model (地函段)
ng_P     = best_prof['mantle']['P']
ng_depth = best_prof['mantle']['depth']

m_shal = Z <= SPLICE_DEPTH

# BML 幾何:a36 由 z_bml_top / z_cmb 給,a35 由 bml_depth_km 的端點給
has_bml   = best_prof['bml'] is not None
z_bml_top = best_prof['z_bml_top']
z_cmb     = best_prof['z_cmb']
z_sl      = best_prof['z_sl']       # 固/液次層界面(a35 是 outer-core-like 深度)


def mark_bml(ax):
    if not has_bml:
        return
    if z_bml_top is not None:
        ax.axhline(z_bml_top, color='orange', ls='--', lw=1.0, alpha=0.6)
    if z_cmb is not None:
        ax.axhline(z_cmb,     color='orange', ls='--', lw=1.0, alpha=0.6)
    if z_sl is not None:
        ax.axhline(z_sl,      color='orange', ls=':',  lw=1.0, alpha=0.5)


# ── Figure 1 深度視窗:由實際載入的剖面決定,不寫死 ───────────────────────────
if FIG1_YLIM is None:
    _tops = [p['prof']['z_bml_top'] for p in ng_profiles
             if p['prof']['z_bml_top'] is not None]
    _bots = [p['prof']['z_cmb'] for p in ng_profiles
             if p['prof']['z_cmb'] is not None]
    if _tops and _bots:
        FIG1_YLIM = (max(_bots) + FIG1_PAD, min(_tops) - FIG1_PAD)
        print(f"  Fig1 深度視窗(自動): {FIG1_YLIM[1]:.0f} - {FIG1_YLIM[0]:.0f} km   "
              f"[z_bml_top {min(_tops):.0f}-{max(_tops):.0f}, "
              f"z_cmb {min(_bots):.0f}-{max(_bots):.0f}]")
    else:
        FIG1_YLIM = (DEPTH_MAX, 1400)


# ── LOAD HeFESTo BEST ─────────────────────────────────────────────────────────
hf = None
hf_man = None
has_hf_bml = False

if PLOT_HEFESTO:
    print(f"\nSearching best in {HEF_PREFIX}...")
    hf = find_best(MCMC_BASE_DIR, HEF_PREFIX)
    if hf['chain_dir'] is None:
        print(f"  no HeFESTo chain found for {HEF_PREFIX}; disabling PLOT_HEFESTO")
        PLOT_HEFESTO = False
    else:
        print(f"  {os.path.basename(hf['chain_dir'])}  step={hf['step']}  "
              f"misfit={hf['misfit']:.4f}")
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
            hf_oc = (hf_bml_depth[np.argmax(oc_mask)] if oc_mask.any()
                     else hf_bml_depth[-1])


# ── PLOT: Figure 1 (Vp+Vs combined panel, Density panel) ──────────────────────
C_NG   = '#C0392B'   # nGibbs (紅)
C_HF   = '#2C3E50'   # HeFESTo (深藍灰)
C_SAM  = '#708090'   # Samuel (灰)
C_KHAN = '#4198B9'   # Khan (藍)
C_RHO  = '#35838D'   # density panel 專用

fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'wspace': 0.4})
fig.patch.set_facecolor('#FAFAFA')


def _spliced(prof, key):
    """地函段:淺於 SPLICE_DEPTH 用共用 Z 網格內插,深部用原節點。回傳 (v, d)"""
    man = prof['mantle']
    if man is None or man.get(key) is None:
        return None, None
    zm, vm = man['depth'], man[key]
    deep = zm > SPLICE_DEPTH
    d = np.concatenate([Z[m_shal], zm[deep]])
    v = np.concatenate([np.interp(Z[m_shal], zm, vm), vm[deep]])
    return v, d


def plot_velocity_panel(ax, key, kvals, med, p16, p84,
                        khan_color, ng_color, ls, label_prefix):
    """畫單一速度量(Vp 或 Vs)到共用 panel 上
    key:        'Vp' 或 'Vs'(正規化後的欄位名,兩種 npz 格式共用)
    khan_color: Khan/Samuel 這組資料的顏色
    ng_color:   nGibbs 這組資料的顏色
    ls:         線型,用來區分 Vp(實線) vs Vs(虛線)
    """
    for kd, kv in zip(khan_d, kvals):
        ax.plot(kv, kd, color=khan_color, alpha=0.04, lw=0.5, ls=ls, rasterized=True)
    ax.fill_betweenx(Z, p16, p84, color=khan_color, alpha=0.15,
                     label=f'Khan {label_prefix} 16–84%')
    ax.plot(med, Z, color=khan_color, lw=2.0, ls=ls,
            label=f'Khan {label_prefix} median')

    for p in ng_profiles[1:]:
        pr = p['prof']
        v, d = _spliced(pr, key)
        if v is not None:
            ax.plot(v, d, color=ng_color, lw=0.8, alpha=0.15, ls=ls, zorder=4)
        if pr['bml'] is not None and pr['bml'].get(key) is not None:
            ax.plot(pr['bml'][key], pr['bml']['depth'], color=ng_color, lw=0.8,
                    alpha=0.15, ls=ls, zorder=4)

    v_best, d_best = _spliced(best_prof, key)
    if v_best is not None:
        ax.plot(v_best, d_best, color=ng_color, lw=2.5, ls=ls, zorder=5,
                label=f'nGibbs {label_prefix} best '
                      f'(misfit={best_ng["misfit"]:.2f})')
    if best_prof['bml'] is not None and best_prof['bml'].get(key) is not None:
        ax.plot(best_prof['bml'][key], best_prof['bml']['depth'], color=ng_color,
                lw=2.5, ls=ls, zorder=5)


# Panel 0: Vp + Vs combined
ax = axes[0]
plot_velocity_panel(ax, 'Vp', khan_vp, vp_med, vp_p16, vp_p84,
                    khan_color=C_KHAN, ng_color=C_NG, ls='-', label_prefix='Vp')
plot_velocity_panel(ax, 'Vs', khan_vs, vs_med, vs_p16, vs_p84,
                    khan_color=C_KHAN, ng_color=C_NG, ls='--', label_prefix='Vs')
ax.plot(sam_vp/1000, r2d(dep_vp), color=C_SAM, lw=2.0, ls='-',  label='Samuel Vp')
ax.plot(sam_vs/1000, r2d(dep_vs), color=C_SAM, lw=2.0, ls='--', label='Samuel Vs')

if PLOT_HEFESTO and hf_man is not None:
    ax.plot(hf_man['Vp'], hf_man['depth'], color=C_HF, lw=2.0, ls='-.', zorder=6,
            label=f'HeFESTo Vp (misfit={hf["misfit"]:.2f})')
    ax.plot(hf_man['Vs'], hf_man['depth'], color=C_HF, lw=1.6, ls=':', zorder=6,
            label='HeFESTo Vs')
    if has_hf_bml:
        ax.plot(hf_bml_Vp, hf_bml_depth, color=C_HF, lw=1.8, ls='-.', zorder=6)
        ax.plot(hf_bml_Vs, hf_bml_depth, color=C_HF, lw=1.8, ls=':', zorder=6)

mark_bml(ax)
ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_ylim(*FIG1_YLIM)
ax.set_ylabel('Depth (km)', fontsize=13)
ax.set_xlabel('Velocity (km/s)', fontsize=13)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.tick_params(labelsize=14)
ax.legend(fontsize=7, loc='upper left', framealpha=0.85, ncol=2)
for sp in ax.spines.values(): sp.set_linewidth(2.0)

# Panel 1: Density
ax = axes[1]
for kd, kv in zip(khan_d, khan_rho):
    ax.plot(kv, kd, color=C_RHO, alpha=0.04, lw=0.5, rasterized=True)
ax.fill_betweenx(Z, rho_p16, rho_p84, color=C_RHO, alpha=0.20, label='Khan 16–84%')
ax.plot(rho_med, Z, color=C_RHO, lw=2.0, label='Khan median')
ax.plot(sam_rho/1000, r2d(dep_rho), color=C_SAM, lw=2.0, label='Samuel 2023')

for p in ng_profiles[1:]:
    pr = p['prof']
    v, d = _spliced(pr, 'rho')
    if v is not None:
        ax.plot(v, d, color=C_NG, lw=0.8, alpha=0.25, zorder=4)
    if pr['bml'] is not None and pr['bml'].get('rho') is not None:
        ax.plot(pr['bml']['rho'], pr['bml']['depth'], color=C_NG, lw=0.8,
                alpha=0.25, zorder=4)

v_best, d_best = _spliced(best_prof, 'rho')
if v_best is not None:
    ax.plot(v_best, d_best, color=C_NG, lw=2.5, zorder=5,
            label=f'nGibbs best (misfit={best_ng["misfit"]:.2f})')
if best_prof['bml'] is not None and best_prof['bml'].get('rho') is not None:
    ax.plot(best_prof['bml']['rho'], best_prof['bml']['depth'], color=C_NG,
            lw=2.5, zorder=5)

if PLOT_HEFESTO and hf_man is not None:
    ax.plot(hf_man['rho'], hf_man['depth'], color=C_HF, lw=2.0, ls='-.', zorder=6,
            label=f'HeFESTo best (misfit={hf["misfit"]:.2f})')
    if has_hf_bml:
        ax.plot(hf_bml_rho, hf_bml_depth, color=C_HF, lw=1.8, ls=':', zorder=6)

mark_bml(ax)
ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_ylim(*FIG1_YLIM)
ax.set_xlim(3.6, 6)
ax.set_xlabel('ρ (g/cm³)', fontsize=13)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.tick_params(labelsize=14)
ax.legend(fontsize=8, loc='upper left', framealpha=0.85)
for sp in ax.spines.values(): sp.set_linewidth(2.0)

title_str = (
    f'Mars Mantle+BML velocity & density: nGibbs (top {len(ng_profiles)}) '
    f'vs Samuel 2023 vs Khan 2023\n'
    f"nGibbs best : {os.path.basename(best_ng['chain_dir'])}  "
    f"step {best_ng['step']}  misfit={best_ng['misfit']:.4f}  "
    f"|  {fmt(best_ng['params'])}"
)
if PLOT_HEFESTO and hf is not None and hf['chain_dir'] is not None:
    title_str += (f"\nHeFESTo best: {os.path.basename(hf['chain_dir'])}  "
                  f"step {hf['step']}  misfit={hf['misfit']:.4f}  "
                  f"|  {fmt(hf['params'])}")
fig.suptitle(title_str, fontsize=9.5, y=1.01)
plt.tight_layout()
if SAVE_FIG:
    out1 = os.path.join(OUT_DIR,
                        f'best_fit_{CHAIN_PREFIX}_top{TOP_N}_vp_vs_rho.png')
    plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out1}")
plt.show()


# ── PLOT: Figure 2 (Temperature, standalone) ───────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 10))
fig2.patch.set_facecolor('#FAFAFA')

ax.plot(sam_T,   r2d(dep_T),   color='#2980B9', lw=2.0, label='Samuel 2023 T')
ax.plot(sam_sol, r2d(dep_sol), color='orange',  lw=1.5, ls='--', label='Solidus')
ax.plot(sam_liq, r2d(dep_liq), color='brown',   lw=1.5, ls='--', label='Liquidus')

T_med, T_p16, T_p84 = khan_stats(khan_T)
ax.fill_betweenx(Z, T_p16, T_p84, color='#4198B9', alpha=0.15, label='Khan T 16-84%')
ax.plot(T_med, Z, color='#4198B9', lw=1.5, ls='-', label='Khan T median')


def full_T_depth(prof, with_core=False):
    """地函 + BML(可選 + 核)的連續溫度剖面。兩種 npz 格式共用。"""
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


for p in ng_profiles[1:]:
    T_full, d_full = full_T_depth(p['prof'])
    ax.plot(T_full, d_full, color='#E67E22', lw=0.8, alpha=0.25, zorder=4)

T_best, d_best_T = full_T_depth(best_prof)
ax.plot(T_best, d_best_T, color='#E67E22', lw=2.5, zorder=5, label='nGibbs best T')

if PLOT_HEFESTO and hf_man is not None:
    hf_T_full = np.concatenate([hf_man['T'], hf_bml_T]) if has_hf_bml else hf_man['T']
    hf_d_full = (np.concatenate([hf_man['depth'], hf_bml_depth]) if has_hf_bml
                 else hf_man['depth'])
    ax.plot(hf_T_full, hf_d_full, color=C_HF, lw=2.0, ls='-.', zorder=6,
            label='HeFESTo T')
    if has_hf_bml:
        ax.axhline(hf_oc, color=C_HF, ls=':', lw=1.5, alpha=0.8)
        ax.text(0.02, hf_oc+10, f'HeFESTo outer core\n{hf_oc:.0f} km',
                transform=ax.get_yaxis_transform(), fontsize=7, color=C_HF, va='top')

if has_bml:
    if z_cmb is not None:
        ax.axhline(z_cmb, color='#E67E22', ls=':', lw=1.5, alpha=0.8)
        ax.text(0.02, z_cmb - 20, f'CMB (core top)\n{z_cmb:.0f} km',
                transform=ax.get_yaxis_transform(), fontsize=7, color='#E67E22',
                va='bottom')
    if z_sl is not None:
        ax.axhline(z_sl, color='#E67E22', ls='-.', lw=1.2, alpha=0.7)
        ax.text(0.02, z_sl + 10, f'BML solid/liquid\n{z_sl:.0f} km',
                transform=ax.get_yaxis_transform(), fontsize=7, color='#E67E22',
                va='top')
    if z_bml_top is not None:
        ax.axhline(z_bml_top, color='orange', ls='--', lw=1.0, alpha=0.6)
    if z_cmb is not None:
        ax.axhline(z_cmb, color='orange', ls='--', lw=1.0, alpha=0.6)

ax.axhline(SPLICE_DEPTH, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_ylim(3300, 0)
ax.set_xlim(0, 3800)
ax.set_ylabel('Depth (km)', fontsize=13)
ax.set_xlabel('T (K)', fontsize=13)
ax.grid(True, ls='--', alpha=0.3, lw=0.8)
ax.tick_params(labelsize=14)
ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
for sp in ax.spines.values(): sp.set_linewidth(2.0)

fig2.suptitle('Mars Mantle+BML Temperature: nGibbs vs Samuel 2023 vs Khan 2023',
              fontsize=11, y=1.01)
plt.tight_layout()
if SAVE_FIG:
    out2 = os.path.join(OUT_DIR,
                        f'best_fit_{CHAIN_PREFIX}_top{TOP_N}_temperature.png')
    plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"Saved: {out2}")
plt.show()