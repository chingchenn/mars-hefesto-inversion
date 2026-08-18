#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core_fes.py —— Fe-S 液態核 EoS(查表加速版),供 70_mcmc 使用

兩條對等的路線,由 configure() 選擇。沒有基準線:各自跑一條 MCMC,
比較兩組後驗,差異就是結論的 EoS 依賴度。

  'A'  Liu & Asimow 2025 的 Fe 端元 + S 端元 + Gamma 活度混合
       內部一致:同一套 FPMD、同一組壓力修正、同一個混合模型
       Fe-S 的校準域是 S <= 16.7 mol%(Fe90S18)且資料在 >= 90 GPa,
       火星壓力與高 S 都在校準域外

  'B'  Liu 的 Fe 端元 + Sakai & Hirose 2026a 的 FeS 端元 + Margules 混合
       兩個端元各自落在火星壓力範圍的實測內(Fe: Kuwayama 21-116 GPa;
       FeS: Sakai 25-55 GPa),但端元與混合律來自不同框架


混合的形式差異(A 與 B 不能共用同一份表的原因):
  Margules  體積可加         V = sum x_i V_i + V_ex          -> 端元各建一張 2D 表
  Liu       等體積下壓力可加  P = sum n_i Gamma_i P_i(V,T)    -> V 必須根搜
                                                              -> 直接對混合物建 3D 表

外部資料的身分:Kuwayama 與 Sakai 在 selftest 裡只「記錄」不「判定」。
兩條路線對同一組外部資料的吻合度是各自建構方式的性質:
  B 的 FeS 吻合是建構的結果(端元就是從 Sakai 擬合而來),不是獨立驗證
  A 的 FeS 偏差是外插的結果
兩句都寫進 methods,不選邊。PASS/FAIL 只留給內部正確性。

介面(70_mcmc 用到的):
    configure(route, s_reading)                            -> 設定路線並備妥表格
    core_properties(P, T, w_S)                             -> dict of arrays
    find_R_IC(prof)                                        -> R_ic (km)
    gap_min(prof)                                          -> min(T - T_liquidus) (K)
    T_peritectic(P)                                        -> K
    selftest() / compare_routes()

表格範圍:P 10-70 GPa、T 1200-4800 K、chi_FeS 0-1。
表格存成 npz 快取,同一組設定第二次 import 是秒讀。
"""
import os
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

_trapz = getattr(np, 'trapezoid', None) or np.trapz

G_GRAV = 6.674e-11
R_GAS  = 8.314462
M_FE, M_S = 55.845, 32.060
M_FES  = M_FE + M_S
N108   = 6.02214076e23/108.0        # 5.5760e21,Liu Table 1 的換算基準

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 路線設定
# ══════════════════════════════════════════════════════════════════════════════
ROUTES = {
    'A': dict(fe='liu',      fes=None,    mix='liu',      tag='Liu Fe + Liu S (Gamma)'),
    'B': dict(fe='liu',      fes='sakai', mix='margules', tag='Liu Fe + Sakai FeS (Margules)'),
    'L': dict(fe='kuwayama', fes='sakai', mix='margules', tag='Kuwayama Fe + Sakai FeS (a35)'),
}

ROUTE         = 'A'
LIU_S_READING = 'B'   # Liu Table 1 的 S 列身分(待 Weiyi 確認):
                      #   'B' = virtual pure-S endmember,直接用
                      #   'A' = Fe90S18 binary 的擬合,需先反解出 endmember
IDEAL_MIXING  = False # 只影響 margules 路線:True 則關掉過剩體積

def _cfg(): return ROUTES[ROUTE]

# ══════════════════════════════════════════════════════════════════════════════
# 組成換算
#   1 mol 混合物 = (1-chi) mol Fe + chi mol FeS
#   原子數:Fe = 1、S = chi,總計 1+chi  ->  n_S = chi/(1+chi)
#   w_S = chi*M_S/(M_FE + chi*M_S)  ->  chi = M_FE*w_S/(M_S*(1-w_S))
# ══════════════════════════════════════════════════════════════════════════════
def wS_to_chi(w_S):
    w_S = float(np.clip(w_S, 0.0, 0.36485))
    return M_FE*w_S/(M_S*(1.0-w_S))

def wS_to_atS(w_S):
    nS, nFe = w_S/M_S, (1.0-w_S)/M_FE
    return 100.0*nS/(nS+nFe)

# ══════════════════════════════════════════════════════════════════════════════
# 端元 EoS
#   MG   Mie-Grueneisen        P = P_vinet(V) + gamma(V)/V*[E(T)-E(T0)]
#   AG   Anderson-Grueneisen   V(P,T) 由 dV/dT|_P = alpha(V)*V 積分
#   LIU  Liu & Asimow 2025     P = BM3(f) + Cv*(a_g + b_g/V)*(T-T0)
#                              gamma = a_g*V + b_g 線性,Cv 常數(Dulong-Petit)
#
# 混合在體積層次,所以不同端元用不同形式沒問題;
# 但同一端元不可混用兩篇的參數。
# ══════════════════════════════════════════════════════════════════════════════
def _vinet(V, V0, K0, Kp):
    x = V/V0; x13 = x**(1.0/3.0)
    return 3*K0*x**(-2.0/3.0)*(1.0-x13)*np.exp(1.5*(Kp-1.0)*(1.0-x13))

# liquid Fe — Kuwayama et al. 2020 PRL 124, 165701
#   火星核 19-56 GPa 完全在量測範圍(21.5-116 GPa)內
FE_KUW = dict(form='MG', M=M_FE, rho0=7.03, T0=1811.0, K_T0=82.1, Kp0=5.80,
              gam0=2.02, b=0.63, e0=0.68e-4, g=-1.0, n=1.0)
FE_KUW['V0'] = FE_KUW['M']/FE_KUW['rho0']

# liquid Fe — Liu & Asimow 2025 JGR-SE Table 1(Fe108 列)
#   單位換算:V0 [A^3] x 5.5760e-3 -> cm3/mol
#             Cv [J/K] x 5.5760e21 -> J/K/mol
#             a_g [A^-3] x 179.34  -> (cm3/mol)^-1
#   V0/K_T0/Kp0 是 T0 = 4000 K 參考態的擬合參數,不是物理量,勿單獨解讀
FE_LIU = dict(form='LIU', M=M_FE, T0=4000.0,
              V0=2690.0*5.5760e-3, K_T0=1.20, Kp0=18.7,
              Cv=7.28e-21*5.5760e21, a_gam=8.05e-4*179.34, b_gam=2.0/3.0, n=1.0)

# virtual pure-S endmember — Liu & Asimow 2025 Table 1(S 列,參考組成 Fe90S18)
S_LIU  = dict(form='LIU', M=M_S, T0=4000.0,
              V0=3170.0*5.5760e-3, K_T0=0.26, Kp0=40.9,
              Cv=6.85e-21*5.5760e21, a_gam=9.60e-4*179.34, b_gam=2.0/3.0, n=1.0)

D_GAM_FE, D_GAM_S = 0.382, 1.0        # Liu eq 22 的 d_Gamma

# liquid FeS — Sakai & Hirose 2026a JGR Planets
#   25-55 GPa / 2320-3260 K 直接實測密度,涵蓋火星核
#   只擬合密度、沒擬合聲速 -> K_S 是外推,Vp 的可靠度低於 rho
#   沒給 Cp,沿用 Samuel 的 62.5(唯一跨論文借用,對結果不敏感)
FES_SAK = dict(form='AG', M=M_FES, V0=M_FES/3.625, T0=1650.0, K_T0=23.8,
               Kp0=4.1, a0=11.8e-5, d0=2.7, kappa=1.4, Cp=62.5, n=2.0)

# liquid FeS — Samuel 2023 SM Table 4(= Xu 2021 + Morard 2013 低壓外插)
#   delta_T = 0.4 會讓 gamma 在壓縮下上升;保留僅供 delta_0 敏感度測試
FES_SAM = dict(form='AG', M=M_FES, V0=24.4, T0=1650.0, K_T0=12.0,
               Kp0=6.9, a0=11.8e-5, d0=0.4, kappa=1.4, Cp=62.5, n=2.0)

_FE_TABLE  = {'kuwayama': FE_KUW, 'liu': FE_LIU}
_FES_TABLE = {'sakai': FES_SAK, 'samuel': FES_SAM}

def FE():  return _FE_TABLE[_cfg()['fe']]
def FES(): return _FES_TABLE[_cfg()['fes']] if _cfg()['fes'] else None

# ── 封閉形式的 P(V,T) ────────────────────────────────────────────────────────
def _P_MG(V, T, em):
    x  = V/em['V0']
    P0 = _vinet(V, em['V0'], em['K_T0'], em['Kp0'])
    e  = em['e0']*x**em['g']
    E  = lambda TT: 3*em['n']*R_GAS*(TT + e*TT*TT)
    return P0 + (em['gam0']*x**em['b'])/(V*1e-6)*(E(T)-E(em['T0']))/1e9

def _P_LIU(V, T, em):
    f   = 0.5*((em['V0']/V)**(2.0/3.0) - 1.0)
    Pc  = 3*em['K_T0']*f*(1.0+2.0*f)**2.5*(1.0 + 1.5*(em['Kp0']-4.0)*f)
    gam = em['a_gam']*V + em['b_gam']
    return Pc + gam*em['Cv']*(T-em['T0'])/(V*1e-6)/1e9

def _alpha_AG(V, em):
    return em['a0']*np.exp(-(em['d0']/em['kappa'])*(1.0-(V/em['V0'])**em['kappa']))

# ── Liu 混合(等體積等溫下壓力可加)────────────────────────────────────────
def _gam_act(d, n):                     # Liu eq 22:Gamma_i = d_i + (1-d_i) n_i
    return d + (1.0-d)*n

def _P_S_end(V, T):
    """LIU_S_READING='B':Table 1 的 S 列就是 virtual endmember
       LIU_S_READING='A':S 列是 Fe90S18 binary 的擬合,反解出 endmember
         P_bin = nFe' Gamma_Fe(nFe') P_Fe + nS' P_S_end,  nFe' = 90/108"""
    if LIU_S_READING == 'B':
        return _P_LIU(V, T, S_LIU)
    nFe_b, nS_b = 90.0/108.0, 18.0/108.0
    return (_P_LIU(V, T, S_LIU)
            - nFe_b*_gam_act(D_GAM_FE, nFe_b)*_P_LIU(V, T, FE_LIU))/nS_b

def _P_mix_liu(V_atom, T, n_S):
    """V_atom = 每 mol 原子的體積 (cm3/mol-atom)"""
    n_Fe = 1.0 - n_S
    return (n_Fe*_gam_act(D_GAM_FE, n_Fe)*_P_LIU(V_atom, T, FE_LIU)
            + n_S *_gam_act(D_GAM_S,  n_S )*_P_S_end(V_atom, T))

# ── 精確解(建表、selftest 用;MCMC 執行期不呼叫)────────────────────────
def _root_largest_V(func, V_lo, V_hi, nscan=400):
    """負熱壓下 P(V) 可能非單調,取最大 V 的根(物理分支)"""
    Vs = np.linspace(V_lo, V_hi, nscan)
    Ps = np.array([func(v) for v in Vs])
    good = np.isfinite(Ps)
    if good.sum() < 2:
        return np.nan
    Vg, Pg = Vs[good], Ps[good]
    idx = np.where(np.diff(np.sign(Pg)) != 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    return brentq(func, Vg[i], Vg[i+1], xtol=1e-12)

def V_endmember_exact(P, T, em):
    if not (np.isfinite(P) and np.isfinite(T)) or P <= 0:
        return np.nan
    if em['form'] == 'MG':
        f = lambda V: _P_MG(V, T, em) - P
        lo, hi = em['V0']*0.25, em['V0']*6.0   # 低P x 極高T 的表格角落需要大 V
        if f(lo)*f(hi) > 0: return np.nan
        return brentq(f, lo, hi, xtol=1e-12)
    if em['form'] == 'LIU':
        return _root_largest_V(lambda V: _P_LIU(V, T, em) - P,
                               em['V0']*0.18, em['V0']*1.15)
    f = lambda V: _vinet(V, em['V0'], em['K_T0'], em['Kp0']) - P
    lo, hi = em['V0']*0.25, em['V0']*4.0
    if f(lo)*f(hi) > 0: return np.nan
    V_T0 = brentq(f, lo, hi, xtol=1e-13)
    sol  = solve_ivp(lambda t, y: [_alpha_AG(y[0], em)*y[0]],
                     [em['T0'], T], [V_T0], rtol=1e-10, atol=1e-12)
    return float(sol.y[0][-1])

# ══════════════════════════════════════════════════════════════════════════════
# Margules 混合(Samuel 2023 SM Eq.12,出處 Irving 2023;W(P) 出自 Xu 2021)
#   V = (1-chi)V_Fe + chi V_FeS + chi(1-chi)[chi W_Fe-FeS + (1-chi) W_FeS-Fe]
#   只有體積是廣延量可加;密度與模數是強度量,不可平均
# ══════════════════════════════════════════════════════════════════════════════
W_FE_FES, W_FES_FE = -9.9, -3.54
B0_MARG,  BP_MARG  = 3.02, 2.6

def W_of_P(w, P):
    return w*np.exp((1.0/BP_MARG)*(1.0-np.sqrt(1.0+2.0*(BP_MARG/B0_MARG)*P)))

def V_mix_exact(P, T, chi):
    """精確混合體積 (cm3/mol 混合物);selftest 的基準"""
    if _cfg()['mix'] == 'liu':
        n_S  = chi/(1.0+chi)
        V_at = _root_largest_V(lambda V: _P_mix_liu(V, T, n_S) - P,
                               FE_LIU['V0']*0.18, FE_LIU['V0']*1.25)
        return V_at*(1.0+chi) if np.isfinite(V_at) else np.nan
    vFe  = V_endmember_exact(P, T, FE())
    vFeS = V_endmember_exact(P, T, FES())
    if not (np.isfinite(vFe) and np.isfinite(vFeS)):
        return np.nan
    V = (1.0-chi)*vFe + chi*vFeS
    if not IDEAL_MIXING:
        V += (1.0-chi)*chi*(chi*W_of_P(W_FE_FES, P) + (1.0-chi)*W_of_P(W_FES_FE, P))
    return V

# ══════════════════════════════════════════════════════════════════════════════
# 查表
#   margules 路線:V_Fe(P,T)、V_FeS(P,T) 兩張 2D 表,執行期相加
#   liu      路線:V_mix(chi,P,T) 一張 3D 表(混合是壓力可加,不能事後相加)
# ══════════════════════════════════════════════════════════════════════════════
P_GRID   = np.arange(10.0, 70.01, 0.5)        # GPa
T_GRID   = np.arange(1200.0, 4800.01, 25.0)   # K,容納 T_core 先驗上限 + 絕熱增溫 + 迭代暫態
CHI_GRID = np.linspace(0.0, 1.0, 21)          # 含純 FeS,供 selftest 的 Sakai 對照

_ITP_FE = _ITP_FES = _ITP_MIX = None
_TABLES_FOR = None          # (route, s_reading) 的標記

def _invert_PV(Pfun, T, V_lo, V_hi, nV=4000):
    """給定封閉形式 P(V) 與溫度 T,在 V 網格上求 P,取單調遞減段後
    以 np.interp 反解出 P_GRID 對應的 V。純向量化,不做逐點根搜。"""
    Vg = np.linspace(V_lo, V_hi, nV)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        Pv = Pfun(Vg, T)
    good = np.isfinite(Pv)
    if good.sum() < 3:
        return np.full(len(P_GRID), np.nan)
    Vg, Pv = Vg[good], Pv[good]
    d   = np.diff(Pv)
    bad = np.where(d >= 0)[0]              # 非單調處截斷(物理分支只到這裡)
    end = int(bad[0]) + 1 if len(bad) else len(Vg)
    if end < 3:
        return np.full(len(P_GRID), np.nan)
    Vs, Ps = Vg[:end][::-1], Pv[:end][::-1]   # Ps 遞增
    return np.interp(P_GRID, Ps, Vs, left=np.nan, right=np.nan)

def _build_table_closed(Pfun, V_lo, V_hi):
    """MG / LIU 這類封閉形式:每個 T 一次向量化反解"""
    tab = np.empty((len(P_GRID), len(T_GRID)))
    for j, T in enumerate(T_GRID):
        tab[:, j] = _invert_PV(lambda V, TT=T: Pfun(V, TT), T, V_lo, V_hi)
    return tab

def _build_table_AG(em):
    """AG 形式:每個等壓線只解一次 ODE(dense output 到整條 T 軸)"""
    tab = np.empty((len(P_GRID), len(T_GRID)))
    for i, P in enumerate(P_GRID):
        f = lambda V: _vinet(V, em['V0'], em['K_T0'], em['Kp0']) - P
        V_T0 = brentq(f, em['V0']*0.25, em['V0']*4.0, xtol=1e-13)
        rhs  = lambda t, y: [_alpha_AG(y[0], em)*y[0]]
        up   = T_GRID[T_GRID >= em['T0']]
        dn   = T_GRID[T_GRID <  em['T0']][::-1]
        if len(up):
            s = solve_ivp(rhs, [em['T0'], up[-1]], [V_T0], t_eval=up,
                          rtol=1e-10, atol=1e-12)
            tab[i, T_GRID >= em['T0']] = s.y[0]
        if len(dn):
            s = solve_ivp(rhs, [em['T0'], dn[-1]], [V_T0], t_eval=dn,
                          rtol=1e-10, atol=1e-12)
            tab[i, T_GRID < em['T0']] = s.y[0][::-1]
    return tab

def _build_table_endmember(em):
    if em['form'] == 'MG':
        return _build_table_closed(lambda V, T: _P_MG(V, T, em),
                                   em['V0']*0.20, em['V0']*6.0)
    if em['form'] == 'LIU':
        return _build_table_closed(lambda V, T: _P_LIU(V, T, em),
                                   em['V0']*0.15, em['V0']*1.20)
    return _build_table_AG(em)

def _build_table_mix_liu():
    """3D:(chi, P, T)。每個 (chi, T) 一次向量化反解。
    存的是「每 mol 混合物」的體積,已乘上 (1+chi)。"""
    tab = np.empty((len(CHI_GRID), len(P_GRID), len(T_GRID)))
    V_lo, V_hi = FE_LIU['V0']*0.15, FE_LIU['V0']*1.30
    for k, chi in enumerate(CHI_GRID):
        n_S = chi/(1.0+chi)
        for j, T in enumerate(T_GRID):
            V_at = _invert_PV(lambda V, TT=T, nS=n_S: _P_mix_liu(V, TT, nS),
                              T, V_lo, V_hi)
            tab[k, :, j] = V_at*(1.0+chi)
    return tab

def _fill_nan_along_T(tab, name):
    """真實剖面不會踩到的表格角落(低P x 極高T)若仍是 NaN,沿 T 軸就近填值,
    只為讓 cubic spline 可建;實際 (P,T) 落在這些角落機率為零。"""
    n_bad = int(np.sum(~np.isfinite(tab)))
    if n_bad:
        print(f'core_fes: {name}: filling {n_bad} unreachable NaN cells', flush=True)
        flat = tab.reshape(-1, tab.shape[-1])
        for row in flat:
            bad = ~np.isfinite(row)
            if bad.any() and (~bad).any():
                row[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad),
                                     row[~bad])
    return tab

def _cache_path():
    return os.path.join(CACHE_DIR,
                        f'core_fes_tab_{ROUTE}_{LIU_S_READING}_'
                        f'{len(P_GRID)}x{len(T_GRID)}x{len(CHI_GRID)}.npz')

def _build_tables(verbose=True):
    global _ITP_FE, _ITP_FES, _ITP_MIX, _TABLES_FOR
    cache = _cache_path()
    tabs  = None
    if os.path.exists(cache):
        try:
            z = np.load(cache)
            tabs = {k: z[k] for k in z.files}
            if verbose:
                print(f'core_fes: loaded tables from {os.path.basename(cache)}',
                      flush=True)
        except Exception:
            tabs = None
    if tabs is None:
        if verbose:
            print(f'core_fes: building tables for route {ROUTE} '
                  f'({_cfg()["tag"]}) ...', flush=True)
        tabs = {}
        if _cfg()['mix'] == 'liu':
            tabs['mix'] = _fill_nan_along_T(_build_table_mix_liu(), 'V_mix')
        else:
            tabs['fe']  = _fill_nan_along_T(_build_table_endmember(FE()),  'V_Fe')
            tabs['fes'] = _fill_nan_along_T(_build_table_endmember(FES()), 'V_FeS')
        try:
            np.savez_compressed(cache, **tabs)
        except Exception as e:
            print(f'core_fes: cache write failed ({e}), continuing', flush=True)

    _ITP_FE = _ITP_FES = _ITP_MIX = None
    if 'mix' in tabs:
        _ITP_MIX = RegularGridInterpolator((CHI_GRID, P_GRID, T_GRID), tabs['mix'],
                                           method='cubic', bounds_error=False,
                                           fill_value=np.nan)
    else:
        _ITP_FE  = RegularGridInterpolator((P_GRID, T_GRID), tabs['fe'],
                                           method='cubic', bounds_error=False,
                                           fill_value=np.nan)
        _ITP_FES = RegularGridInterpolator((P_GRID, T_GRID), tabs['fes'],
                                           method='cubic', bounds_error=False,
                                           fill_value=np.nan)
    _TABLES_FOR = (ROUTE, LIU_S_READING)
    if verbose:
        print(f'core_fes: route {ROUTE} ready  '
              f'(P {P_GRID[0]:.0f}-{P_GRID[-1]:.0f} GPa, '
              f'T {T_GRID[0]:.0f}-{T_GRID[-1]:.0f} K, '
              f'chi {CHI_GRID[0]:.2f}-{CHI_GRID[-1]:.2f})', flush=True)

def configure(route=None, s_reading=None, ideal_mixing=None, verbose=True):
    """70_mcmc 在 import 之後呼叫一次。表格是 lazy build,設定變了才重建。"""
    global ROUTE, LIU_S_READING, IDEAL_MIXING
    if route is not None:
        if route not in ROUTES:
            raise ValueError(f'unknown route {route!r}, expected one of {list(ROUTES)}')
        ROUTE = route
    if s_reading is not None:
        LIU_S_READING = s_reading
    if ideal_mixing is not None:
        IDEAL_MIXING = ideal_mixing
    _build_tables(verbose=verbose)

def _ensure_tables():
    if _TABLES_FOR != (ROUTE, LIU_S_READING):
        _build_tables(verbose=True)

# ══════════════════════════════════════════════════════════════════════════════
# 混合體積與熱容(執行期只走查表)
# ══════════════════════════════════════════════════════════════════════════════
def _V_mix_arr(P, T, chi):
    """1 mol 混合物的體積 (cm3/mol);P, T 同形狀陣列"""
    _ensure_tables()
    shp = np.shape(P)
    if _ITP_MIX is not None:
        chi_a = np.full(np.size(P), float(chi))
        pts = np.stack([chi_a, np.ravel(P), np.ravel(T)], axis=-1)
        return _ITP_MIX(pts).reshape(shp)
    pts  = np.stack([np.ravel(P), np.ravel(T)], axis=-1)
    vFe  = _ITP_FE(pts).reshape(shp)
    vFeS = _ITP_FES(pts).reshape(shp)
    V = (1.0-chi)*vFe + chi*vFeS
    if not IDEAL_MIXING:
        Pa = np.asarray(P, float)
        V = V + (1.0-chi)*chi*(chi*W_of_P(W_FE_FES, Pa)
                               + (1.0-chi)*W_of_P(W_FES_FE, Pa))
    return V

def _Cv_mix(P, T, chi):
    """每 mol 混合物的 Cv (J/K/mol)

    LIU 端元:Cv 為常數(Dulong-Petit),Liu eq 19 的混合同樣是 n_i Gamma_i 加權
    MG  端元:古典均分 + Sommerfeld 電子項,隨 V, T 變
    AG  端元:Cp - Cv = alpha^2 K_T V T,只在參考態換算一次。
              若每個 (P,T) 重算會得到低於 Dulong-Petit 的 Cv,物理上不可能。
    """
    _ensure_tables()
    if _cfg()['mix'] == 'liu':
        n_S  = chi/(1.0+chi); n_Fe = 1.0-n_S
        cv_atom = (n_Fe*_gam_act(D_GAM_FE, n_Fe)*FE_LIU['Cv']
                   + n_S *_gam_act(D_GAM_S,  n_S )*S_LIU['Cv'])
        return np.full(np.shape(P), cv_atom*(1.0+chi))

    fe = FE()
    if fe['form'] == 'LIU':
        cv_fe = np.full(np.shape(P), fe['Cv'])
    else:
        pts = np.stack([np.ravel(P), np.ravel(T)], axis=-1)
        vFe = _ITP_FE(pts).reshape(np.shape(P))
        x   = vFe/fe['V0']
        cv_fe = 3*fe['n']*R_GAS*(1.0 + 2.0*fe['e0']*x**fe['g']*np.asarray(T))

    em = FES()
    cv_fes = em['Cp'] - em['a0']**2*(em['K_T0']*1e9)*(em['V0']*1e-6)*em['T0']
    return (1.0-chi)*cv_fe + chi*cv_fes

# ══════════════════════════════════════════════════════════════════════════════
# 數值微分 + 熱力學鏈
#   K_T = -V(dP/dV)_T ;  alpha = (1/V)(dV/dT)_P
#   gamma = alpha K_T V/Cv ;  K_S = K_T(1 + alpha gamma T)
#   Vp = sqrt(K_S/rho) ;  dT/dP|_S = gamma T/K_S
# ══════════════════════════════════════════════════════════════════════════════
def core_properties(P, T, w_S, dP=0.1, dT=10.0):
    """向量化:P, T 可為陣列。回傳 dict of arrays。
    dP/dT 取表格解析度的量級,太小會放大內插雜訊。"""
    P = np.asarray(P, float); T = np.asarray(T, float)
    chi   = wS_to_chi(w_S)
    M_mix = (1.0-chi)*M_FE + chi*M_FES
    V   = _V_mix_arr(P,    T,    chi)
    Vpp = _V_mix_arr(P+dP, T,    chi); Vpm = _V_mix_arr(P-dP, T, chi)
    Vtp = _V_mix_arr(P,    T+dT, chi); Vtm = _V_mix_arr(P, T-dT, chi)
    rho   = M_mix/V
    K_T   = -V*(2*dP)/(Vpp-Vpm)
    alpha = (Vtp-Vtm)/(2*dT)/V
    cv    = _Cv_mix(P, T, chi)
    gamma = alpha*(K_T*1e9)*(V*1e-6)/cv
    K_S   = K_T*(1.0+alpha*gamma*T)
    Vp    = np.sqrt(np.clip(K_S, 0, None)*1e9/(rho*1e3))/1000.0
    return dict(rho=rho, Vp=Vp, K_T=K_T, K_S=K_S, alpha=alpha,
                gamma=gamma, dTdP=gamma*T/K_S, V=V)

# ══════════════════════════════════════════════════════════════════════════════
# 核剖面自洽積分
#   M(r) = int 4 pi r^2 rho dr ;  g = GM/r^2 ;  dP/dr = -rho g
#   dT/dP = gamma T/K_S(絕熱)
#   中心壓力對組成極敏感(純 FeS ~33、純 Fe ~50 GPa),不能用固定壓力場
# ══════════════════════════════════════════════════════════════════════════════
def build_core_profile(T_core, w_S, R_cmb_km=1646.2, P_cmb=22.0,
                       n=60, n_iter=40, tol=1.0, verbose=False):
    r   = np.linspace(R_cmb_km*1e3, 1.0, n)
    dr  = np.diff(r)
    rho = np.full(n, 6500.0); P = np.full(n, P_cmb); T = np.full(n, T_core)
    for it in range(n_iter):
        ra, rhoa = r[::-1], rho[::-1]
        M = np.zeros(n)
        rm  = 0.5*(ra[1:]+ra[:-1]); rhm = 0.5*(rhoa[1:]+rhoa[:-1])
        M[1:] = np.cumsum(4*np.pi/3*rhm*(ra[1:]**3 - ra[:-1]**3))
        M = M[::-1]
        g = np.zeros(n); g[r > 1.0] = G_GRAV*M[r > 1.0]/r[r > 1.0]**2
        P[0], T[0] = P_cmb, T_core
        dTdP = core_properties(P, T, w_S)['dTdP']
        dTdP = np.where(np.isfinite(dTdP), dTdP, 0.0)
        w = rho*g
        for i in range(1, n):
            P[i] = P[i-1] - 0.5*(w[i] + w[i-1])*dr[i-1]/1e9
            T[i] = T[i-1] + dTdP[i-1]*(P[i]-P[i-1])
        props   = core_properties(P, T, w_S)
        rho_new = props['rho']*1e3
        if not np.all(np.isfinite(rho_new)):
            return None                          # 超出表格範圍 -> 70 端 reject
        dmax = float(np.max(np.abs(rho_new-rho)))
        rho  = 0.5*rho + 0.5*rho_new
        if verbose:
            print(f'  iter {it+1:2d}: max|drho|={dmax:8.2f}  '
                  f'P_c={P[-1]:6.2f}  T_c={T[-1]:6.0f}')
        if dmax < tol:
            break
    props = core_properties(P, T, w_S)
    if not np.all(np.isfinite(props['Vp'])):
        return None
    return dict(r=r, P=P, T=T, rho=rho, Vp=props['Vp'], g=g, M=M,
                P_center=float(P[-1]), T_center=float(T[-1]), n_iter=it+1,
                rho_mean=float(3*_trapz(rho[::-1]*r[::-1]**2, r[::-1])/r[0]**3))

# ══════════════════════════════════════════════════════════════════════════════
# 液相線與內核
# ══════════════════════════════════════════════════════════════════════════════
def T_peritectic(P):
    """Sakai & Hirose 2026b Eq.2,Fe12S7 peritectic;定義域 P >= 21 GPa
    驗證:T_peritectic(175) 應約 2990 K(實測 3010 +- 190 K)"""
    return 1473.0*((np.asarray(P, float)-21.0)/3.2 + 1.0)**(1.0/5.5)

def T_eutectic(P):
    """Sakai & Hirose 2026b Eq.2,Fe2S solidus"""
    return 1373.0*((np.asarray(P, float)-21.0)/5.3 + 1.0)**(1.0/5.1)

def gap_min(prof, P_valid=27.0):
    """min(T_adiabat - T_liquidus) 沿核剖面 (K)。

    <= 0 表示絕熱線碰到液相線 -> 存在內核。
    平滑的觀測量,適合直接進 likelihood(取代對 R_ic 數值本身的比對);
    R_ic 由 find_R_IC 記錄,當後驗預測用,不進 misfit。
    液相線僅在 P > P_valid 有效(Fe12S7 場)。"""
    if prof is None:
        return np.nan
    P, T = prof['P'], prof['T']
    with np.errstate(invalid='ignore'):
        d = np.where(P > P_valid, T - T_peritectic(P), np.nan)
    return float(np.nanmin(d)) if np.isfinite(d).any() else np.nan

def find_R_IC(prof, P_valid=27.0):
    """絕熱線與液相線最淺交點的半徑 (km);無交點回傳 0。
    靜態溫度計:不含固核密度回饋與結晶分異。"""
    if prof is None:
        return 0.0
    P, T, r = prof['P'], prof['T'], prof['r']/1e3
    with np.errstate(invalid='ignore'):
        Tl = np.where(P > P_valid, T_peritectic(P), np.nan)
    sub = np.isfinite(Tl) & (T < Tl)
    if not sub.any():
        return 0.0
    i = int(np.argmax(sub))
    if i == 0:
        return float(r[0])
    f0, f1 = T[i-1]-Tl[i-1], T[i]-Tl[i]
    if not (np.isfinite(f0) and np.isfinite(f1)) or f0 == f1:
        return float(r[i])
    return float(r[i-1] + f0/(f0-f1)*(r[i]-r[i-1]))

# ══════════════════════════════════════════════════════════════════════════════
# selftest / compare_routes
# ══════════════════════════════════════════════════════════════════════════════
# Sakai & Hirose 2026a Table 1:P (GPa), T_average (K), rho (kg/m3)
SAKAI_OBS = [(24.9, 2320, 5281), (35.1, 3040, 5775), (40.0, 2850, 6155),
             (48.8, 2400, 6501), (54.9, 3260, 6588), (53.3, 2710, 6799)]

# Kuwayama 2020 Table SII 抽點:P (GPa), T (K), rho (g/cm3)
KUW_OBS = [(20, 2000, 8.149), (40, 2000, 8.928), (20, 3000, 7.688),
           (40, 3000, 8.582), (25, 4000, 7.408), (55, 4000, 8.729)]

def selftest(verbose=True):
    """PASS/FAIL 只判內部正確性:查表 vs 精確解、剖面收斂、速度。

    對 Kuwayama(純 Fe)與 Sakai(純 FeS)的殘差只「記錄」,不判定。
    兩條路線對同一組外部資料的吻合度是各自建構方式的性質:
      B 的 FeS 吻合是建構的結果(端元就是從 Sakai 擬合而來),不是獨立驗證
      A 的 FeS 偏差是外插的結果(校準域 S <= 16.7 mol%、P >= 90 GPa)
    兩句都寫進 methods,不選邊。
    """
    import time
    _ensure_tables()
    rng = np.random.default_rng(0)
    chi = wS_to_chi(0.17)

    # (1) 查表 vs 精確解 —— 這才是真的對錯
    errs = []
    for _ in range(40):
        P = rng.uniform(15, 60); T = rng.uniform(1400, 3400)
        V_e = V_mix_exact(P, T, chi)
        V_t = float(_V_mix_arr(np.array([P]), np.array([T]), chi)[0])
        if np.isfinite(V_e) and np.isfinite(V_t):
            errs.append(abs(V_t-V_e)/V_e)
    err_max = max(errs) if errs else np.nan

    # (2) 外部比對 —— 只記錄
    d_fe = []
    for P, T, o in KUW_OBS:
        V = float(_V_mix_arr(np.array([P]), np.array([T]), 0.0)[0])
        d_fe.append(100*(M_FE/V - o)/o if np.isfinite(V) else np.nan)
    rms_fe = float(np.sqrt(np.nanmean(np.array(d_fe)**2)))

    d_fes = []
    for P, T, o in SAKAI_OBS:
        V = float(_V_mix_arr(np.array([P]), np.array([T]), 1.0)[0])
        d_fes.append(100*(M_FES/V*1000 - o)/o if np.isfinite(V) else np.nan)
    rms_fes = float(np.sqrt(np.nanmean(np.array(d_fes)**2)))

    # (3) 剖面收斂與速度
    t0 = time.time()
    prof = build_core_profile(2200.0, 0.17, R_cmb_km=1650.0, P_cmb=22.0)
    dt = time.time()-t0

    ok = ((np.isfinite(err_max) and err_max < 5e-4)
          and (prof is not None) and (dt < 1.0))

    if verbose:
        print(f'selftest [{ROUTE}: {_cfg()["tag"]}]')
        print(f'  [判定] table-vs-exact max err = {err_max:.2e}   (需 <5e-4)')
        if prof:
            print(f'  [判定] profile: P_c={prof["P_center"]:.2f} GPa  '
                  f'T_c={prof["T_center"]:.0f} K  '
                  f'rho_mean={prof["rho_mean"]/1e3:.3f} g/cm3  '
                  f'Vp_cmb={prof["Vp"][0]:.2f} km/s  '
                  f'R_ic={find_R_IC(prof):.0f} km  '
                  f'gap_min={gap_min(prof):+.0f} K  '
                  f'iters={prof["n_iter"]}  ({dt*1e3:.0f} ms)')
        else:
            print('  [判定] profile: None(超出表格範圍)')
        print(f'  [記錄] Fe  (chi=0) vs Kuwayama 6 pts: rms {rms_fe:5.2f}%   '
              f'{" ".join(f"{x:+5.2f}" for x in d_fe)}')
        print(f'  [記錄] FeS (chi=1) vs Sakai    6 pts: rms {rms_fes:5.2f}%   '
              f'{" ".join(f"{x:+5.1f}" for x in d_fes)}')
        print('  ->', 'PASS' if ok else 'FAIL')
    return ok

def compare_routes(w_S=0.17, T_core=2200.0, R_cmb_km=1650.0, P_cmb=22.0,
                   routes=('A', 'B'), verbose=True):
    """在同一組參數下,把各路線的核剖面關鍵量並列。

    刻意不印差值、不換算 sigma:兩條路線對等,沒有基準線。
    要比較的是各自 MCMC 跑出來的後驗,不是這裡的單點差。
    這張表的用途是確認兩條路線都能收斂、數值都在合理範圍。
    """
    _r0, _s0 = ROUTE, LIU_S_READING
    out = {}
    if verbose:
        print(f'compare_routes: w_S={w_S:.3f}  T_core={T_core:.0f} K  '
              f'R_cmb={R_cmb_km:.0f} km  P_cmb={P_cmb:.2f} GPa')
        print(f'  {"route":<6}{"rho_mean":>10}{"P_center":>10}{"T_center":>10}'
              f'{"Vp_cmb":>9}{"R_ic":>8}{"gap_min":>9}   tag')
    for rt in routes:
        configure(route=rt, verbose=False)
        p = build_core_profile(T_core, w_S, R_cmb_km=R_cmb_km, P_cmb=P_cmb)
        if p is None:
            if verbose:
                print(f'  {rt:<6}  未收斂 / 超出表格範圍')
            continue
        out[rt] = p
        if verbose:
            print(f'  {rt:<6}{p["rho_mean"]/1e3:10.3f}{p["P_center"]:10.2f}'
                  f'{p["T_center"]:10.0f}{p["Vp"][0]:9.2f}'
                  f'{find_R_IC(p):8.0f}{gap_min(p):+9.0f}   {ROUTES[rt]["tag"]}')
    configure(route=_r0, s_reading=_s0, verbose=False)
    return out

if __name__ == '__main__':
    for rt in ('B', 'A'):
        configure(route=rt)
        selftest()
        print()
    compare_routes()