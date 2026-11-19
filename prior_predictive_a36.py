import importlib.util, sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mars_config import FIG_DIR
SAVE_FIG = os.environ.get('SAVE_FIG') == '1'   # off by default; run with SAVE_FIG=1 to save to FIG_DIR
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

# ── config ────────────────────────────────────────────────────────────────────
SRC_DIR       = HERE
MODULE_PATH   = f"{SRC_DIR}/70.mcmc_core_nGibbs.py"
ROUTE         = "A"                     # 'A' = all-Liu Gamma, 'B' = Liu+Sakai Margules

SC_N_ITER_TAB = 14                      # 建表時放寬鬆弛上限(runtime 是 8,那個 FAIL 卡在 dP=0.015)
R_CMB_NODES     = 13
MG_STRUCT_NODES = 3                     # 太慢就設 1
Z_GRID          = np.linspace(0.0, 2100.0, 421)      # km

N_SAMPLES     = 200_000                 # 簡化分類器(向量化,瞬間)
N_EXACT       = 5000                    # 精確端點分類器的子樣本數
SEED          = 0

TABLE_STRUCT  = f"{SRC_DIR}/struct_table_prior_a36{ROUTE}.npz"
OUT_SAMPLES   = f"{SRC_DIR}/prior_predictive_a36{ROUTE}.npz"
OUT_FIG       = f"{FIG_DIR}/prior_predictive_a36{ROUTE}.png"

# ── load a36 module ───────────────────────────────────────────────────────────
sys.path.insert(0, SRC_DIR)             # 70.py 內有 `import core_fes as CF`
spec = importlib.util.spec_from_file_location("m70", MODULE_PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

m.CF.configure(route=ROUTE, verbose=True)   # __main__ 沒跑,route 必須自己設
m.CORE_EOS_ROUTE = ROUTE
m.SC_N_ITER = SC_N_ITER_TAB

R_MARS = m.MARS_RADIUS
print("\nPRIOR check:")
for k in ('T_core', 'Mg#_bulk_bml', 'BML_thickness', 'R_cmb', 'Mg#'):
    print(f"  {k:16s} {m.PRIOR[k]}")

# ── 步驟 1:(R_cmb, Mg#) 上的自洽壓力剖面表 ──────────────────────────────────
# a36 的 P 不再是固定場。完整自洽需要 8 個參數,但幾何的主控是 R_cmb、
# 地函密度的主控是 Mg#,其餘固定在先驗中位數。
if os.path.exists(TABLE_STRUCT):
    z = np.load(TABLE_STRUCT)
    R_nodes, MgS_nodes, P_zgrid = z['R_nodes'], z['MgS_nodes'], z['P_zgrid']
    print(f"loaded struct table from {TABLE_STRUCT}")
else:
    R_nodes   = np.linspace(*m.PRIOR['R_cmb'], R_CMB_NODES)
    MgS_nodes = (np.array([0.5*sum(m.PRIOR['Mg#'])]) if MG_STRUCT_NODES == 1
                 else np.linspace(*m.PRIOR['Mg#'], MG_STRUCT_NODES))
    P_zgrid = np.full((len(R_nodes), len(MgS_nodes), len(Z_GRID)), np.nan)

    base = {k: 0.5*(lo + hi) for k, (lo, hi) in m.PRIOR.items()}
    t0 = time.time(); n_total = len(R_nodes)*len(MgS_nodes); n_done = 0
    for i, Rc in enumerate(R_nodes):
        for j, MgS in enumerate(MgS_nodes):
            p = base.copy(); p['R_cmb'] = float(Rc); p['Mg#'] = float(MgS)
            prof = m.sc_build_profile(p, m._samuel_cache)
            if prof is not None:
                P_zgrid[i, j] = np.interp(Z_GRID, prof['z_km'], prof['P_GPa'],
                                          left=np.nan, right=np.nan)
            n_done += 1; el = time.time() - t0
            print(f"  struct {n_done}/{n_total}  R_cmb={Rc:.0f}  Mg#={MgS:.3f}  "
                  f"{'ok' if prof is not None else 'FAIL'}  "
                  f"{el:.0f}s  ETA {el/n_done*(n_total-n_done):.0f}s")
    np.savez(TABLE_STRUCT, R_nodes=R_nodes, MgS_nodes=MgS_nodes, P_zgrid=P_zgrid)
    print(f"saved -> {TABLE_STRUCT}")

# 一個 NaN 角點會讓相鄰 cell 的三線性內插整個回 NaN,必須是 0 才能往下走
bad = np.isnan(P_zgrid[:, :, 0])
print(f"\nstruct fail nodes = {int(bad.sum())}")
for i, j in zip(*np.nonzero(bad)):
    print(f"  BAD  R_cmb={R_nodes[i]:.0f}  Mg#={MgS_nodes[j]:.3f}")
if bad.any():
    raise SystemExit("struct 表有 NaN 節點,提高 SC_N_ITER_TAB 後刪檔重跑")

# ── 步驟 2:先驗抽樣 ─────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
Tc   = rng.uniform(*m.PRIOR['T_core'],        N_SAMPLES)
Mgb  = rng.uniform(*m.PRIOR['Mg#_bulk_bml'],  N_SAMPLES)
Dk   = rng.uniform(*m.PRIOR['BML_thickness'], N_SAMPLES)
Rcmb = rng.uniform(*m.PRIOR['R_cmb'],         N_SAMPLES)
MgM  = rng.uniform(*m.PRIOR['Mg#'],           N_SAMPLES)

cmb_depth = R_MARS - Rcmb                      # 與 sc_build_grid 的 z_cmb 一致
z_bml_top = cmb_depth - Dk
geom_bad  = z_bml_top <= m.SC_Z_FIXED          # sc_build_grid 的幾何拒絕條件

if len(MgS_nodes) == 1:
    _f = RegularGridInterpolator((R_nodes, Z_GRID), P_zgrid[:, 0, :],
                                 bounds_error=False)
    f_P = lambda R, Mg, z: _f(np.column_stack([R, z]))
else:
    _f = RegularGridInterpolator((R_nodes, MgS_nodes, Z_GRID), P_zgrid,
                                 bounds_error=False)
    f_P = lambda R, Mg, z: _f(np.column_stack([R, Mg, z]))

P_cmb = f_P(Rcmb, MgM, cmb_depth)
P_top = f_P(Rcmb, MgM, z_bml_top)
print(f"\nP_cmb      5/50/95%  {np.nanpercentile(P_cmb[~geom_bad],5):.2f} / "
      f"{np.nanmedian(P_cmb[~geom_bad]):.2f} / "
      f"{np.nanpercentile(P_cmb[~geom_bad],95):.2f} GPa")
print(f"P_bml_top  5/50/95%  {np.nanpercentile(P_top[~geom_bad],5):.2f} / "
      f"{np.nanmedian(P_top[~geom_bad]):.2f} / "
      f"{np.nanpercentile(P_top[~geom_bad],95):.2f} GPa")

# ── 步驟 3:精確端點分類器 ──────────────────────────────────────────────────
# sc_bml 的三態只由 _resid 的兩個端點決定,brentq 只在兩者都不觸發時才跑
# (那就是 layered),所以分類不需要解 h_solid:
#
#   gD = D - st(D)[0] <= 0  ->  solid    界面在 CMB,T_int = T_core,
#                                        查相圖於 (T_core, P_cmb)
#   g0 =   - st(0)[0] >= 0  ->  molten   液體絕熱線由 (P_cmb, T_core) 積到
#                                        P_bml_top,查相圖於 (T_int, P_top)
#   else                    ->  layered
rho_S = m._RHO_S_CONST
ok    = np.flatnonzero(~geom_bad & np.isfinite(P_cmb) & np.isfinite(P_top))
idx   = rng.choice(ok, size=min(N_EXACT, ok.size), replace=False)

state   = np.full(len(idx), 'fail', dtype='<U8')
h_liq   = np.full(len(idx), np.nan)
T_int_a = np.full(len(idx), np.nan)

t0 = time.time()
for n, k in enumerate(idx):
    z_b = np.linspace(z_bml_top[k], cmb_depth[k], m.SC_N_BML)
    P_b = f_P(np.full(m.SC_N_BML, Rcmb[k]), np.full(m.SC_N_BML, MgM[k]), z_b)
    if not np.all(np.isfinite(P_b)):
        continue
    args = (float(Tc[k]), z_b, P_b, float(Mgb[k]), float(Dk[k]), rho_S)

    st_D = m._sc_interface_state(float(Dk[k]), *args, n_seg=5)
    if st_D is None:
        continue
    if Dk[k] - st_D[0] <= 0.0:
        state[n] = 'solid'; h_liq[n] = 0.0; T_int_a[n] = st_D[2]
    else:
        st_0 = m._sc_interface_state(0.0, *args, n_seg=5)
        if st_0 is None:
            continue
        if -st_0[0] >= 0.0:
            state[n] = 'molten'; h_liq[n] = Dk[k]; T_int_a[n] = st_0[2]
        else:
            # layered:解 h_solid = D * phi_S(h_solid),與 sc_bml 同一條方程
            f = lambda h: h - m._sc_interface_state(h, *args, n_seg=5)[0]
            h_s = float(m.brentq(f, 0.0, float(Dk[k]), xtol=0.5))
            st_h = m._sc_interface_state(h_s, *args, n_seg=5)
            state[n] = 'molten' if h_s < 1.0 else 'layered'
            h_liq[n] = Dk[k] - h_s
            T_int_a[n] = st_h[2]

    if (n + 1) % 200 == 0:
        el = time.time() - t0
        print(f"  exact {n+1}/{len(idx)}  {el:.0f}s  "
              f"ETA {el/(n+1)*(len(idx)-n-1):.0f}s")

good = state != 'fail'
tot  = int(good.sum())
print(f"\nthree-state prior fractions (a36{ROUTE}, R_cmb free, exact classifier)")
for s in ('solid', 'layered', 'molten'):
    n = int((state == s).sum())
    print(f"  {s:8s} {n:6d}  {100*n/tot:5.1f}%")
print(f"  fail     {int((~good).sum()):6d}   (n_used = {tot})")
print("  a34 對照(R_cmb 固定 1743.3,簡化分類器):solid 21.4 / layered 9.6 / molten 69.0 %")

print(f"\nh_liquid prior")
print(f"  5/50/95%  {np.percentile(h_liq[good],5):.1f} / "
      f"{np.median(h_liq[good]):.1f} / {np.percentile(h_liq[good],95):.1f} km")
print(f"  prior sd:  h_liquid {np.std(h_liq[good]):7.2f} km   "
      f"BML_thickness {np.std(Dk[idx][good]):7.2f} km")
print("  -> chain 的 h_liquid_km 後驗 sd 除以上面這個,就是 h_liquid 的 shrink")

# ── 步驟 4:先驗 odds(給 BF 用)────────────────────────────────────────────
n_s = int((state == 'solid').sum()); n_l = int((state == 'layered').sum())
n_m = int((state == 'molten').sum())
print(f"\nprior odds  layered:solid  = {n_l/n_s:.4f}" if n_s else "\nprior odds  layered:solid  = inf")
print(f"prior odds  layered:molten = {n_l/n_m:.4f}" if n_m else "prior odds  layered:molten = inf")
print(f"  Monte Carlo 標準誤 ~ sqrt(p(1-p)/n),n = {tot};BF 要照這個傳播")

np.savez(OUT_SAMPLES, idx=idx, T_core=Tc[idx], Mg_bulk_bml=Mgb[idx],
         BML_thickness=Dk[idx], R_cmb=Rcmb[idx], Mg_mantle=MgM[idx],
         P_cmb=P_cmb[idx], P_bml_top=P_top[idx], cmb_depth=cmb_depth[idx],
         z_bml_top=z_bml_top[idx], state=state, h_liquid=h_liq, T_int=T_int_a)
print(f"saved -> {OUT_SAMPLES}")

# ── 步驟 5:對照圖 ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

ax[0].hist(P_cmb[~geom_bad], bins=60, color='steelblue', label='P_cmb')
ax[0].hist(P_top[~geom_bad], bins=60, color='darkorange', alpha=0.6, label='P_bml_top')
ax[0].axvline(21.70, color='k', ls='--', lw=1.5, label='Samuel fixed P_cmb')
ax[0].set_xlabel('P [GPa]'); ax[0].set_ylabel('count')
ax[0].legend(fontsize=11); ax[0].tick_params(labelsize=14)

ax[1].hist(Dk[idx][good], bins=50, color='0.75', label='BML_thickness prior')
ax[1].hist(h_liq[good],   bins=50, color='steelblue', alpha=0.75, label='h_liquid prior')
ax[1].set_xlabel('thickness [km]'); ax[1].set_ylabel('count')
ax[1].legend(fontsize=11); ax[1].tick_params(labelsize=14)

frac = [n_s/tot, n_l/tot, n_m/tot]
ax[2].bar(['solid', 'layered', 'molten'], frac, color=['0.4', 'steelblue', 'firebrick'])
ax[2].set_ylabel('prior fraction'); ax[2].tick_params(labelsize=14)
for i, f in enumerate(frac):
    ax[2].text(i, f + 0.01, f'{100*f:.1f}%', ha='center', fontsize=13)

if SAVE_FIG: fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150)
print(f"saved -> {OUT_FIG}")