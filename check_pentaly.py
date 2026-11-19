import numpy as np, json, os, io, importlib.util, traceback, contextlib
spec = importlib.util.spec_from_file_location("m", "59.mcmc_nGibbs_Mars_a25.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

N      = 120           # 取樣數。120 對比例的誤差約 +-4.5%,足夠做決策
OUT    = 'uc_scan.json'
RESUME = True          # True: 接續 OUT 已有的樣本,可以隨時 Ctrl-C 再續跑

rows = []
if RESUME and os.path.exists(OUT):
    rows = json.load(open(OUT))
    print(f"resume: 已有 {len(rows)} 個樣本")

core_mask    = m._samuel_cache['core_z'] >= m.TRUE_CMB_DEPTH
rho_core_top = float(m._samuel_cache['core_rho'][core_mask][0])

rng = np.random.default_rng(0)
ffor i in range(len(rows), N):
    rng = np.random.default_rng(1000 + i)
    p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in m.PRIOR.items()}

for i in range(len(rows), N):
    p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in m.PRIOR.items()}
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # 吃掉逐樣本的雜訊
            bml_top = m.TRUE_CMB_DEPTH - p['BML_thickness']
            P_top   = float(m.pressure_mars(bml_top))
            f56     = m.run_ngibbs(p, P_bml_top=P_top)
            bml     = None
            if f56 is not None:
                rho_mantle_bot = float(f56['rho'][-1])
                T_mb = float(np.interp(P_top, f56['P_profile'], f56['T_profile']))
                bml  = m.run_ngibbs_bml(p, T_core=p['T_core'], T_mantle_bottom=T_mb,
                                        true_cmb_depth=m.TRUE_CMB_DEPTH,
                                        rho_solid_bml_si=m._RHO_S_CONST * 1000.0)
        if bml is None:
            print(f"  {i:4d}  skipped")
            continue
        rows.append(dict(
            uc    = float(bml['rho'][0]) - rho_mantle_bot,
            lc    = rho_core_top - float(bml['rho'][-1]),
            h_sol = float(bml['h_solid_km']),
            h_liq = float(bml['h_liquid_km']),
            T_core         = p['T_core'],
            Mg_bulk_bml    = p['Mg#_bulk_bml'],
            BML_thickness  = p['BML_thickness'],
            rho_mantle_bot = rho_mantle_bot,
            rho_solid_bml  = float(bml['rho_solid_bml']),
            rho_liquid_bml = float(bml['rho_liquid_bml']),
            Ra             = float(bml['Ra']) if np.isfinite(bml['Ra']) else -1.0,
        ))
        r = rows[-1]
        print(f"  {i:4d}  h_sol={r['h_sol']:5.0f}  h_liq={r['h_liq']:5.0f}  "
              f"uc={r['uc']:+.4f}  rho_L={r['rho_liquid_bml']:.3f}")
    except Exception:
        print(f"--- sample {i} failed ---"); traceback.print_exc()
        continue
    if i % 10 == 0:
        json.dump(rows, open(OUT, 'w'), indent=1)

json.dump(rows, open(OUT, 'w'), indent=1)

# ── 分析 ──────────────────────────────────────────────────────────────────────
uc      = np.array([r['uc'] for r in rows])
lc      = np.array([r['lc'] for r in rows])
h_sol   = np.array([r['h_sol'] for r in rows])
h_liq   = np.array([r['h_liq'] for r in rows])
solid   = h_liq < 1.0
molten  = h_sol < 1.0
layered = ~solid & ~molten

print(f"\nn={len(uc)}   (比例的 1-sigma 約 +-{100*np.sqrt(0.25/len(uc)):.1f}%)")
print(f"  三態: solid {solid.sum()} ({100*solid.mean():.1f}%)  "
      f"layered {layered.sum()} ({100*layered.mean():.1f}%)  "
      f"molten {molten.sum()} ({100*molten.mean():.1f}%)")
print(f"        a31 舊錨點: 28.4 / 13.2 / 58.4 %")

print(f"\n  uc median={np.median(uc):+.4f}  min={uc.min():+.4f}  max={uc.max():+.4f}")
print(f"  uc < 0 全體  : {100*np.mean(uc < 0):.1f}%")
if molten.sum():
    print(f"  uc < 0 全熔組: {100*np.mean(uc[molten] < 0):.1f}%  (n={molten.sum()})")
if layered.sum():
    print(f"  uc < 0 分層組: {100*np.mean(uc[layered] < 0):.1f}%  (n={layered.sum()})")
print(f"  lc < 0 全體  : {100*np.mean(lc < 0):.1f}%")
if (uc < 0).any():
    print(f"  uc<0 的 T_core 中位數={np.median([r['T_core'] for r in rows if r['uc']<0]):.0f} K  "
          f"Mg#_bml 中位數={np.median([r['Mg_bulk_bml'] for r in rows if r['uc']<0]):.3f}")

rl = np.array([r['rho_liquid_bml'] for r in rows]); rl = rl[rl > 0]
rs = np.array([r['rho_solid_bml']  for r in rows]); rs = rs[rs > 0]
rm = np.array([r['rho_mantle_bot'] for r in rows])
print(f"\n  rho_liquid_bml : median {np.median(rl):.3f}  [{rl.min():.3f}, {rl.max():.3f}]  n={len(rl)}")
if len(rs):
    print(f"  rho_solid_bml  : median {np.median(rs):.3f}  [{rs.min():.3f}, {rs.max():.3f}]  n={len(rs)}"
          f"   (mole_to_volume_fraction 用的常數 = {m._RHO_S_CONST})")
print(f"  rho_mantle_bot : median {np.median(rm):.3f}  [{rm.min():.3f}, {rm.max():.3f}]")

ra = np.array([r['Ra'] for r in rows]); ra = ra[ra > 0]
if len(ra):
    print(f"\n  Ra: median {np.median(ra):.1e}  max {ra.max():.1e}   "
          f"Ra_c={m._RA_C:.0e}   Ra>1e3 的比例 {100*np.mean(ra > 1e3):.1f}%")