import importlib.util, numpy as np, time
import os
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

spec = importlib.util.spec_from_file_location("m59",
    f"{HERE}/59.mcmc_nGibbs_Mars_a25.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("PRIOR check:", m.PRIOR['T_core'], m.PRIOR['Mg#_bulk_bml'], m.PRIOR['BML_thickness'])
TABLE = f"{HERE}/bml_phase_table_prior_a34_61x201.npz"
# ── 步驟 1:在 (P, Mg) 網格上算 T_sol / T_liq 表(唯一慢的部分)──
D_lo, D_hi = m.PRIOR['BML_thickness']
P_lo = float(m.pressure_mars(m.TRUE_CMB_DEPTH - 0.5*D_hi))
P_hi = float(m.pressure_mars(m.TRUE_CMB_DEPTH - 0.5*D_lo))

if os.path.exists(TABLE):
    z = np.load(TABLE)
    P_grid, Mg_grid, Tsol, Tliq = z['P_grid'], z['Mg_grid'], z['Tsol'], z['Tliq']
    print(f"loaded table from {TABLE}")
else:
    P_grid  = np.linspace(P_lo - 0.05, P_hi + 0.05, 61)
    Mg_grid = np.linspace(*m.PRIOR['Mg#_bulk_bml'], 201)
    Tsol = np.full((len(P_grid), len(Mg_grid)), np.nan)
    Tliq = np.full((len(P_grid), len(Mg_grid)), np.nan)
    t0 = time.time()
    n_total = len(P_grid) * len(Mg_grid)
    n_done = 0
    for i, P in enumerate(P_grid):
        for j, Mg in enumerate(Mg_grid):
            pd = m.bml_phase_diagram(2500.0, P, Mg)
            if pd is not None:
                Tsol[i, j] = pd['T_solidus']; Tliq[i, j] = pd['T_liquidus']
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t0
                rate = elapsed / n_done
                eta = rate * (n_total - n_done)
                print(f"  {n_done}/{n_total}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")
    print(f"table: {n_total} nodes, {time.time()-t0:.1f}s, "
        f"fail nodes = {int(np.isnan(Tsol).sum())}")
    np.savez(TABLE, P_grid=P_grid, Mg_grid=Mg_grid, Tsol=Tsol, Tliq=Tliq)
    print(f"saved -> {TABLE}")
# ── 步驟 2:大樣本向量化分類(瞬間)──
N = 200_000; rng = np.random.default_rng(0)
Tc = rng.uniform(*m.PRIOR['T_core'], N)
Mg = rng.uniform(*m.PRIOR['Mg#_bulk_bml'], N)
Dk = rng.uniform(*m.PRIOR['BML_thickness'], N)
P  = np.interp(m.TRUE_CMB_DEPTH - 0.5*Dk, m._pres_depth, m._pres_gpa)  # 向量化 pressure_mars
f_sol = RegularGridInterpolator((P_grid, Mg_grid), Tsol, bounds_error=False)
f_liq = RegularGridInterpolator((P_grid, Mg_grid), Tliq, bounds_error=False)

ts = f_sol(np.column_stack([P, Mg])); tl = f_liq(np.column_stack([P, Mg]))
fail    = np.isnan(ts) | np.isnan(tl)
solid   = (~fail) & (Tc <= ts)
molten  = (~fail) & (Tc >= tl)
layered = (~fail) & ~solid & ~molten

tot = N - fail.sum()
for name, mask in (('solid', solid), ('layered', layered), ('molten', molten)):
    print(f"{name:8s} {mask.sum():7d}  {100*mask.sum()/tot:5.1f}%")
print("fail", int(fail.sum()))