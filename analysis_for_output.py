import importlib.util, time
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this script

MODULE = f"{HERE}/59.mcmc_nGibbs_Mars_a25.py"
OUT    = f"{HERE}/tables_a31.npz"

N_P    = 15
N_MGB  = 41
N_MG   = 41
N_T    = 29
T_LO   = 1800.0
T_HI   = 3200.0

spec = importlib.util.spec_from_file_location("m59", MODULE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("PRIOR:", m.PRIOR['T_core'], m.PRIOR['Mg#_bulk_bml'], m.PRIOR['BML_thickness'])

D_lo, D_hi = m.PRIOR['BML_thickness']
P_lo = float(m.pressure_mars(m.TRUE_CMB_DEPTH - 0.5*D_hi))
P_hi = float(m.pressure_mars(m.TRUE_CMB_DEPTH - 0.5*D_lo))
print(f"BML mid-pressure range: {P_lo:.3f} - {P_hi:.3f} GPa")

P_grid   = np.linspace(P_lo - 0.05, P_hi + 0.05, N_P)
MgB_grid = np.linspace(*m.PRIOR['Mg#_bulk_bml'], N_MGB)
Mg_grid  = np.linspace(0.15, 0.95, N_MG)
T_grid   = np.linspace(T_LO, T_HI, N_T)


# ── 1. solidus / liquidus on (P, Mg_bulk) ───────────────────────────────────
Tsol = np.full((N_P, N_MGB), np.nan)
Tliq = np.full((N_P, N_MGB), np.nan)
t0 = time.time()
for i, P in enumerate(P_grid):
    for j, Mgb in enumerate(MgB_grid):
        pd = m.bml_phase_diagram(2500.0, float(P), float(Mgb))
        if pd is not None:
            Tsol[i, j] = pd['T_solidus']
            Tliq[i, j] = pd['T_liquidus']
print(f"[1] Tsol/Tliq  {N_P*N_MGB} nodes  {time.time()-t0:.1f}s  "
      f"nan = {int(np.isnan(Tsol).sum())}")


# ── 2. phase split on (T, P, Mg_bulk) ───────────────────────────────────────
PD_KEY_S = None
PD_KEY_L = None
probe = m.bml_phase_diagram(2500.0, float(P_grid[N_P//2]), float(MgB_grid[N_MGB//2]))
print("[2] bml_phase_diagram keys:", sorted(probe.keys()) if probe else None)
for cand in ('XS', 'Mg_solid', 'Mg_S', 'X_solid'):
    if probe and cand in probe:
        PD_KEY_S = cand; break
for cand in ('XL', 'Mg_liquid', 'Mg_L', 'X_liquid'):
    if probe and cand in probe:
        PD_KEY_L = cand; break
print(f"    using solid key = {PD_KEY_S}   liquid key = {PD_KEY_L}")

f_solid = np.full((N_T, N_P, N_MGB), np.nan)
XS_tab  = np.full((N_T, N_P, N_MGB), np.nan)
XL_tab  = np.full((N_T, N_P, N_MGB), np.nan)
t0 = time.time()
for a, T in enumerate(T_grid):
    for i, P in enumerate(P_grid):
        for j, Mgb in enumerate(MgB_grid):
            pd = m.bml_phase_diagram(float(T), float(P), float(Mgb))
            if pd is None:
                continue
            f_solid[a, i, j] = pd['f_solid']
            if PD_KEY_S: XS_tab[a, i, j] = pd[PD_KEY_S]
            if PD_KEY_L: XL_tab[a, i, j] = pd[PD_KEY_L]
    print(f"    T = {T:7.1f} K   done   ({time.time()-t0:.0f}s elapsed)")
print(f"[2] phase split  {N_T*N_P*N_MGB} nodes  {time.time()-t0:.1f}s  "
      f"nan = {int(np.isnan(f_solid).sum())}")


# ── 3. liquid EoS on (Mg, P, T) ─────────────────────────────────────────────
rho_liq = np.full((N_MG, N_P, N_T), np.nan)
vp_liq  = np.full((N_MG, N_P, N_T), np.nan)
t0 = time.time()
for i, Mg in enumerate(Mg_grid):
    for j, P in enumerate(P_grid):
        for k, T in enumerate(T_grid):
            r, v, _ = m.liquid_bml_properties(float(P), float(T), float(Mg))
            rho_liq[i, j, k] = r
            vp_liq[i, j, k]  = v
print(f"[3] liquid EoS  {N_MG*N_P*N_T} nodes  {time.time()-t0:.1f}s")


# ── 4. solid density / Vp / Vs on (Mg, P, T) via nGibbs ─────────────────────
rho_sol = np.full((N_MG, N_P, N_T), np.nan)
vp_sol  = np.full((N_MG, N_P, N_T), np.nan)
vs_sol  = np.full((N_MG, N_P, N_T), np.nan)
t0 = time.time()
for i, Mg in enumerate(Mg_grid):
    for k, T in enumerate(T_grid):
        p  = m.composition_from_params({'Mg#': float(Mg), 'T_lit': float(T),
                                        'P_lit': float(P_grid[0])})
        O  = m.compute_oxygen(p)
        cv = m._comp_values(p, O)
        T_arr = np.full(N_P, float(T))
        try:
            out = m._ng_forward(P_grid, T_arr, cv, m.NG_T_HEADERS, ['component_moles'])
            pr  = m._ng_props(out['component_moles'], P_grid, T_arr, ['rho', 'Vp', 'Vs'])
            rho_sol[i, :, k] = np.asarray(pr['rho'],  dtype=float)
            vp_sol[i, :, k]  = np.asarray(pr['Vp'],   dtype=float)
            vs_sol[i, :, k]  = np.asarray(pr['Vs'],   dtype=float)
        except Exception as e:
            print(f"    nGibbs fail Mg={Mg:.3f} T={T:.0f}: {e}")
    print(f"    Mg# = {Mg:.3f}   done   ({time.time()-t0:.0f}s elapsed)")
print(f"[4] solid nGibbs  {N_MG*N_P*N_T} nodes  {time.time()-t0:.1f}s  "
      f"nan = {int(np.isnan(rho_sol).sum())}")


# ── 5. everything else the plotting script needs ────────────────────────────
np.savez_compressed(
    OUT,
    P_grid=P_grid, MgB_grid=MgB_grid, Mg_grid=Mg_grid, T_grid=T_grid,
    Tsol=Tsol, Tliq=Tliq,
    f_solid=f_solid, XS_tab=XS_tab, XL_tab=XL_tab,
    rho_liq=rho_liq, vp_liq=vp_liq,
    rho_sol=rho_sol, vp_sol=vp_sol, vs_sol=vs_sol,
    pres_depth=np.asarray(m._pres_depth), pres_gpa=np.asarray(m._pres_gpa),
    TRUE_CMB_DEPTH=float(m.TRUE_CMB_DEPTH),
    RA_C=float(getattr(m, '_RA_C', 1.0e5)),
    MG_DUNCAN_REF=float(getattr(m, 'MG_DUNCAN_REF', 0.75)),
    prior_T_core=np.asarray(m.PRIOR['T_core']),
    prior_Mg_bulk=np.asarray(m.PRIOR['Mg#_bulk_bml']),
    prior_D=np.asarray(m.PRIOR['BML_thickness']),
    key_solid=str(PD_KEY_S), key_liquid=str(PD_KEY_L),
)
print(f"\nsaved -> {OUT}")
print("rsync this file to your laptop, then run plot_a31.py")