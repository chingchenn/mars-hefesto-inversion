#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bml_segregation_check.py  —  跑在 cluster (beno3) 上

檢查「固液完全分離、無 mush」這個假設是否自洽。

對後驗每一步（chain.json 已含被拒步的原地重複 → 天然的後驗權重）：
  1. 用 nGibbs 求固態 BML 密度  rho_S(P_mid, T, Mg#=X_S)
  2. 用 Thomas EoS 求液態 BML 密度 rho_L(P_mid, T, Mg#=X_L)
  3. Delta_rho = rho_L - rho_S          <- 驅動滲流分離的密度差
  4. w0 = (k/mu)((1-phi)/phi) Delta_rho g      McKenzie 1984 (B7)
     其中 phi = f_liquid = h_liquid/(h_solid+h_liquid)   (反事實：若整層是 mush)
  5. delta_c = sqrt(k(zeta+4eta/3)/mu)          (3.3)
  6. 排空時間 t_drain = D_bml / w0

判準：t_drain << 4.5 Gyr  →  完全分離假設可辯護。

用法:
  python bml_segregation_check.py --mcmc_dir /net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc --prefix chain_a31 --burnin 800 --drop 06,09,11,14 --out bml_seg_a31
"""
import os, sys, json, glob, argparse, importlib.util
import numpy as np

SEC_YR = 3.156e7
AGE_MARS = 4.5e9

MAIN = '/home/jcchen2/mars-hefesto-inversion/59.mcmc_nGibbs_Mars_a25.py'

# ── percolation parameters (掃描用) ───────────────────────────────────────────
A_GRAIN = 1e-3        # m,   grain radius
C_KC    = 1000.0      # Blake-Kozeny-Carman constant (McKenzie 1984 §4)
MU_MELT = 1.0         # Pa s, melt viscosity  (Fe-rich depolymerised: 0.05-1)
ETA_MTX = 1e19        # Pa s, matrix shear viscosity
G_MARS  = 3.72

SENS = {              # 敏感度掃描
    'a_m':   [3e-4, 1e-3, 3e-3],
    'mu':    [0.1, 1.0],
    'C':     [100.0, 1000.0, 3000.0],
    'n':     [2.0, 3.0],
}


# ── robust chain reader (salvage truncated json) ─────────────────────────────
def read_chain(path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        txt = open(path).read()
        cut = txt.rfind('\n  },')
        if cut < 0:
            return []
        try:
            return json.loads(txt[:cut] + '\n  }\n]')
        except json.JSONDecodeError:
            return []


# ── permeability / velocities ────────────────────────────────────────────────
def perm(phi, a=A_GRAIN, C=C_KC, n=3.0):
    return a**2 * phi**n / (C * (1.0 - phi)**2)


def w0_mcken(phi, drho, a=A_GRAIN, C=C_KC, mu=MU_MELT, n=3.0):
    """McKenzie (B7), m/s.  drho in kg/m3."""
    k = perm(phi, a, C, n)
    return k / mu * (1.0 - phi) / phi * drho * G_MARS


def delta_c(phi, a=A_GRAIN, C=C_KC, mu=MU_MELT, eta=ETA_MTX, n=3.0,
            zeta_divergent=True):
    k = perm(phi, a, C, n)
    eta_b = eta * (1.0/phi + 4.0/3.0) if zeta_divergent else eta * (1.0 + 4.0/3.0)
    return np.sqrt(k * eta_b / mu)


# ── load the main module (filename starts with a digit → importlib) ──────────
def load_main(path=MAIN):
    spec = importlib.util.spec_from_file_location('mars_main', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rho_solid_ngibbs(M, P_GPa, T_K, Mg):
    """nGibbs solid density at (P, T, Mg#), g/cm3."""
    p = M.composition_from_params({'Mg#': float(Mg), 'T_lit': float(T_K),
                                   'P_lit': float(P_GPa)})
    O = M.compute_oxygen(p)
    out = M._ng_forward([P_GPa], [T_K], M._comp_values(p, O),
                        M.NG_T_HEADERS, ['component_moles'])
    pr = M._ng_props(out['component_moles'], [P_GPa], [T_K], ['rho'])
    return float(np.asarray(pr['rho'])[0])


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcmc_dir', required=True)
    ap.add_argument('--prefix', default='chain')
    ap.add_argument('--burnin', type=int, default=800)
    ap.add_argument('--drop', default='', help='comma-separated chain ids to skip')
    ap.add_argument('--main', default=MAIN)
    ap.add_argument('--no-ngibbs', action='store_true',
                    help='skip nGibbs; use rho_S from a fixed value (debug)')
    ap.add_argument('--rho_s_fallback', type=float, default=4.15)
    ap.add_argument('--out', default='bml_segregation')
    args = ap.parse_args()

    drop = {int(x) for x in args.drop.split(',') if x.strip()}

    M = None
    if not args.no_ngibbs:
        print('loading main module (builds gravity/pressure + Samuel cache) ...')
        M = load_main(args.main)
        pressure_mars = M.pressure_mars
        TRUE_CMB = M.TRUE_CMB_DEPTH
    else:
        # crude Mars P(z): only for --no-ngibbs debugging
        TRUE_CMB = 1743.32
        pressure_mars = lambda z: np.interp(z, [0, 500, 1000, 1500, 1743.32],
                                            [0, 5.6, 11.3, 17.2, 19.8])

    rows = []
    for f in sorted(glob.glob(os.path.join(args.mcmc_dir, f'{args.prefix}_*', 'chain.json'))):
        cid = int(os.path.basename(os.path.dirname(f)).split('_')[-1])
        if cid in drop:
            print(f'  chain_{cid:02d}: skipped'); continue
        ch = read_chain(f)
        if len(ch) <= args.burnin:
            print(f'  chain_{cid:02d}: {len(ch)} steps <= burnin, skipped'); continue
        print(f'  chain_{cid:02d}: {len(ch)} steps, using {len(ch)-args.burnin}')
        for s in ch[args.burnin:]:
            rows.append((cid, s))

    print(f'\ntotal posterior samples: {len(rows)}')

    rec = []
    n_layered = n_solid = n_molten = 0
    cache = {}
    for cid, s in rows:
        st = s.get('thermal_state')
        hs = s.get('h_solid_km') or 0.0
        hl = s.get('h_liquid_km') or 0.0
        D = s['params']['BML_thickness']

        if hl < 1.0:
            n_solid += 1; continue
        if hs < 1.0:
            n_molten += 1; continue
        n_layered += 1

        XS = s.get('Mg_solid')
        XL = s.get('Mg_liquid')
        Tint = s.get('T_interface')
        if XS is None or XL is None or XL < 0 or Tint is None:
            continue

        top = TRUE_CMB - D
        P_int = float(pressure_mars(top + hs))          # 固/液介面壓力
        T_int = float(Tint)

        if args.no_ngibbs:
            rS = args.rho_s_fallback
        else:
            key = (round(P_int, 2), round(T_int, 0), round(XS, 3))
            if key not in cache:
                try:
                    cache[key] = rho_solid_ngibbs(M, P_int, T_int, XS)
                except Exception:
                    cache[key] = np.nan
            rS = cache[key]
        if not np.isfinite(rS):
            continue

        rL = (M.liquid_bml_properties(P_int, T_int, XL)[0] if M is not None
              else np.nan)
        if not np.isfinite(rL):
            continue

        drho = (rL - rS) * 1000.0                       # kg/m3
        phi = hl / (hs + hl)                            # 反事實 mush 的熔體分率
        rec.append(dict(cid=cid, step=s['step'], D=D, hs=hs, hl=hl, phi=phi,
                        P=P_int, T=T_int, XS=XS, XL=XL,
                        rho_S=rS, rho_L=rL, drho=drho))

    print(f'  solid {n_solid}   layered {n_layered}   molten {n_molten}')
    print(f'  usable layered samples: {len(rec)}')
    if not rec:
        print('no layered samples — nothing to check'); return

    phi = np.array([r['phi'] for r in rec])
    drho = np.array([r['drho'] for r in rec])
    D_m = np.array([r['D'] for r in rec]) * 1e3

    def q(x, name, unit=''):
        p5, p50, p95 = np.percentile(x, [5, 50, 95])
        print(f'  {name:24s} {p50:12.4g}  [{p5:.4g}, {p95:.4g}] {unit}')

    print('\n' + '=' * 74)
    print('POSTERIOR (layered state only)')
    print('=' * 74)
    q(phi, 'phi = f_liquid')
    q(drho, 'Delta_rho', 'kg/m3')
    q(np.array([r['rho_S'] for r in rec]), 'rho_S (nGibbs)', 'g/cm3')
    q(np.array([r['rho_L'] for r in rec]), 'rho_L (Thomas)', 'g/cm3')
    q(np.array([r['P'] for r in rec]), 'P at interface', 'GPa')
    print(f"  Delta_rho < 0 的比例: {100*np.mean(drho <= 0):.2f} %   "
          f"(<0 表示熔體較輕 → 會往上跑，與模型幾何矛盾)")

    pos = drho > 0
    print('\n' + '=' * 74)
    print('SEGREGATION CHECK   (只用 Delta_rho > 0 的樣本)')
    print('=' * 74)
    print(f"{'a(mm)':>6} {'mu':>5} {'C':>6} {'n':>3} "
          f"{'delta_c med(m)':>15} {'t_drain med(yr)':>16} {'P(t>4.5Gyr)':>13}")
    for a_ in SENS['a_m']:
        for mu_ in SENS['mu']:
            for C_ in SENS['C']:
                for n_ in SENS['n']:
                    w = w0_mcken(phi[pos], drho[pos], a=a_, C=C_, mu=mu_, n=n_)
                    t = D_m[pos] / w / SEC_YR
                    dc = delta_c(phi[pos], a=a_, C=C_, mu=mu_, n=n_)
                    print(f"{a_*1e3:6.1f} {mu_:5.2f} {C_:6.0f} {n_:3.0f} "
                          f"{np.median(dc):15.4g} {np.median(t):16.4g} "
                          f"{100*np.mean(t > AGE_MARS):12.2f}%")

    # 參考參數下的完整統計
    w = w0_mcken(phi[pos], drho[pos])
    t = D_m[pos] / w / SEC_YR
    dc = delta_c(phi[pos])
    print('\nreference (a=1mm, mu=1, C=1000, n=3):')
    q(dc, 'delta_c', 'm')
    q(D_m[pos] / dc, 'D / delta_c')
    q(w * SEC_YR, 'w0', 'm/yr')
    q(t, 't_drain', 'yr')
    print(f'  P(t_drain > 4.5 Gyr) = {100*np.mean(t > AGE_MARS):.2f} %')
    print(f'  P(t_drain > 100 Myr) = {100*np.mean(t > 1e8):.2f} %')

    np.savez(f'{args.out}.npz',
             phi=phi, drho=drho, D_m=D_m,
             rho_S=np.array([r['rho_S'] for r in rec]),
             rho_L=np.array([r['rho_L'] for r in rec]),
             P=np.array([r['P'] for r in rec]),
             T=np.array([r['T'] for r in rec]),
             XS=np.array([r['XS'] for r in rec]),
             XL=np.array([r['XL'] for r in rec]))
    print(f'\nsaved -> {args.out}.npz')

    # ── figure ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))

        ax[0].hist(drho, bins=50, color='#4C78A8')
        ax[0].axvline(0, color='k', lw=2)
        ax[0].set_xlabel(r'$\Delta\rho=\rho_L-\rho_S$ (kg m$^{-3}$)', fontsize=12)
        ax[0].set_ylabel('posterior count', fontsize=12)
        ax[0].tick_params(labelsize=14)

        ax[1].hist(phi, bins=50, color='#F58518')
        ax[1].set_xlabel(r'$\phi=f_{\rm liquid}$', fontsize=12)
        ax[1].tick_params(labelsize=14)

        ax[2].hist(np.log10(dc), bins=50, color='#54A24B')
        ax[2].set_xlabel(r'$\log_{10}\,\delta_c$ (m)', fontsize=12)
        ax[2].tick_params(labelsize=14)

        ax[3].hist(np.log10(t), bins=50, color='#E45756')
        ax[3].axvline(np.log10(AGE_MARS), color='k', ls='--', lw=2,
                      label='4.5 Gyr')
        ax[3].set_xlabel(r'$\log_{10}\,(D/w_0)$ (yr)', fontsize=12)
        ax[3].legend(fontsize=10)
        ax[3].tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(f'{args.out}.png', dpi=150)
        print(f'saved -> {args.out}.png')
    except Exception as e:
        print('plotting skipped:', e)


if __name__ == '__main__':
    main()