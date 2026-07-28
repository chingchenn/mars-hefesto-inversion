#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bml_segregation_stokes.py  —  取代 bml_segregation_check.py

修正版：a31 後驗的分層態 phi = 0.70 [0.48, 0.83]，遠超過瓦解門檻
(McKenzie 1984 §4: phi ~ 0.2；一般文獻 RCMF 0.25-0.4)。
=> 固體不連通，沒有 matrix，McKenzie 的兩相壓實理論不適用。
=> 正確機制是「晶體懸浮於熔體中的阻滯 Stokes 沉降」。

本腳本同時算兩者並診斷該用哪一個：

  Darcy  (McKenzie B7):  w0     = a^2 drho g /(C mu) * phi^2/(1-phi)
  Stokes (+Richardson-Zaki): v  = (2/9) a^2 drho g / mu * phi^n_RZ
         (阻滯因子用 (1 - f_solid)^n_RZ = phi^n_RZ, n_RZ ~ 4.65)

另外檢查 Stokes 的適用性：Reynolds number Re = rho_L v a / mu << 1。

用法:
  python bml_segregation_stokes.py --mcmc_dir DIR --prefix chain_a31 \
      --burnin 800 --drop 06,09,11,14 --out bml_stokes_a31
"""
import argparse, importlib.util
import numpy as np

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共用：讀 chain.json（含 salvage）、三態分類、ESS 估計。"""
import os, json, glob
import numpy as np


def read_chain(path):
    """讀 chain.json；若檔案被寫到一半截斷，救回最後一個完整步。"""
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


def load_chains(mcmc_dir, prefix='chain', burnin=800, drop=()):
    """回傳 {chain_id: [steps]}（已去掉 burn-in）。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(mcmc_dir, f'{prefix}_*', 'chain.json'))):
        cid = int(os.path.basename(os.path.dirname(f)).split('_')[-1])
        if cid in drop:
            print(f'  chain_{cid:02d}: skipped'); continue
        ch = read_chain(f)
        if len(ch) <= burnin:
            print(f'  chain_{cid:02d}: {len(ch)} steps <= burnin, skipped'); continue
        out[cid] = ch[burnin:]
        print(f'  chain_{cid:02d}: {len(ch)} steps, using {len(out[cid])}')
    return out


def classify(step, h_min=1.0):
    """三態分類。回傳 'solid' / 'layered' / 'molten' / None。"""
    hs = step.get('h_solid_km')
    hl = step.get('h_liquid_km')
    if hs is None or hl is None:
        return None
    hs, hl = float(hs), float(hl)
    if hl < h_min and hs >= h_min:
        return 'solid'
    if hs < h_min and hl >= h_min:
        return 'molten'
    if hs >= h_min and hl >= h_min:
        return 'layered'
    return None


def act_ess(x):
    """
    自相關時間 tau_int 與 ESS，用 initial positive sequence estimator
    (Geyer 1992)。x 為單條鏈的一維序列。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20 or np.allclose(x, x[0]):
        return np.nan, float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    # 成對加總，取到第一個負值為止
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s
    return float(tau), float(n / tau)


def pooled_ess(series_by_chain):
    """把每條鏈的 ESS 加起來。"""
    taus, esss = [], []
    for s in series_by_chain:
        t, e = act_ess(s)
        if np.isfinite(t):
            taus.append(t)
        esss.append(e)
    return float(np.sum(esss)), (float(np.median(taus)) if taus else np.nan)

SEC_YR = 3.156e7
AGE_MARS = 4.5e9
G_MARS = 3.72
MAIN = '/home/jcchen2/mars-hefesto-inversion/59.mcmc_nGibbs_Mars_a25.py'

RCMF_LO, RCMF_HI = 0.25, 0.40      # 瓦解 / 懸浮的過渡區
N_RZ = 4.65                        # Richardson-Zaki 指數


# ── velocities ───────────────────────────────────────────────────────────────
def v_darcy(phi, drho, a=1e-3, C=1000.0, mu=1.0, n=3.0):
    """McKenzie (B7) minimum fluidization velocity, m/s."""
    k = a**2 * phi**n / (C * (1 - phi)**2)
    return k / mu * (1 - phi) / phi * drho * G_MARS


def delta_c(phi, a=1e-3, C=1000.0, mu=1.0, eta=1e19, n=3.0):
    k = a**2 * phi**n / (C * (1 - phi)**2)
    return np.sqrt(k * eta * (1.0/phi + 4.0/3.0) / mu)


def v_stokes(drho, a=1e-3, mu=1.0):
    """單顆晶體 Stokes 末速, m/s.  a = 晶體半徑."""
    return 2.0 * drho * G_MARS * a**2 / (9.0 * mu)


def v_stokes_hindered(phi, drho, a=1e-3, mu=1.0, n_rz=N_RZ):
    """阻滯沉降：乘上 (1 - f_solid)^n_rz = phi^n_rz."""
    return v_stokes(drho, a, mu) * phi**n_rz


def reynolds(v, a, rho_L_gcc, mu):
    return rho_L_gcc * 1e3 * v * a / mu


# ── nGibbs ───────────────────────────────────────────────────────────────────
def load_main(path=MAIN):
    spec = importlib.util.spec_from_file_location('mars_main', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rho_solid_ngibbs(M, P, T, Mg):
    p = M.composition_from_params({'Mg#': float(Mg), 'T_lit': float(T),
                                   'P_lit': float(P)})
    O = M.compute_oxygen(p)
    out = M._ng_forward([P], [T], M._comp_values(p, O),
                        M.NG_T_HEADERS, ['component_moles'])
    pr = M._ng_props(out['component_moles'], [P], [T], ['rho'])
    return float(np.asarray(pr['rho'])[0])


def q(x, name, unit=''):
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    print(f'  {name:26s} {p50:12.4g}  [{p5:.4g}, {p95:.4g}] {unit}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcmc_dir', required=True)
    ap.add_argument('--prefix', default='chain')
    ap.add_argument('--burnin', type=int, default=800)
    ap.add_argument('--drop', default='')
    ap.add_argument('--main', default=MAIN)
    ap.add_argument('--out', default='bml_stokes')
    args = ap.parse_args()

    drop = {int(x) for x in args.drop.split(',') if x.strip()}

    print('loading main module ...')
    M = load_main(args.main)
    TRUE_CMB = M.TRUE_CMB_DEPTH

    chains = load_chains(args.mcmc_dir, args.prefix, args.burnin, drop)

    rec, cache = [], {}
    n = {'solid': 0, 'layered': 0, 'molten': 0, None: 0}
    for cid, steps in chains.items():
        for s in steps:
            lab = classify(s)
            n[lab] = n.get(lab, 0) + 1
            if lab != 'layered':
                continue
            XS, XL, Tint = s.get('Mg_solid'), s.get('Mg_liquid'), s.get('T_interface')
            if XS is None or XL is None or XL < 0 or Tint is None:
                continue
            D = s['params']['BML_thickness']
            hs, hl = float(s['h_solid_km']), float(s['h_liquid_km'])
            P_int = float(M.pressure_mars(TRUE_CMB - D + hs))
            key = (round(P_int, 2), round(float(Tint)), round(XS, 3))
            if key not in cache:
                try:
                    cache[key] = rho_solid_ngibbs(M, P_int, float(Tint), XS)
                except Exception:
                    cache[key] = np.nan
            rS = cache[key]
            rL = M.liquid_bml_properties(P_int, float(Tint), XL)[0]
            if not (np.isfinite(rS) and np.isfinite(rL)):
                continue
            rec.append(dict(D=D, hs=hs, hl=hl, phi=hl/(hs+hl), P=P_int,
                            T=float(Tint), rho_S=rS, rho_L=rL,
                            drho=(rL - rS)*1000.0))

    print(f"\n  solid {n['solid']}   layered {n['layered']}   molten {n['molten']}"
          f"   unclassified {n[None]}")
    print(f'  usable layered samples: {len(rec)}')
    if not rec:
        print('nothing to do'); return

    phi = np.array([r['phi'] for r in rec])
    drho = np.array([r['drho'] for r in rec])
    rhoL = np.array([r['rho_L'] for r in rec])
    D_m = np.array([r['D'] for r in rec]) * 1e3

    print('\n' + '=' * 78)
    print('POSTERIOR (layered state)')
    print('=' * 78)
    q(phi, 'phi = f_liquid')
    q(drho, 'Delta_rho', 'kg/m3')
    q(np.array([r['rho_S'] for r in rec]), 'rho_S (nGibbs)', 'g/cm3')
    q(rhoL, 'rho_L (Thomas)', 'g/cm3')
    q(np.array([r['P'] for r in rec]), 'P at interface', 'GPa')
    q(D_m/1e3, 'D_bml', 'km')

    # ── regime diagnosis ────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('REGIME DIAGNOSIS  —  該用哪一套理論？')
    print('=' * 78)
    print(f'  P(phi > {RCMF_HI}) = {100*np.mean(phi > RCMF_HI):6.2f} %'
          f'   → 懸浮液，用 Stokes')
    print(f'  P(phi < {RCMF_LO}) = {100*np.mean(phi < RCMF_LO):6.2f} %'
          f'   → 有連通骨架，用 Darcy/compaction')
    print(f'  P({RCMF_LO} < phi < {RCMF_HI}) = '
          f'{100*np.mean((phi >= RCMF_LO) & (phi <= RCMF_HI)):6.2f} %  → 過渡區')
    dc = delta_c(phi)
    q(dc, 'delta_c (if Darcy)', 'm')
    q(D_m/dc, 'D / delta_c')
    print(f'  P(D/delta_c < 1) = {100*np.mean(D_m/dc < 1):.2f} %'
          f'   → 整層都在壓實邊界層內，McKenzie w0 本來就不適用')

    # ── settling ────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('SETTLING TIME   t = D / v      (a = 晶體半徑)')
    print('=' * 78)
    print(f"{'a(mm)':>6} {'mu':>5} {'n_RZ':>6} | "
          f"{'v_Stokes(m/yr)':>15} {'t_Stokes(yr)':>13} {'Re':>10} | "
          f"{'t_Darcy(yr)':>13} {'P(t>4.5Gyr)':>12}")
    for a_ in [3e-4, 1e-3, 3e-3]:
        for mu_ in [0.05, 0.1, 1.0]:
            for nrz in [4.65, 2.0]:
                vs = v_stokes_hindered(phi, drho, a=a_, mu=mu_, n_rz=nrz)
                ts = D_m / vs / SEC_YR
                re = reynolds(vs, a_, rhoL, mu_)
                vd = v_darcy(phi, drho, a=a_, mu=mu_)
                td = D_m / vd / SEC_YR
                print(f"{a_*1e3:6.1f} {mu_:5.2f} {nrz:6.2f} | "
                      f"{np.median(vs)*SEC_YR:15.4g} {np.median(ts):13.4g} "
                      f"{np.median(re):10.2e} | {np.median(td):13.4g} "
                      f"{100*np.mean(ts > AGE_MARS):11.2f}%")
    print('\n  Re << 1 才滿足 Stokes 的層流假設；Re ~ 1 要改用 Schiller-Naumann 修正')

    vs = v_stokes_hindered(phi, drho)
    ts = D_m / vs / SEC_YR
    print('\nreference (a = 1 mm, mu = 1 Pa s, n_RZ = 4.65):')
    q(vs * SEC_YR, 'v_settle', 'm/yr')
    q(ts, 't_settle', 'yr')
    q(reynolds(vs, 1e-3, rhoL, 1.0), 'Reynolds number')
    print(f'  P(t_settle > 4.5 Gyr) = {100*np.mean(ts > AGE_MARS):.3f} %')
    print(f'  P(t_settle > 1 Myr)   = {100*np.mean(ts > 1e6):.3f} %')

    np.savez(f'{args.out}.npz', phi=phi, drho=drho, D_m=D_m, rho_L=rhoL,
             rho_S=np.array([r['rho_S'] for r in rec]),
             P=np.array([r['P'] for r in rec]),
             T=np.array([r['T'] for r in rec]),
             v_settle=vs, t_settle=ts, delta_c=dc)
    print(f'\nsaved -> {args.out}.npz')

    # ── figure ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))

        ax[0].hist(phi, bins=50, color='#F58518')
        ax[0].axvspan(RCMF_LO, RCMF_HI, color='k', alpha=.18)
        ax[0].text(RCMF_HI + .01, ax[0].get_ylim()[1]*.85,
                   'disaggregation\nthreshold', fontsize=10)
        ax[0].set_xlabel(r'$\phi = f_{\rm liquid}$', fontsize=12)
        ax[0].set_ylabel('posterior count', fontsize=12)
        ax[0].set_title('後驗熔體分率 vs 瓦解門檻', fontsize=12)
        ax[0].tick_params(labelsize=14)

        ax[1].hist(drho, bins=50, color='#4C78A8')
        ax[1].axvline(0, color='k', lw=2)
        ax[1].set_xlabel(r'$\Delta\rho = \rho_L-\rho_S$ (kg m$^{-3}$)', fontsize=12)
        ax[1].set_title('固液密度差（>0 = 熔體下沉）', fontsize=12)
        ax[1].tick_params(labelsize=14)

        pg = np.logspace(-4, np.log10(0.95), 300)
        dm = np.median(drho)
        ax[2].loglog(pg, v_darcy(pg, dm)*SEC_YR, lw=2.6, color='#4C78A8',
                     label='Darcy (McKenzie B7)')
        ax[2].loglog(pg, v_stokes_hindered(pg, dm)*SEC_YR, lw=2.6, color='#E45756',
                     label='hindered Stokes')
        ax[2].axvspan(RCMF_LO, RCMF_HI, color='k', alpha=.18)
        ax[2].axvspan(np.percentile(phi, 5), np.percentile(phi, 95),
                      color='#F58518', alpha=.25)
        ax[2].text(np.median(phi), ax[2].get_ylim()[0]*3, 'posterior\n5–95%',
                   fontsize=10, ha='center', color='#B45309')
        ax[2].set_xlabel(r'$\phi$', fontsize=12)
        ax[2].set_ylabel('separation velocity (m/yr)', fontsize=12)
        ax[2].set_title('兩套理論在門檻附近接得上', fontsize=12)
        ax[2].tick_params(labelsize=14); ax[2].legend(fontsize=10)
        ax[2].grid(alpha=.3, which='both')

        ax[3].hist(np.log10(ts), bins=50, color='#54A24B')
        ax[3].axvline(np.log10(AGE_MARS), color='k', ls='--', lw=2, label='4.5 Gyr')
        ax[3].set_xlabel(r'$\log_{10}$ settling time (yr)', fontsize=12)
        ax[3].set_title('完全分離所需時間', fontsize=12)
        ax[3].tick_params(labelsize=14); ax[3].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{args.out}.png', dpi=150)
        print(f'saved -> {args.out}.png')
    except Exception as e:
        print('plotting skipped:', e)


if __name__ == '__main__':
    main()