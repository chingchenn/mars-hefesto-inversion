import json, glob, os, sys, shutil, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 載入你的 mcmc 模組（跟舊腳本一樣）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mcmc",
    "/home/jcchen2/mars-hefesto-inversion/15.mcmc_withBML_MOI.py")
mcmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcmc)

os.chdir('/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc')
os.makedirs('cache', exist_ok=True)

mcmc.load_gravity_profile()
khan_cache = mcmc.compute_khan_median()

# ── 參數 ──────────────────────────────────────
N_ENSEMBLE   = 30      # 抽幾個 posterior sample
BURN_IN_STEP = 200     # 前幾步不算

phases_to_plot = ['S-P', 'PP-P', 'SS-S', 'ScS-S']
SIGMA_dict = mcmc.SIGMA   # 用來 normalize
all_steps = []
for f in sorted(glob.glob('/net/beno3/data1/jcchen2/mars-hefesto-runs/mcmc/chain_a04*/chain.json')):
    with open(f) as fp:
        try:
            chain = json.load(fp)
        except Exception:
            continue
    all_steps.extend([s for s in chain if s.get('misfit', 999) < 10])

print(f"Total valid steps loaded: {len(all_steps)}")

best = min(all_steps, key=lambda x: x['misfit'])
print(f"Best misfit: {best['misfit']:.4f}")
print(f"Best params: {best['params']}")
# ── 收集 ensemble ─────────────────────────────
rng      = np.random.default_rng(0)
ensemble = [s for s in all_steps
            if s['accepted'] and s['step'] > BURN_IN_STEP]
sampled  = rng.choice(ensemble, size=min(N_ENSEMBLE, len(ensemble)),
                      replace=False)

# ── 對每個 ensemble member 計算殘差 ───────────
def get_residuals(params, model_name):
    """回傳 {event: {phase: pred-obs}} 或 None"""
    # 先檢查 cache
    cache_mantle = f'cache/{model_name}_mantle.npz'
    cache_bml    = f'cache/{model_name}_bml.npz'
    os.makedirs('cache', exist_ok=True)

    if os.path.exists(cache_mantle):
        d = np.load(cache_mantle)
        fd = {k: d[k] for k in d}
        bd = None
        if os.path.exists(cache_bml):
            d2  = np.load(cache_bml)
            bd  = {k: d2[k] for k in d2}
    else:
        tmp = tempfile.mkdtemp()
        fort56 = mcmc.run_hefesto(params, tmp)
        if fort56 is None:
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        fd = mcmc.read_fort56(fort56)
        np.savez(cache_mantle, **fd)

        # BML
        P_bottom = mcmc.P_BML_FIXED
        P_top    = P_bottom - 1.95
        bml_path = mcmc.run_hefesto_bml(
            params={'T_lit': params['T_bml'],
                    'P_lit': P_top, 'Mg#': params['Mg#_bml']},
            run_dir=tmp, P_top=P_top,
            P_bottom=P_bottom, T_bml=params['T_bml'])
        bd = None
        if bml_path:
            bml_raw = mcmc.read_fort56(bml_path)
            if bml_raw:
                bml_top_km = khan_cache['cmb_depth'] - mcmc.BML_THICKNESS
                bml_raw['depth_km'] += bml_top_km
                if float(bml_raw['rho'][0]) > float(fd['rho'][-1]):
                    bd = bml_raw
                    np.savez(cache_bml, **bd)
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        taup = mcmc.build_taup(fd, model_name, khan_cache, bml_data=bd)
    except Exception:
        return None

    # 計算殘差
    result = {}
    for event, obs in mcmc.SAMUEL_DATA.items():
        try:
            arr = taup.get_travel_times(
                source_depth_in_km=obs.get('depth', 10.0),
                distance_in_degree=obs['delta'],
                phase_list=['P','S','pP','sP','PP','PPP',
                            'SS','SSS','sS','ScS','SKS'])
        except Exception:
            continue
        times = {}
        for a in arr:
            if a.name not in times:
                times[a.name] = a.time
        def diff(a, b):
            return times.get(a, None) and times.get(b, None) and \
                   times[a] - times[b]
        pred = {
            'S-P':  diff('S','P'),   'PP-P':  diff('PP','P'),
            'SS-S': diff('SS','S'),  'ScS-S': diff('ScS','S'),
        }
        resid = {}
        for ph, pval in pred.items():
            if pval and ph in obs and obs[ph]:
                resid[ph] = pval - obs[ph]   # pred - obs
        result[event] = {'delta': obs['delta'], 'resid': resid}
    return result

# ── 畫圖 ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# best fit 殘差
best_resid = get_residuals(best['params'], 'best_fit')

for ax, phase in zip(axes.flatten(), phases_to_plot):
    sigma = SIGMA_dict.get(phase, 1.0)

    # ensemble（淡藍）
    for i, s in enumerate(sampled):
        ens_resid = get_residuals(s['params'], f'ens_{i:03d}')
        if ens_resid is None:
            continue
        xs = [v['delta'] for v in ens_resid.values()
              if phase in v['resid']]
        ys = [v['resid'][phase] for v in ens_resid.values()
              if phase in v['resid']]
        ax.scatter(xs, ys, color='steelblue', alpha=0.15,
                   s=15, zorder=2)

    # best fit（紅）
    if best_resid:
        xs = [v['delta'] for v in best_resid.values()
              if phase in v['resid']]
        ys = [v['resid'][phase] for v in best_resid.values()
              if phase in v['resid']]
        ax.scatter(xs, ys, color='red', s=60, zorder=5,
                   label='Best fit', marker='^')

    # 參考線
    ax.axhline(0,   color='k',    lw=1.2, ls='--')
    ax.axhline( sigma, color='gray', lw=0.8, ls=':')
    ax.axhline(-sigma, color='gray', lw=0.8, ls=':',
               label=f'±1σ ({sigma}s)')

    ax.set_xlabel('Epicentral distance (°)')
    ax.set_ylabel(f'Residual (pred−obs)  [s]')
    ax.set_title(f'{phase}  (n={len(xs)})')
    ax.legend(fontsize=8)

plt.suptitle(
    f'Travel time residual vs distance\n'
    f'best misfit={best["misfit"]:.4f}  '
    f'Mg#={best["params"]["Mg#"]:.3f}  '
    f'T_bml={best["params"]["T_bml"]:.0f}K',
    fontsize=11)
plt.tight_layout()
plt.savefig('residual_vs_distance.png', dpi=150, bbox_inches='tight')
print("Saved: residual_vs_distance.png")
